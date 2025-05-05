import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/feto', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='feto/BiEn', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=1,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "endo" in dataset:
        ref_dict = {"1": 343, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "feto":
        ref_dict = {"1": 379, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    elif "lung":
        ref_dict = {"1": 256, "2": 128, "3": 52, "5": 168,
                    "9": 64, "18": 289, "25": 401, "32": 512}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    student_model = create_model()
    teacher_model = create_model()
    aux_teacher_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    teacher_model.train()
    student_model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(list(student_model.parameters()) + list(teacher_model.parameters()), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=255)
    dice_loss = losses.DiceLoss(num_classes)
    fc_loss = losses.ConLoss()
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    K = 1
    L = 1
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            labeled_volume_batch = volume_batch[:args.labeled_bs]
            # input perturbation
            weak_noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.1, 0.1)
            teacher_inputs = unlabeled_volume_batch + weak_noise
            strong_noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            aux_teacher_inputs = unlabeled_volume_batch + strong_noise
            # supervised training phase
            for k in range(K):
                labeled_outputs, _= teacher_model(labeled_volume_batch)
                labeled_outputs_soft = torch.softmax(labeled_outputs, dim=1)

                loss_ce = ce_loss(labeled_outputs,
                                label_batch[:][:args.labeled_bs].long())
                loss_dice = dice_loss(
                    labeled_outputs_soft, label_batch[:args.labeled_bs].unsqueeze(1))
                supervised_loss = 0.5 * (loss_dice + loss_ce)
                optimizer.zero_grad()
                supervised_loss.backward()
                optimizer.step()
                update_ema_variables(teacher_model, student_model, 0.98, iter_num)

            with torch.no_grad():
                aux_teacher_model.load_state_dict(teacher_model.state_dict())
            # unsupervised training phase
            for l in range(L):
                update_ema_variables(student_model, teacher_model, 0.95, iter_num)
                update_ema_variables(student_model, aux_teacher_model, 0.95, iter_num)

                teacher_outputs, teacher_embeddings = teacher_model(teacher_inputs)
                teacher_outputs_soft = torch.softmax(teacher_outputs, dim=1)
                with torch.no_grad():
                    aux_teacher_outputs, aux_teacher_embeddings = aux_teacher_model(aux_teacher_inputs)
                    aux_teacher_outputs_soft = torch.softmax(aux_teacher_outputs, dim=1)
                # feature perturbation
                features, _ = student_model.encoder(unlabeled_volume_batch)
                radv = losses.get_r_adv_t(
                    features,
                    teacher_model.decoder,
                    aux_teacher_model.decoder,
                    it=1, xi=1e-6, eps=2.0
                )
                student_model.decoder.enable_vat = True
                student_outputs, student_embeddings = student_model(unlabeled_volume_batch, radv=radv)
                student_model.decoder.enable_vat = False

                student_outputs_soft = torch.softmax(student_outputs, dim=1)
                _, student_embeddings_labeled= student_model(labeled_volume_batch)
                
                if iter_num < 1000:
                    sc_loss = 0.0
                else:
                    sc_loss = 0.5 * (torch.mean((student_outputs_soft-teacher_outputs_soft)**2) + torch.mean((teacher_outputs_soft-aux_teacher_outputs_soft)**2))
                consistency_weight = get_current_consistency_weight(iter_num//150)
                unsupervised_loss = consistency_weight * (sc_loss + fc_loss(student_embeddings_labeled, student_embeddings, teacher_embeddings, aux_teacher_embeddings))
                optimizer.zero_grad()
                unsupervised_loss.backward()
                optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('info/unsupervised_loss', unsupervised_loss, iter_num)

            logging.info(
                'iteration %d : supervised_loss : %f, unsupervised_loss: %f' %
                (iter_num, supervised_loss.item(), unsupervised_loss.item()))

            if iter_num > 0 and iter_num % 100 == 0:
                student_model.eval()
                cumulative_metrics = {cls: {'dice': [], 'hd95': []} for cls in range(1, num_classes)}
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], student_model, classes=num_classes)
                    for cls_idx, (dice, hd95) in enumerate(metric_i, start=1):
                        if not np.isnan(dice):
                            cumulative_metrics[cls_idx]['dice'].append(dice)
                        if not np.isnan(hd95):
                            cumulative_metrics[cls_idx]['hd95'].append(hd95)
                average_metrics = {}
                for cls_idx in range(1, num_classes):
                    average_metrics[cls_idx] = {}
                    for metric_name in ['dice', 'hd95']:
                        if cumulative_metrics[cls_idx][metric_name]:
                            average_metrics[cls_idx][metric_name] = np.mean(cumulative_metrics[cls_idx][metric_name])
                        else:
                            average_metrics[cls_idx][metric_name] = np.nan
                for cls_idx in range(1, num_classes):
                    writer.add_scalar(f'info/val_{cls_idx}_dice', average_metrics[cls_idx]['dice'], iter_num)
                    writer.add_scalar(f'info/val_{cls_idx}_hd95', average_metrics[cls_idx]['hd95'], iter_num)

                performance = np.nanmean([average_metrics[cls_idx]['dice'] for cls_idx in range(1, num_classes)])
                mean_hd95 = np.nanmean([average_metrics[cls_idx]['hd95'] for cls_idx in range(1, num_classes)])
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(student_model.state_dict(), save_mode_path)
                    torch.save(student_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : student_mean_dice : %f student_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                student_model.train()

                teacher_model.eval()
                cumulative_metrics = {cls: {'dice': [], 'hd95': []} for cls in range(1, num_classes)}
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], teacher_model, classes=num_classes)
                    for cls_idx, (dice, hd95) in enumerate(metric_i, start=1):
                        if not np.isnan(dice):
                            cumulative_metrics[cls_idx]['dice'].append(dice)
                        if not np.isnan(hd95):
                            cumulative_metrics[cls_idx]['hd95'].append(hd95)
                average_metrics = {}
                for cls_idx in range(1, num_classes):
                    average_metrics[cls_idx] = {}
                    for metric_name in ['dice', 'hd95']:
                        if cumulative_metrics[cls_idx][metric_name]:
                            average_metrics[cls_idx][metric_name] = np.mean(cumulative_metrics[cls_idx][metric_name])
                        else:
                            average_metrics[cls_idx][metric_name] = np.nan
                for cls_idx in range(1, num_classes):
                    writer.add_scalar(f'info/val_{cls_idx}_dice', average_metrics[cls_idx]['dice'], iter_num)
                    writer.add_scalar(f'info/val_{cls_idx}_hd95', average_metrics[cls_idx]['hd95'], iter_num)

                performance = np.nanmean([average_metrics[cls_idx]['dice'] for cls_idx in range(1, num_classes)])
                mean_hd95 = np.nanmean([average_metrics[cls_idx]['hd95'] for cls_idx in range(1, num_classes)])
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(teacher_model.state_dict(), save_mode_path)
                    torch.save(teacher_model.state_dict(), save_best)

                logging.info(
                    'iteration %d : teacher_mean_dice : %f teacher_mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                teacher_model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(teacher_model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
