import argparse
import os
import shutil
from PIL import Image
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/feto', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='feto/BiEn', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=1,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95, asd
    elif gt.sum() == 0:
        return np.nan, np.nan, np.nan
    else:
        shape = gt.shape
        max_coords = [dim - 1 for dim in shape]
        max_distance = np.sqrt(sum(coord**2 for coord in max_coords))
        return 0, max_distance, max_distance


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    slice = image[:, :]
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (256 / x, 256 / y, 1), order=0)
    input = torch.from_numpy(slice).permute(2, 0, 1).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urds":
            out_main, _, _, _ = net(input)
        else:
            out_main = net(input)[0]
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / 256, y / 256), order=0)
        prediction = pred

    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))

    image_uint8 = (image * 255).astype(np.uint8)
    img = Image.fromarray(image_uint8)
    img.save(os.path.join(test_save_path, f"{case}_img.png"))
    pred_uint8 = prediction.astype(np.uint8)
    pred_img = Image.fromarray(pred_uint8)
    pred_img.save(os.path.join(test_save_path, f"{case}_pred.png"))
    label_uint8 = label.astype(np.uint8)
    label_img = Image.fromarray(label_uint8)
    label_img.save(os.path.join(test_save_path, f"{case}_gt.png"))
    return metric_list


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=3,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    num_classes = FLAGS.num_classes
    dice_list = [[] for _ in range(num_classes - 1)]
    hd95_list = [[] for _ in range(num_classes - 1)]
    asd_list = [[] for _ in range(num_classes - 1)]
    for case in tqdm(image_list):
        metric_list = test_single_volume(case, net, test_save_path, FLAGS)
        for class_idx, (dice, hd95, asd) in enumerate(metric_list):
            if not np.isnan(dice):
                dice_list[class_idx].append(dice)
            if not np.isnan(hd95):
                hd95_list[class_idx].append(hd95)
            if not np.isnan(asd):
                asd_list[class_idx].append(asd)
    avg_dice_per_class = [
        np.nanmean(dice) if len(dice) > 0 else 0.0
        for dice in dice_list
    ]
    avg_hd95_per_class = [
        np.nanmean(hd95) if len(hd95) > 0 else 0.0
        for hd95 in hd95_list
    ]
    avg_asd_per_class = [
        np.nanmean(asd) if len(asd) > 0 else 0.0
        for asd in asd_list
    ]
    overall_avg_dice = np.nanmean(avg_dice_per_class) if avg_dice_per_class else 0.0
    overall_avg_hd95 = np.nanmean(avg_hd95_per_class) if avg_hd95_per_class else 0.0
    overall_avg_asd = np.nanmean(avg_asd_per_class) if avg_asd_per_class else 0.0
    avg_metric = [avg_dice_per_class, avg_hd95_per_class, avg_asd_per_class, [overall_avg_dice, overall_avg_hd95, overall_avg_asd]]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print("Average Dice for each class:", metric[0])
    print("Average HD95 for each class:", metric[1])
    print("Average ASD for each class:", metric[2])
    print("Overall average metrics:", metric[3])
