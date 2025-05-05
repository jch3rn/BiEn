import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _l2_normalize(d):
    norm = torch.norm(d, p=2, dim=(1,2,3), keepdim=True)
    return d / (norm + 1e-8)


def get_r_adv_t(features, decoder1, decoder2, it=1, xi=1e-6, eps=10.0):
    decoder1.eval()
    decoder2.eval()
    x4 = features[-1].detach()
    d = torch.randn_like(x4, requires_grad=True)
    
    for _ in range(it):
        if d.grad is not None:
            d.grad.data.zero_()
        perturbed_x4 = x4 + xi * d
        perturbed_features = list(features)
        perturbed_features[-1] = perturbed_x4
        pred_hat = (decoder1(perturbed_features) + decoder2(perturbed_features)) / 2
        logp_hat = F.log_softmax(pred_hat, dim=1)
        with torch.no_grad():
            pred_orig = (decoder1(features) + decoder2(features)) / 2
            p_orig = F.softmax(pred_orig, dim=1)
        adv_distance = F.kl_div(logp_hat, p_orig, reduction='batchmean')
        adv_distance.backward(retain_graph=False)
        d.data = _l2_normalize(d.grad.data) * eps
    
    decoder1.train()
    decoder2.train()

    return d.data


class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        mask = (target != self.ignore_index).float()
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * target * mask)
        z_sum = torch.sum(score * score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class ConLoss(nn.Module):
    def __init__(self):
        super(ConLoss, self).__init__()
    
    def forward(self, reference_input, input1, input2, input3, temperature=0.1):
        positive_exp_sum = torch.exp(F.cosine_similarity(input1, input2) / temperature) + torch.exp(F.cosine_similarity(input1, input3) / temperature)
        negative_exp_sum = torch.exp(F.cosine_similarity(reference_input, input1) / temperature) + torch.exp(F.cosine_similarity(reference_input, input2) / temperature) + torch.exp(F.cosine_similarity(reference_input, input3) / temperature)

        loss = -torch.log(positive_exp_sum / (positive_exp_sum + negative_exp_sum))
        return loss.mean()
