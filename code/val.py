import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif gt.sum() == 0:
        return np.nan, np.nan
    else:
        shape = gt.shape
        max_coords = [dim - 1 for dim in shape]
        max_distance = np.sqrt(sum(coord**2 for coord in max_coords))
        return 0, max_distance


def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    slice = image[:, :]
    x, y = slice.shape[0], slice.shape[1]
    slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y, 1), order=0)
    input = torch.from_numpy(slice).permute(2, 0, 1).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = net(input)[0]
        out = torch.argmax(torch.softmax(
            out, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
