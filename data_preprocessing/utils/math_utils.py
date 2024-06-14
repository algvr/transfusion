import math
import logging
import numpy as np
import torch
from detectron2.structures.boxes import Boxes

HEATMAP_STD = 1


def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return (
        1.0
        / (2.0 * np.pi * sx * sy)
        * np.exp(-((x - mx) ** 2.0 / (2.0 * sx ** 2.0) + (y - my) ** 2.0 / (2.0 * sy ** 2.0)))
    )


def gaus2d_torch(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return (
        1.0
        / (2.0 * 3.14159 * sx * sy)
        * torch.exp(-((x - mx) ** 2.0 / (2.0 * sx ** 2.0) + (y - my) ** 2.0 / (2.0 * sy ** 2.0)))
    )


def custom_round(x, base=5):
    if isinstance(x, np.ndarray):
        return (base * (x.astype(float) / base).round()).astype(int)

    return int(base * round(float(x) / base))


def get_lin_space(img_w, img_h, max_std_h, max_std_w):
    mapped_img_x = np.interp(np.arange(img_w), [0, img_w], [-max_std_w, max_std_w])
    mapped_img_y = np.interp(np.arange(img_h), [0, img_h], [-max_std_h, max_std_h])
    return mapped_img_x, mapped_img_y


def get_img_heatmap(heatmap_type):
    logging.warn(f"Using {heatmap_type=} mode!")
    if heatmap_type == "gaussian":
        return get_gaussian_heatmap
    elif heatmap_type == "const":
        return get_constant_heatmap
    elif heatmap_type == "gaussian_dist":
        return get_gaussian_heatmap_dist
    else:
        raise ValueError(f"Heatmap {heatmap_type=} not supported")


def get_constant_heatmap(mapped_x, mapped_y, boxes, sx=HEATMAP_STD):
    x, y = np.meshgrid(mapped_x, mapped_y)
    hmap = np.zeros_like(x)

    for box in boxes:
        x0, y0, x1, y1 = box
        hmap[y0:y1, x0:x1] = 1

    return hmap


def get_gaussian_heatmap(mapped_x, mapped_y, boxes, sx=HEATMAP_STD):
    x, y = np.meshgrid(mapped_x, mapped_y)
    hmap = np.zeros_like(x)

    for box in boxes:
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.floor(x1), math.floor(y1)
        w = x1 - x0
        h = y1 - y0
        z = gaus2d(x, y, mx=mapped_x[x0 + w // 2], my=mapped_y[y0 + h // 2], sx=sx, sy=sx * h / w)
        hmap += z / z.max()

    return hmap / hmap.max()


def get_gaussian_heatmap_dist(mapped_x, mapped_y, boxes, sx=HEATMAP_STD):
    x, y = np.meshgrid(mapped_x, mapped_y)
    hmap = np.zeros_like(x)

    blend = 1 / len(boxes)

    for box in boxes:
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.floor(x1), math.floor(y1)
        w = x1 - x0
        h = y1 - y0
        z = gaus2d(x, y, mx=mapped_x[x0 + w // 2], my=mapped_y[y0 + h // 2], sx=sx, sy=h / w)
        hmap += z * blend

    hmap = hmap / hmap.sum()
    return hmap


def get_gaussian_mask_heatmap(mapped_x, mapped_y, boxes, sx=HEATMAP_STD):
    x, y = np.meshgrid(mapped_x, mapped_y)
    hmap = np.zeros_like(x)

    for box in boxes:
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = math.floor(x0), math.floor(y0), math.floor(x1), math.floor(y1)
        w = x1 - x0
        h = y1 - y0
        z = gaus2d(x, y, mx=mapped_x[x0 + w // 2], my=mapped_y[y0 + h // 2], sx=sx, sy=h / w)
        hmap += z / z.max()

    hmap = hmap / hmap.max()

    hmap = np.where(hmap - hmap.std() > 1e-5, hmap, 0)
    return hmap


def generate_heatmap_from_dist(dist_obj, mesh_x, mesh_y, max_norm):
    n = dist_obj.batch_shape[0]
    dist_mean = dist_obj.mean
    dist_std = dist_obj.stddev
    hmaps = []
    maxes = []
    for i in range(n):
        hmap_i = gaus2d_torch(
            mesh_x,
            mesh_y,
            mx=dist_mean[i, 1],
            my=dist_mean[i, 0],
            sx=dist_std[i, 1],
            sy=dist_std[i, 0],
        ).detach()
        hmaps.append(hmap_i)
        maxes.append(hmap_i.max())
    hmaps = torch.stack(hmaps, dim=0)

    if max_norm:
        maxes = torch.stack(maxes, dim=0)
        hmaps = hmaps / maxes[:, None, None]
    else:
        hmaps = hmaps / hmaps.sum(dim=(1, 2), keepdim=True)
    return hmaps


def get_boxes_area(boxes):
    torch_boxes = Boxes(np.array(boxes))
    return torch_boxes.area().sum().numpy()
