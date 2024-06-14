import importlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import BCEWithLogitsLoss, L1Loss, MSELoss, SmoothL1Loss

from data_preprocessing.utils.dataset_utils import MAX_STD
from runner.metrics_losses.hmap_metrics import t_unravel_index
from runner.metrics_losses.radam_optim import RAdam

opt_mapping = {"sgd": "SGD", "adam": "Adam", "rmsprop": "RMSprop", "adamw": "AdamW", "radam": "RAdam"}


def get_optimizer(opt_name):
    opt_cls = opt_mapping[opt_name]
    if opt_cls == "RAdam":
        clazz = RAdam
    else:
        module = importlib.import_module(f"torch.optim.{opt_cls.lower()}")
        clazz = getattr(module, opt_cls)
    return clazz


def get_hmap_criterion(criterion, pixel_w):
    if criterion.get("mae", 0):
        return w_loss(L1Loss(reduction="none"), pixel_w, criterion["agg"])
    elif criterion.get("smooth_mae", 0):
        return w_loss(SmoothL1Loss(reduction="none"), pixel_w, criterion["agg"])
    elif "mse" in criterion:
        return w_loss(MSELoss(reduction="none"), pixel_w, criterion["agg"])
    elif "ce" in criterion:
        return w_loss(BCEWithLogitsLoss(reduction="none"), pixel_w, criterion["agg"])
    elif "focal" in criterion:
        return w_loss(BCEWithLogitsLoss(reduction="none"), pixel_w, criterion["agg"])
    elif criterion.get("multivar_n", 0):
        return multivar_n_loss(criterion)
    elif "kl_div" in criterion:
        return w_loss(nn.KLDivLoss(reduction="none"), pixel_w, criterion["agg"])
    else:
        return w_loss(L1Loss(reduction="none"), pixel_w, criterion["agg"])


def multivar_n_loss(
    criterion,
):
    no_samples = criterion["no_samples"]

    def loss(distribution, gt_heatmap, *kwargs):
        samples = torch.multinomial(
            gt_heatmap.view([gt_heatmap.size(0), -1]), num_samples=no_samples, replacement=False
        )
        samples = t_unravel_index(samples, shape=gt_heatmap.shape[1:]).to(gt_heatmap.device)
        h, w = gt_heatmap.shape[1:]
        sy = h / w
        samples = samples.cpu().numpy()
        samples_y = np.interp(samples[:, :, 0], [0, h], [-MAX_STD * sy, MAX_STD * sy])
        samples_x = np.interp(samples[:, :, 1], [0, w], [-MAX_STD, MAX_STD])
        samples = torch.from_numpy(np.stack([samples_y, samples_x], axis=-1)).to(gt_heatmap.device).type(torch.float32)
        samples = torch.permute(samples, (1, 0, 2))
        return -distribution.log_prob(samples).mean()

    return loss


def w_loss(loss, pixel_w, agg):
    if pixel_w != 1 and pixel_w != "reg":
        fg, bg = 1 - 1 / (1 + pixel_w), 1 / (1 + pixel_w)

    def apply(preds, targets, fg_perc, bg_perc):
        bs = preds.shape[0]

        losses = loss(preds, targets)

        if pixel_w == "reg":
            targets = targets.view((bs, -1))
            losses = losses.view((bs, -1))
            fg_w = 1 - fg_perc
            bg_w = 1 - bg_perc

            losses = torch.where(targets > 0, losses * fg_w[:, None], losses * bg_w[:, None])
            # return losses.mean()

        elif pixel_w != 1:
            targets = targets.view((bs, -1))
            losses = losses.view((bs, -1))
            losses = torch.where(targets > 0, losses * fg, losses * bg)
            # return losses.mean()

        if agg == "sum":
            return losses.sum(axis=-1).mean()
        else:
            return losses.mean()

    return apply


def box_loss(class_logits, box_regression, _labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tensor
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        _labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(_labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / max(labels.numel(), 1)

    return box_loss
