from typing import Tuple
import numpy as np

import torch
from torchmetrics import MeanAbsoluteError, Metric


class MAEwithNorm(MeanAbsoluteError):
    def __init__(self, metric_norm, **kwargs) -> None:
        super().__init__(**kwargs)
        self.metric_norm = metric_norm

    def update(self, preds, target):
        if self.metric_norm:
            # target_max = torch.max(torch.amax(target, dim=(-1, -2), keepdim=True), torch.Tensor([1e-6]))
            target_max = torch.amax(target, dim=(-1, -2), keepdim=True)
            target = target / target_max
            preds = preds / target_max

        return super().update(preds, target)


class HeatmapAccuracy(Metric):
    def __init__(self, metric_norm, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.metric_norm = metric_norm
        self.add_state("diffs_cum_sum", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))

        self.persistent(False)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        if self.metric_norm:
            # target_max = torch.max(torch.amax(target, dim=(-1, -2), keepdim=True), torch.Tensor([1e-6]))
            target_max = torch.amax(target, dim=(-1, -2), keepdim=True)
            target_norm = target / target_max
            preds_norm = preds / target_max

            if torch.any(torch.isnan(target_norm)):
                print(f"Nan found at target_norm in norm mae metric")
                print(f"Target max:{target_max}")
                print(f"Target :{target}")
                print(f"Target norm:{target_norm}")
                exit(1)

            if torch.any(torch.isnan(preds_norm)):
                print(f"Nan found at preds_norm in norm mae metric")
                print(f"Target max:{target_max}")
                exit(1)

        else:
            target_norm = target
            preds_norm = preds

        target_area = torch.sum(target_norm > target_norm.std(dim=[-1, -2], unbiased=True, keepdim=True))

        if torch.any(torch.isnan(target_area)):
            print(f"Nan found at target area in norm mae metric")
            exit(1)

        self.diffs_cum_sum += torch.sum(torch.abs(preds_norm - target_norm))
        self.total += target_area

    def compute(self):
        return self.diffs_cum_sum / self.total


def t_unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


class CoordMetric(Metric):
    def __init__(self, img_h_w, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("diffs_cum_sum", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))
        self.img_h_w = torch.Tensor(img_h_w).type(torch.float32)
        self.persistent(False)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        n = preds.shape[0]
        _, target_argmaxs = torch.max(target.view(n, -1), dim=-1)
        _, preds_argmaxs = torch.max(preds.view(n, -1), dim=-1)

        target_coords = t_unravel_index(target_argmaxs, target.shape[1:])
        preds_coords = t_unravel_index(preds_argmaxs, preds.shape[1:])

        dists = torch.max(torch.abs(target_coords - preds_coords) / self.img_h_w, dim=-1)[0]

        self.diffs_cum_sum += torch.sum(dists)
        self.total += preds.shape[0]

    def compute(self):
        return self.diffs_cum_sum / self.total
