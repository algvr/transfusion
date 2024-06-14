from typing import Dict, List, Tuple, Optional
from torchvision.models.detection import *
from torchvision.models.detection.backbone_utils import *
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes, _resize_image_and_masks
from torchvision.models.detection.faster_rcnn import TwoMLPHead

import torch
from torch import nn, Tensor
import copy

from torchvision.ops import boxes as box_ops

from copy import deepcopy

from torch.nn.functional import interpolate

def is_torch_18v(torch_v):
    return torch_v == "1.8.1+cu101"


class NoNormTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std, size_divisible=32, fixed_size=None, min_size_pairs=None) -> None:
        if min_size_pairs is not None:
            self.multiscale_transform = True 
            self.no_choices = range(len(min_size_pairs))
        else:
            self.multiscale_transform = False
            
        self.min_size_pairs = min_size_pairs

        if is_torch_18v(torch.__version__):
            super().__init__(
                min_size,
                max_size,
                image_mean=image_mean,
                image_std=image_std,
                # size_divisible=size_divisible,
                # fixed_size=fixed_size,
            )
        else:
            super().__init__(
                min_size,
                max_size,
                image_mean=image_mean,
                image_std=image_std,
                size_divisible=size_divisible,
                fixed_size=fixed_size,
            )

    def resize(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if self.multiscale_transform:
            return self.multiscale_resize(image, target)
        else:
            return super().resize(image, target)

    def multiscale_resize(self,
               image: Tensor,
               target: Optional[Dict[str, Tensor]] = None,
               ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            idx = self.torch_choice(self.no_choices)
            size_h = self.min_size[idx]
            size_w = self.min_size_pairs[idx]
        else:
            # FIXME assume for now that testing uses the largest scale
            size_h = float(self.min_size[-1])
            size_w = float(self.min_size_pairs[-1])
        # image, target = _resize_image_and_masks(image, size_h, float(self.max_size), target, (size_h, size_w))
        scale_factors = (size_h/h, size_w/w)
        image = interpolate(image[None], mode="bilinear", align_corners=False, scale_factor=scale_factors)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def normalize(self, image):
        return image

    def postprocess(
        self,
        result,  # type: List[Dict[str, Tensor]]
        image_shapes,  # type: List[Tuple[int, int]]
        original_image_sizes,  # type: List[Tuple[int, int]]
    ):
        # if self.training:
        #     return result

        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

        return result

class RegionProposalNetworkWrapper(torch.nn.Module):
    def __init__(self, rpn_wrap):
        super().__init__()
        self.rpn_wrap = copy.deepcopy(rpn_wrap)

    def pre_nms_top_n(self):
        return self.rpn_wrap.pre_nms_top_n()

    def post_nms_top_n(self):
        return self.rpn_wrap.post_nms_top_n()

    def assign_targets_to_anchors(self, anchors, targets):
        return self.rpn_wrap.assign_targets_to_anchors(anchors, targets)

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        return self.rpn_wrap._get_top_n_idx(objectness, num_anchors_per_level)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        return self.rpn_wrap.filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        return self.rpn_wrap.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)

    def forward(
        self,
        images,  # type
        features,  # type: Dict[str, Tensor]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # RPN uses all feature maps that are available
        features = [v for k, v in features.items() if "hand_" not in k]
        objectness, pred_bbox_deltas = self.rpn_wrap.head(features)
        anchors = self.rpn_wrap.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.rpn_wrap.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.rpn_wrap.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.rpn_wrap.box_coder.encode(matched_gt_boxes, anchors)
        else:
            regression_targets = None
            labels = None

        # the loss is done outside because we want to be able to weight it and enable/disable obj loss
        # loss_objectness, loss_rpn_box_reg = self.compute_loss(
        #     objectness, pred_bbox_deltas, labels, regression_targets
        # )
        # losses = {
        #     "loss_objectness": loss_objectness,
        #     "loss_rpn_box_reg": loss_rpn_box_reg,
        # }

        return {
            "boxes": boxes,
            "objectness": objectness,
            "pred_bbox_deltas": pred_bbox_deltas,
            "labels": deepcopy(labels),
            "reg_targets": regression_targets,
        }
