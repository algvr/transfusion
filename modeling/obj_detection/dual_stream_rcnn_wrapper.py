from types import MethodType
from typing import List, OrderedDict, Tuple
import torch
from torchvision.models.detection import *

from modeling.obj_detection.faster_rcnn_wrapper import FasteRCNNWrapper
from modeling.obj_detection.roi_wrappers import RoIHeadsWrapper
from modeling.obj_detection.wrapper_utils import \
    RegionProposalNetworkWrapper


class DualStreamRCNNWrapper(FasteRCNNWrapper):
    def __init__(
        self,
        rcnn_to_wrap,
        noun_classes=91,
        verb_classes=None,
        box_1_dropout=0,
        box_2_dropout=0,
        classif_dropout=0,
        representation_size=1024,
        roi_heads_wrapper_cls=RoIHeadsWrapper,
        rpn_wrapper_cls=RegionProposalNetworkWrapper,
    ):
        super().__init__(
            rcnn_to_wrap,
            noun_classes,
            verb_classes,
            box_1_dropout,
            box_2_dropout,
            classif_dropout,
            representation_size,
            roi_heads_wrapper_cls,
            rpn_wrapper_cls,
        )
        rcnn_to_wrap.forward = MethodType(dual_stream_rcnn_forward, rcnn_to_wrap)

    def forward(self, x, targets=None):
        images = x["image"]
        flow_data = x["flow_data"]
        outs = self.rcnn_to_wrap(images, flow_data, targets)
        return outs


def dual_stream_rcnn_forward(self, images, flow_data, targets=None):
    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")

    if self.training:
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError(
                        "Expected target boxes to be a tensor" "of shape [N, 4], got {:}.".format(boxes.shape)
                    )
            else:
                raise ValueError("Expected target boxes to be of type " "Tensor, got {:}.".format(type(boxes)))

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    " Found invalid box {} for target at index {}.".format(degen_bb, target_idx)
                )

    features = self.backbone(images.tensors, flow_data)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    rpn_proposals = self.rpn(images, features, targets)
    roi_outputs = self.roi_heads(features, rpn_proposals["boxes"], images.image_sizes, targets)

    outputs = {}

    outputs["roi_outputs"] = roi_outputs
    outputs["proposals"] = rpn_proposals
    outputs["image_sizes"] = images.image_sizes
    outputs["original_image_sizes"] = original_image_sizes

    return outputs
