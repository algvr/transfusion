import os
from types import MethodType
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import *

from modeling.commons import freeze_all_but_bn
from modeling.obj_detection.roi_wrappers import COND_VERB_ARGS, RoIHeadsWrapper
from modeling.obj_detection.wrapper_utils import *
from modeling.ttc_pred import TTCPredictionHead


DSAMPLE_TIMES = {
    "resnet": np.array([2, 3, 4, 5]),
}
MIN_TTC = 0.251

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class FasteRCNNWrapper(nn.Module):
    def __init__(
        self,
        rcnn_to_wrap,
        noun_classes=91,
        verb_classes=None,
        box_1_dropout=0.0,
        box_2_dropout=0.0,
        classif_dropout=0.0,
        representation_size=1024,
        roi_heads_wrapper_cls=RoIHeadsWrapper,
        rpn_wrapper_cls=RegionProposalNetworkWrapper,
        resize_spec=None,
        trainable_layers=0,
        train_ep=0,
        verb_classifier_args=COND_VERB_ARGS,
        ttc_on=False,
        ttc_hand_head_args=None,
        replace_heads=False,
        additional_postprocessing=False,
        additional_postprocessing_frequencies=None
    ):
        super().__init__()
        self.rcnn_to_wrap = rcnn_to_wrap
        self.model_type = rcnn_to_wrap.model_type
        old_transform = rcnn_to_wrap.transform
        self.vis_input_key = "image"
        self.dsample_factors = 2 ** DSAMPLE_TIMES[self.model_type]
        self.resize_spec = resize_spec
        self.trainable_layers = trainable_layers

        if isinstance(self.resize_spec[0], list):
            self.img_size = resize_spec[0][-1]
        else:
            self.img_size = resize_spec[0]

        self.train_ep = train_ep
        self.ttc_on = ttc_on
        self.ttc_hand_head_args = ttc_hand_head_args
        self.noun_classes = noun_classes
        self.verb_classes = verb_classes

        if is_torch_18v(torch.__version__):
            rcnn_to_wrap.transform = NoNormTransform(
                min_size=resize_spec[0],
                max_size=old_transform.max_size,
                image_mean=old_transform.image_mean,
                image_std=old_transform.image_std,
                min_size_pairs=resize_spec[1] if isinstance(resize_spec[1], list) else None,
            )
        else:
            rcnn_to_wrap.transform = NoNormTransform(
                min_size=resize_spec[0],
                max_size=old_transform.max_size,
                image_mean=old_transform.image_mean,
                image_std=old_transform.image_std,
                size_divisible=old_transform.size_divisible,
                fixed_size=old_transform.fixed_size,
                min_size_pairs=resize_spec[1] if isinstance(resize_spec[1], list) else None
            )

        self.representation_size = representation_size
        assert noun_classes or verb_classes

        self.box_dropout = nn.Dropout(box_2_dropout) if box_2_dropout else nn.Identity()

        if noun_classes:
            box_regressor = nn.Sequential(self.box_dropout, nn.Linear(representation_size, 4 * noun_classes))
            noun_classifier = nn.Linear(representation_size, noun_classes)
        else:
            box_regressor = nn.Sequential(self.box_dropout, nn.Linear(representation_size, 4 * noun_classes))
            noun_classifier = None

        if verb_classes:
            verb_classifier = nn.Linear(representation_size, verb_classes)
            self.classify_verb = True
        else:
            verb_classifier = None
            self.classify_verb = False

        self.additional_postprocessing = additional_postprocessing
        rcnn_to_wrap.roi_heads = roi_heads_wrapper_cls(
            rcnn_to_wrap.roi_heads,
            box_1_dropout,
            classif_dropout,
            box_regressor,
            noun_classifier,
            verb_classifier,
            representation_size,
            verb_classifier_args,
            ttc_on,
            ttc_hand_head_args,
            replace_heads,
            additional_postprocessing,
            additional_postprocessing_frequencies
        )
        rcnn_to_wrap.rpn = rpn_wrapper_cls(rcnn_to_wrap.rpn)
        rcnn_to_wrap.forward = MethodType(rcnn_forward, rcnn_to_wrap)
        rcnn_to_wrap.forward_features = MethodType(rcnn_forward_features, rcnn_to_wrap)
        rcnn_to_wrap.apply_fpn = MethodType(rcnn_apply_fpn, rcnn_to_wrap)

        self.max_ttc_boxes_per_sample = ttc_hand_head_args["max_ttc_boxes_per_image"]
        self.rcnn_to_wrap.backbone.body.apply(freeze_all_but_bn)

    def forward(self, x, targets=None):
        images = x[self.vis_input_key]
        outs = self.rcnn_to_wrap(images, targets)
        return outs

    def unfreeze_features(self, trainable_layers):
        if self.model_type == "resnet":
            assert 0 <= trainable_layers <= 5
            layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]

            if trainable_layers == 5:
                layers_to_train.append("bn1")

            for name, parameter in self.rcnn_to_wrap.backbone.body.named_parameters():
                if any([name.startswith(layer) for layer in layers_to_train]):
                    parameter.requires_grad_(True)

        elif self.model_type == "mobilenet":
            # code to freeze mobilenet stages
            stage_indices = [0, 2, 4, 7, 13, 16, 17]
            num_stages = 6

            # find the index of the layer from which we wont freeze
            assert 0 <= trainable_layers <= num_stages
            freeze_before = 17 if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

            string_ids = [str(x) for x in np.arange(freeze_before, 17)]
            for id in string_ids:
                b = self.rcnn_to_wrap.backbone.body[id]
                # for b in [:freeze_before]:
                for parameter in b.parameters():
                    parameter.requires_grad_(True)
        else:
            raise NotImplementedError(f"Layer freezing is not implemented for {self.model_type} wrapper")

    def forward_features(self, images, targets=None):
        # outs here is features (from visual backbone), images, targets, original_im_sizes (needed for bbox remap)
        # features is ordered dict used for fpn reasons
        features_dict = self.rcnn_to_wrap.forward_features(images, targets)
        return features_dict

    def apply_fpn(self, features_dict):
        # forwards the feature dict with FPN stage
        features_dict["features"] = self.rcnn_to_wrap.apply_fpn(features_dict["features"])
        return features_dict

    def apply_rpn_roi_on_features(self, features_dict):
        images, features, targets, orig_im_sizes = (
            features_dict["images"],
            features_dict["features"],
            features_dict["targets"],
            features_dict["orig_im_sizes"],
        )

        if "hand_boxes" in features_dict:
            features["hand_boxes"] = features_dict["hand_boxes"]
        if "hand_poses" in features_dict:
            features["hand_poses"] = features_dict["hand_poses"]

        rpn_proposals = self.rcnn_to_wrap.rpn(images, features, targets)
        # here I need hand_pos entry in the features ordered dict
        roi_outputs = self.rcnn_to_wrap.roi_heads(features,
                                                  rpn_proposals["boxes"],
                                                  images.image_sizes, targets)
        outputs = {}

        outputs["roi_outputs"] = roi_outputs
        outputs["proposals"] = rpn_proposals
        outputs["image_sizes"] = images.image_sizes
        outputs["original_image_sizes"] = orig_im_sizes

        return outputs

    def forward_w_dets(self, x, targets=None):
        outs = self(x, targets)
        original_image_shapes = [tuple(img.shape[1:]) for img in x["image"]]
        return self.dets_from_outs(outs, orig_img_shapes=original_image_shapes)

    def dets_from_outs(self, outs, orig_img_shapes=None, targets=None, hand_poses=None, hand_boxes=None):
        with torch.no_grad():
            image_sizes = outs["image_sizes"]
            # proposals = outs["proposals"]
            detections = outs["roi_outputs"]
            if orig_img_shapes:
                postprocessed_dets = self.postprocess_detections(detections, detections["proposals"], image_sizes, orig_img_shapes, targets=targets)
            else:
                postprocessed_dets = self.postprocess_detections(
                    detections, detections["proposals"], image_sizes, outs["original_image_sizes"], targets=targets
                )
        
        if self.ttc_on and isinstance(self.rcnn_to_wrap.roi_heads.ttc_pred_layer, TTCPredictionHead):
            # concat, then distribute again
            box_features_list = []
            object_box_list = []
            hand_boxes_list = []
            hand_poses_list = []
            orig_shape_list = []
            sample_num_boxes = []
            for idx, det in enumerate(postprocessed_dets):
                box_features = det["box_features"][:self.max_ttc_boxes_per_sample]
                det["idxs"] = det["idxs"][:self.max_ttc_boxes_per_sample]
                box_features_list.append(box_features)
                object_box_list.append((det["boxes"][:self.max_ttc_boxes_per_sample]
                                        / torch.tensor([ [orig_img_shapes[idx][1], orig_img_shapes[idx][0], orig_img_shapes[idx][1], orig_img_shapes[idx][0]] ], device=det["boxes"].device)))
                B = box_features.shape[0]
                sample_num_boxes.append(B)
                if orig_img_shapes is not None:
                    orig_shape_list.extend([orig_img_shapes[idx]] * B)
                if hand_boxes is not None:
                    hand_boxes_list.extend([hand_boxes[idx:idx+1]] * B)
                if hand_poses is not None:
                    hand_poses_list.extend([hand_poses[idx:idx+1]] * B)

            if sum(sample_num_boxes) > 0:
                box_features_cat = torch.cat(box_features_list, dim=0)
                object_box_cat = torch.cat(object_box_list, dim=0)
                hand_boxes_cat = torch.cat(hand_boxes_list, dim=0)
                hand_poses_cat = torch.cat(hand_poses_list, dim=0)

                ttc_prelim = self.rcnn_to_wrap.roi_heads.ttc_pred_layer({"box_features": box_features_cat,
                                                                         "object_boxes": object_box_cat,
                                                                         "hand_boxes":   hand_boxes_cat,
                                                                         "hand_poses":   hand_poses_cat,
                                                                         "orig_shapes":  orig_shape_list})
                ttc_interim = F.softplus(ttc_prelim)
                if self.training or not self.additional_postprocessing:
                    ttc_post = ttc_interim
                else:
                    ttc_post = torch.maximum(torch.full_like(ttc_interim, MIN_TTC), ttc_interim)

                acc = 0
                for idx, increment in enumerate(sample_num_boxes):
                    postprocessed_dets[idx]["ttcs"] = ttc_post[acc:acc+increment]
                    acc += increment
        else:
            if not self.training and self.additional_postprocessing:
                for idx, det in enumerate(postprocessed_dets):
                    postprocessed_dets[idx]["ttcs"] = torch.maximum(torch.full_like(postprocessed_dets[idx]["ttcs"], MIN_TTC), postprocessed_dets[idx]["ttcs"])

        return postprocessed_dets

    def compute_rpn_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        return self.rcnn_to_wrap.rpn.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)

    def call_model_epoch_triggers(self, epoch):
        if epoch >= self.train_ep and self.train_ep != -1:
            self.unfreeze_features(self.trainable_layers)
            print("Unfroze model parameters")

    def postprocess_detections(self, detections, proposals, image_sizes, original_image_shapes, targets=None):
        if self.classify_verb and self.ttc_on:
            idxs, frame_ids, boxes, box_features, scores, nouns, verbs, ttcs = self.rcnn_to_wrap.roi_heads.postprocess_detections(
                detections["class_logits"],
                detections["verb_logits"],
                detections["ttcs"],
                detections["box_regression"],
                detections["box_features"],
                proposals,
                image_sizes,
                targets
            )

        elif self.classify_verb:
            idxs, frame_ids, boxes, box_features, scores, nouns, verbs, _ = self.rcnn_to_wrap.roi_heads.postprocess_detections(
                detections["class_logits"],
                detections["verb_logits"],
                detections["ttcs"],
                detections["box_regression"],
                detections["box_features"],
                proposals,
                image_sizes,
                targets
            )

        else:
            idxs, frame_ids, boxes, box_features, scores, nouns = self.rcnn_to_wrap.roi_heads.postprocess_detections(
                detections["class_logits"], detections["box_regression"], detections["box_features"], proposals, image_sizes, targets
            )

        result = []
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "idxs": idxs[i],
                    "box_features": box_features[i],
                    "frame_ids": frame_ids[i],  # one frame ID for this image
                    "boxes": boxes[i],
                    "nouns": nouns[i],
                    "verbs": verbs[i] if self.classify_verb else nouns[i],
                    "ttcs": ttcs[i] if self.ttc_on else scores[i],
                    "scores": scores[i],
                }
            )

        detections = self.rcnn_to_wrap.transform.postprocess(result, image_sizes, original_image_shapes)

        # try painting boxes
        #from PIL import Image, ImageDraw
        #idx = 0
        #img = Image.open(f"/local/home/agavryushin/Datasets/Ego4d/object_frames/{detections[idx]['frame_ids']}.jpg")
        #draw = ImageDraw.Draw(img)
        #for box_idx in range(5):
        #    draw.rectangle(detections[idx]['boxes'][box_idx].detach().cpu().float().tolist(), width=3, outline="#f00")
        #img.save("/local/home/agavryushin/checkpoint_resume_test.png")
        
        return detections

    def get_vis_input_key(self):
        return self.vis_input_key

    def get_features_out_channels(self):
        channels = []
        fpn_mod = self.rcnn_to_wrap.backbone.fpn
        for block in fpn_mod.inner_blocks:
            channels.append(getattr(block, "in_channels", block._modules["0"].in_channels if "0" in block._modules else block.in_channels))
        return channels

    def get_fpn_features_out_channels(self):
        channels = []
        fpn_mod = self.rcnn_to_wrap.backbone.fpn
        for block in fpn_mod.layer_blocks:
            channels.append(block.out_channels)
        return channels

    def get_dsampled_shapes(self):
        if self.model_type == "resnet":
            return_layers = [int(layer_id) for layer_id in self.rcnn_to_wrap.backbone.body.return_layers.values()]
        elif self.model_type == "mobilenet":
            return_layers = [int(layer_id) for layer_id in self.rcnn_to_wrap.backbone.body.return_layers]
        else:
            raise NotImplementedError()

        # needed because the model resizes to self.img_size by itself
        if isinstance(self.resize_spec[0], list):
            ratio_to_img_size = self.resize_spec[0][-1] / self.img_size
            specs = [self.resize_spec[0][-1], self.resize_spec[1][-1]]
            final_h_w = (np.array(specs) * ratio_to_img_size).astype(int)
        else:
            ratio_to_img_size = self.resize_spec[0] / self.img_size
            final_h_w = (np.array(self.resize_spec) * ratio_to_img_size).astype(int)

        return [final_h_w // dsample_factor for dsample_factor in self.dsample_factors[return_layers]]


def rcnn_forward_features(self, images, targets=None):
    """ "Here we forward until the fpn module, which is done independently. Needed to more easily assign features in the dfn"""
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

    features = self.backbone.body(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    return {"features": features, "images": images, "targets": targets, "orig_im_sizes": original_image_sizes}


def rcnn_apply_fpn(self, ord_features_dict):
    ord_features_dict = self.backbone.fpn(ord_features_dict)
    return ord_features_dict


def rcnn_forward(self, images, targets=None):
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

    features = self.backbone(images.tensors)
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
