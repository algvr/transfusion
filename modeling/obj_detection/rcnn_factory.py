from modeling.obj_detection.dual_stream_rcnn_wrapper import DualStreamRCNNWrapper
from modeling.obj_detection.faster_rcnn_wrapper import FasteRCNNWrapper
import modeling.obj_detection.mobilenet_fpn_utils as mobilenet_fpn_utils
from modeling.obj_detection.roi_wrappers import (
    DualClassRoiHeadsWrapper,
    RoIHeadsWrapper,
)
from model_urls import model_urls
from modeling.obj_detection.utils import replace_frozen_bn
#from torchvision.models.detection.faster_rcnn import model_urls

import os
import sys
sys.path.append(os.path.dirname(__file__))
import torch
import torchvision
from torchvision.models.detection import *
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url


rcnn_dict = {
    "res50": mobilenet_fpn_utils.fasterrcnn_resnet50_fpn,
    "mobilenet": mobilenet_fpn_utils.fasterrcnn_mobilenet_v3_large_fpn,
    "mobilenet_320": mobilenet_fpn_utils.fasterrcnn_mobilenet_v3_large_320_fpn,
}

rcnn_weights_name = {
    "mobilenet_320": "fasterrcnn_mobilenet_v3_large_320_fpn_coco",
    "mobilenet": "fasterrcnn_mobilenet_v3_large_fpn_coco",
    "res50": "fasterrcnn_resnet50_fpn_coco",
}


def get_roi_heads_wrapper(verbs, verb_classifier_args=None):
    if verbs:
        if not verb_classifier_args or verb_classifier_args["type"] is False or verb_classifier_args["type"] is None:
            return DualClassRoiHeadsWrapper
        else:
            raise NotImplementedError
    else:
        return RoIHeadsWrapper


def get_rcnn_model(
    model_type,
    pretrained,
    trainable_backbone_layers,
    num_classes={"noun": 5, "verb": 5},
    box_1_dropout=0.0,
    box_2_dropout=0.0,
    classif_dropout=0.0,
    representation_size=1024,
    dual_stream=False,
    resize_spec=None,
    rcnn_kwargs={},
    load_fpn_rpn=True,
    fpn_layers_to_return=None,
    train_ep=0,
    batch_norm={"use": False},
    adapt_to_detectron=False,
    verb_classifier_args=None,
    ttc_on=False,
    ttc_hand_head_args=None,
    replace_heads=False,
    fpn_out_channels=256,
    additional_postprocessing=False,
    train_noun_verb_frequencies=None
):
    base_rcnn_clzz = rcnn_dict[model_type]
    box_predictor = None

    if isinstance(pretrained, str):
        print(f"Loading weights from {pretrained}")
        base_rcnn = base_rcnn_clzz(
            pretrained=False,
            trainable_backbone_layers=trainable_backbone_layers,
            box_predictor=box_predictor,
            returned_layers=fpn_layers_to_return,
            fpn_out_channels=fpn_out_channels,
            **rcnn_kwargs,
        )
        checkpoint = torch.load(pretrained, map_location="cpu")
        checkpoint["state_dict"] = {
            k.replace("model.rcnn_model.rcnn_to_wrap.", "")
            .replace("rpn.rpn_wrap.", "rpn.")
            .replace("roi_heads.roi_head_wrap.", "roi_heads.")
            .replace("roi_heads.roi_head_wrap.", "roi_heads."): v
            for k, v in checkpoint["state_dict"].items()
        }
        if not load_fpn_rpn:
            keys_to_eject = []
            for k in checkpoint["state_dict"].keys():
                if "rpn" in k or "roi" in k or "fpn" in k:
                    keys_to_eject.append(k)
            for k in keys_to_eject:
                checkpoint["state_dict"].pop(k)

        if fpn_layers_to_return != [1,2,3,4]:
            fpn_weights = {layer:w for layer,w in checkpoint["state_dict"].items() if "fpn" in layer}
            dif = 4 - len(fpn_layers_to_return)
            for i in range(len(fpn_layers_to_return)):
                checkpoint["state_dict"][f"backbone.fpn.inner_blocks.{i}.weight"] = fpn_weights[f"backbone.fpn.inner_blocks.{i+dif}.weight"]
                checkpoint["state_dict"][f"backbone.fpn.inner_blocks.{i}.bias"] = fpn_weights[f"backbone.fpn.inner_blocks.{i+dif}.bias"]

        base_rcnn.load_state_dict(checkpoint["state_dict"], strict=False)
        keys_to_save = [
            "roi_heads.noun_classifier.weight",
            "roi_heads.noun_classifier.bias",
            "roi_heads.verb_classifier.weight",
            "roi_heads.verb_classifier.bias",
            "roi_heads.box_regressor.1.weight",
            "roi_heads.box_regressor.1.bias",
        ]
        saved_checkpoint_params = {
            k: checkpoint["state_dict"][k] for k in keys_to_save if k in checkpoint["state_dict"].keys()
        }
        setattr(base_rcnn.roi_heads, "saved_checkpoint_params", saved_checkpoint_params)
    
    elif load_fpn_rpn:
        base_rcnn = base_rcnn_clzz(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            box_predictor=box_predictor,
            returned_layers=fpn_layers_to_return,
            **rcnn_kwargs,
        )
    else:
        weights_name = rcnn_weights_name[model_type]
        base_rcnn = base_rcnn_clzz(
            pretrained=False,
            trainable_backbone_layers=trainable_backbone_layers,
            box_predictor=box_predictor,
            returned_layers=fpn_layers_to_return,
            **rcnn_kwargs,
        )
        if faster_rcnn.model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=True)
        keys_to_eject = []
        for k in state_dict.keys():
            if "rpn" in k or "roi" in k or "fpn" in k:
                keys_to_eject.append(k)
        for k in keys_to_eject:
            state_dict.pop(k)
        base_rcnn.load_state_dict(state_dict, strict=False)

    roi_wrapper = get_roi_heads_wrapper(num_classes["verb"], verb_classifier_args)

    setattr(base_rcnn, "model_type", {"mobilenet_320": "mobilenet", "res50": "resnet"}.get(model_type, model_type))
    if batch_norm["use"]:
        base_rcnn = replace_frozen_bn(base_rcnn, batch_norm)

    if adapt_to_detectron:
        if model_type == "res50":
            base_rcnn.backbone.body.layer2[0].conv1.stride = (2, 2)
            base_rcnn.backbone.body.layer2[0].conv2.stride = (1, 1)

            base_rcnn.backbone.body.layer3[0].conv1.stride = (2, 2)
            base_rcnn.backbone.body.layer3[0].conv2.stride = (1, 1)

            base_rcnn.backbone.body.layer4[0].conv1.stride = (2, 2)
            base_rcnn.backbone.body.layer4[0].conv2.stride = (1, 1)

            orig_roi_align = torchvision.ops.roi_align
            torchvision.ops.roi_align = lambda orig_roi_align=orig_roi_align, *args, **kwargs: orig_roi_align(
                *args, **{**kwargs, "aligned": True}
            )

        base_rcnn.roi_heads.box_roi_pool.sampling_ratio = 0

    if not dual_stream:
        return FasteRCNNWrapper(
                base_rcnn,
                num_classes["noun"],
                num_classes["verb"],
                box_1_dropout,
                box_2_dropout,
                classif_dropout,
                representation_size,
                roi_wrapper,
                resize_spec=resize_spec,
                trainable_layers=trainable_backbone_layers,
                train_ep=train_ep,
                verb_classifier_args=verb_classifier_args,
                ttc_on=ttc_on,
                ttc_hand_head_args=ttc_hand_head_args,
                replace_heads=replace_heads,
                additional_postprocessing=additional_postprocessing,
                additional_postprocessing_frequencies=train_noun_verb_frequencies
            )
    else:
        return DualStreamRCNNWrapper(
            base_rcnn,
            num_classes["noun"],
            num_classes["verb"],
            box_1_dropout,
            box_2_dropout,
            classif_dropout,
            representation_size,
            roi_wrapper,
            resize_spec=resize_spec,
            trainable_layers=trainable_backbone_layers,
            train_ep=train_ep
        )
