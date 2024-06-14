from copy import copy, deepcopy
from math import ceil
from typing import Iterable
import torch
from torch import nn
from torch.nn import functional as F

from modeling.cross_fusion.ego_fusion.cross_f_box_wrapper import CrossFusionBoxWrapper
from modeling.cross_fusion.utils import (
    RegroupPatchesLayerBox,
    patchify_image,
)
from modeling.cross_fusion.utils import PositionalEmbeddingLayer, RegroupPatchesLayerBox, patchify_image, get_visual_token_mask


class VisLangFusionBoxWrapper(CrossFusionBoxWrapper):
    def __init__(self, rcnn_model, cross_layer_args=..., narr_embed_args=..., criterion=None, vis_args=None):
        self.flow_fusion_args = deepcopy(cross_layer_args)
        
        super().__init__(rcnn_model, cross_layer_args, narr_embed_args, criterion)

        flow_args = cross_layer_args["flow_args"]
        self.vis_args = vis_args

        self.flow_fusion_args["positional_embedding"] = cross_layer_args["flow_pos_embedding"]
        
        self.flow_fusion_args["args"].update(flow_args)

        self.vis_fusion_encoders = nn.ModuleList(self.setup_cross_fusion_encoders(self.flow_fusion_args))
        self.token_dim = self.cross_encoder_args["args"]["input_f_size"]

        self.vis_in_features = vis_args["vis_in_features"]

        self.proj_module =  nn.Linear(
            in_features=self.vis_in_features,
            out_features=self.token_dim,
            bias=False,
        )

        self.pos_embedding_layer = PositionalEmbeddingLayer(
                self.flow_fusion_args["pos_embedding"],
                vis_args["num_frames"], 
                self.token_dim,
                temporal_dim=0
            )

    def forward(self, x, targets=None):
        visual_data = x[self.vis_input_key]
        bs = len(visual_data)

        features_dict = self.rcnn_model.forward_features(visual_data, targets)

        aux_vis_features = self.proj_module(F.normalize(x["visual_features"], dim=2))
        # we have bs x T x token_dm 
        if self.flow_fusion_args["temporal_embedding"]:
            aux_vis_patches = torch.flatten(self.pos_embedding_layer(aux_vis_features), 1, 2)
        else:
            aux_vis_patches = self.pos_embedding_layer(aux_vis_features)

        # early fusion vision with vision
        for i, key in enumerate(self.fpn_features_idx):
            key = str(key)
            
            self.tokens_to_features[i].init_h = features_dict["features"][key].shape[2]
            self.tokens_to_features[i].init_w = features_dict["features"][key].shape[3]

            vis_tokens = self.patches_to_token[i](features_dict["features"][key])
            # vis_tokens_mask = get_visual_token_mask(vis_tokens.shape[2:], self.vis_mask_type)
            vis_tokens = patchify_image(vis_tokens, 1, 1)

            fused_features, fused_aux_v_features ,atts, _ = self.vis_fusion_encoders[i](vis_tokens, aux_vis_patches, None)
            features_dict["features"][key] = fused_features

        # get the language features from the encoder
        if self.cross_encoder_args["narr_out_mode"] == "embedding":
            language_f, att_w, _ = self.narr_pooling_layer(x["language_f"], pad_mask=False)
        else:
            language_f, att_w, att_mask = self.narr_pooling_layer(x["language_f"], pad_mask=True)

        # keep going with the late cross modal fusion
        for i, key in enumerate(self.fpn_features_idx):
            key = str(key)

            vis_tokens = features_dict["features"][key]
            # if the att_mask is coming from hugging face we have to invert it
            if self.cross_encoder_args["narr_out_mode"] == "embedding":
                fused_features, fused_l_features, atts, _ = self.cross_fusion_encoders[i](vis_tokens, language_f.unsqueeze(1), None)[0]
            else:
                fused_features, fused_l_features, atts, _ = self.cross_fusion_encoders[i](
                    vis_tokens, language_f, ~(att_mask.type(torch.bool))
                )

            fused_features = self.tokens_to_features[i](fused_features)
            features_dict["features"][key] = fused_features

        if "hand_boxes" in x:
            features_dict["hand_boxes"] = x["hand_boxes"]
        if "hand_poses" in x:
            features_dict["hand_boxes"] = x["hand_poses"]
        features_dict = self.rcnn_model.apply_fpn(features_dict)
        return self.rcnn_model.apply_rpn_roi_on_features(features_dict)
