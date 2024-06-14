import os
import torch
from torch import nn

from modeling.cross_fusion.cross_f_layers import TransposeLayer
from modeling.cross_fusion.cross_f_wrapper import CROSS_LAYER_ARGS, DEF_NARR_EMBED_ARGS
from modeling.cross_fusion.ego_fusion.cross_f_box_asymm import (
    AsymmetricCrossFModuleBox,
    AsymmetricCrossFTokenModuleBox,
)
from modeling.cross_fusion.ego_fusion.cross_f_box_layers import (
    CrossTransformerModuleBox,
    CrossTransformerTokenModule,
    SpaceTimeFusionModule,
)
from modeling.cross_fusion.ego_fusion.lm_layers import get_lm_layer
from modeling.cross_fusion.utils import PositionalEmbeddingLayer, RegroupPatchesLayerBox, patchify_image, get_visual_token_mask
from modeling.narration_embeds.narr_pooling_layers import get_narr_pooling_layer


MAX_NUM_PATCHES = 8192


def get_cross_box_encoder(cross_type, class_token_only):
    if cross_type == "cross_transformer":
        if class_token_only:
            return CrossTransformerTokenModule
        else:
            return CrossTransformerModuleBox
    elif cross_type == "space_time":
        return SpaceTimeFusionModule
    elif cross_type == "asymmetric":
        if class_token_only:
            return AsymmetricCrossFTokenModuleBox
        else:
            return AsymmetricCrossFModuleBox
    else:
        raise ValueError(f"{cross_type=} not implemented")


class CrossFusionBoxWrapper(nn.Module):
    def __init__(
        self, rcnn_model, cross_layer_args=CROSS_LAYER_ARGS, narr_embed_args=DEF_NARR_EMBED_ARGS, criterion=None
    ):
        super().__init__()
        self.rcnn_model = rcnn_model
        self.narr_embed_args = narr_embed_args
        # compatibility stuff when we could do only final_ln or nothing
        if "final_ln" in cross_layer_args["args"]:
            w_ln = cross_layer_args["args"].pop("final_ln")
            # del cross_layer_args["args"]["final_ln"]
            cross_layer_args["args"]["final_norm"] = "ln" if w_ln else False

        self.cross_encoder_args = cross_layer_args
        self.forward_language_f = self.cross_encoder_args.get("forward_language_f", False)
        self.vis_mask_type = self.cross_encoder_args.get("vis_mask_type", "global")

        self.dsampled_shapes = rcnn_model.get_dsampled_shapes()
        self.in_rgb_channels = rcnn_model.get_features_out_channels()
        self.fpn_features_idx = self.cross_encoder_args["fpn_features"][: len(self.dsampled_shapes)]
        self.vis_input_key = "image"
        self.token_dim = self.cross_encoder_args["args"]["input_f_size"]

        self.narr_pooling_layer = get_narr_pooling_layer(narr_embed_args["text_pooling"])(
            narr_embed_args, cross_layer_args["narr_out_mode"]
        )

        cross_fusion_encoders = self.setup_cross_fusion_encoders(self.cross_encoder_args)
        patches_to_token = self.setup_patches_to_token()
        tokens_to_features_layers = self.setup_token_to_features_layers()

        self.cross_fusion_encoders = nn.ModuleList(cross_fusion_encoders)
        self.patches_to_token = nn.ModuleList(patches_to_token)
        self.tokens_to_features = nn.ModuleList(tokens_to_features_layers)

        self.criterion = criterion
        if criterion.get("lm", None):
            self.lm_layer = get_lm_layer(self)
        self.lm_on = criterion.get("lm", False)
        self.use_lm_f = self.cross_encoder_args["lm_args"].get("use_lm_f", False)
        self.multi_lm = self.cross_encoder_args["lm_args"].get("multi", False) and self.lm_on and not self.use_lm_f

    def setup_cross_fusion_encoders(self, cross_encoder_args):
        cross_fusion_encoders = []
        cross_encoder_clzz = get_cross_box_encoder(
            cross_encoder_args["type"],
            class_token_only=cross_encoder_args["narr_out_mode"] == "embedding",
        )

        all_num_layers = cross_encoder_args["args"].pop("num_layers")
        if not isinstance(all_num_layers, list):
            all_num_layers = [all_num_layers] * len(self.dsampled_shapes)

        for i in range(len(self.dsampled_shapes)):
            pos_embedding_layer = PositionalEmbeddingLayer(
                cross_encoder_args["pos_embedding"], MAX_NUM_PATCHES, self.token_dim
            )

            if cross_encoder_args.get("lang_pos_embedding", False):
                lang_pos_embedding_layer = PositionalEmbeddingLayer(
                    cross_encoder_args["lang_pos_embedding"]["embedding_type"], 256, self.token_dim
                )
            else:
                lang_pos_embedding_layer = None

            cross_f_encoder = cross_encoder_clzz(
                no_patches=MAX_NUM_PATCHES,
                pos_embedding_layer=pos_embedding_layer,
                lang_pos_embedding=lang_pos_embedding_layer,
                num_layers=all_num_layers[i],
                **cross_encoder_args["args"],
            )
            cross_fusion_encoders.append(cross_f_encoder)

        return cross_fusion_encoders

    def setup_token_to_features_layers(self):
        tokens_to_features = []
        for i, shape in enumerate(self.dsampled_shapes):
            feature_h = shape[0]
            feature_w = shape[1]
            patch_h = self.cross_encoder_args["patch_h"][i]
            patch_w = self.cross_encoder_args["patch_w"][i]
            token_dim = self.token_dim
            out_channels = self.in_rgb_channels[i]

            tokens_to_features.append(
                RegroupPatchesLayerBox(
                    token_dim,
                    feature_h,
                    feature_w,
                    patch_h,
                    patch_w,
                    out_channels,
                    self.cross_encoder_args["backproj_dropout"],
                    self.cross_encoder_args.get("backproj_activ_f", None),
                )
            )

        return tokens_to_features

    def setup_patches_to_token(self):
        patches_to_token = []
        for i, shape in enumerate(self.dsampled_shapes):
            patch_h = self.cross_encoder_args["patch_h"][i]
            patch_w = self.cross_encoder_args["patch_w"][i]

            in_channels = self.in_rgb_channels[i]

            token_dim = self.token_dim
            patch_dim = in_channels * patch_w * patch_h
            patches_to_token.append(
                self.setup_patch_to_token(
                    self.cross_encoder_args["patch_norm"],
                    patch_dim,
                    token_dim,
                    in_channels=in_channels,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
            )

        return patches_to_token

    def forward(self, x, targets=None):
        visual_data = x[self.vis_input_key]

        features_dict = self.rcnn_model.forward_features(visual_data, targets)

        if self.cross_encoder_args["narr_out_mode"] == "embedding":
            language_f, att_w, _ = self.narr_pooling_layer(x["language_f"], pad_mask=False)
        else:
            language_f, att_w, att_mask = self.narr_pooling_layer(x["language_f"], pad_mask=True)

        mscale_l_features = []

        for i, key in enumerate(self.fpn_features_idx):
            key = str(key)

            self.tokens_to_features[i].init_h = features_dict["features"][key].shape[2]
            self.tokens_to_features[i].init_w = features_dict["features"][key].shape[3]

            vis_tokens = self.patches_to_token[i](features_dict["features"][key])
            vis_tokens_mask = get_visual_token_mask(vis_tokens.shape[2:], self.vis_mask_type)
            vis_tokens = patchify_image(vis_tokens, 1, 1)

            # if the att_mask is coming from hugging face we have to invert it
            if self.cross_encoder_args["narr_out_mode"] == "embedding":
                fused_features, fused_l_features, atts, _ = self.cross_fusion_encoders[i](
                    vis_tokens, language_f.unsqueeze(1), None, vis_tokens_mask=vis_tokens_mask
                )
            else:
                # for torch mask, true positions are ignored in attention, false positions are left as they are
                # used to ignore the padded positions, of shape B x seq_len
                fused_features, fused_l_features, atts, _ = self.cross_fusion_encoders[i](
                    vis_tokens, language_f, ~(att_mask.type(torch.bool)), vis_tokens_mask=vis_tokens_mask
                )

            if self.multi_lm:
                mscale_l_features.append(fused_l_features)

            if self.forward_language_f:
                if self.forward_language_f == "direct":
                    language_f = fused_l_features
                elif self.forward_language_f == "sum":
                    language_f += fused_l_features
                else:
                    raise NotImplementedError()
                
                
            fused_features = self.tokens_to_features[i](fused_features)
            features_dict["features"][key] = fused_features

        features_dict = self.rcnn_model.apply_fpn(features_dict)

        if "hand_boxes" in x:
            features_dict["hand_boxes"] = x["hand_boxes"]
        if "hand_poses" in x:
            features_dict["hand_poses"] = x["hand_poses"]

        rcnn_outs = self.rcnn_model.apply_rpn_roi_on_features(features_dict)

        if self.lm_on:
            fused_l_preds = self.lm_layer(
                mscale_l_features if self.multi_lm else fused_l_features if not self.use_lm_f else language_f,
                None if self.cross_encoder_args["narr_out_mode"] == "embedding" else att_mask.type(torch.bool),
            )
            rcnn_outs["lm"] = fused_l_preds

        return rcnn_outs

    def call_model_epoch_triggers(self, epoch):
        if epoch >= self.narr_embed_args["train_ep"] and self.narr_embed_args["train_ep"] != -1:
            self.narr_pooling_layer.unfreeze_embeddings()

        self.rcnn_model.call_model_epoch_triggers(epoch)

    def dets_from_outs(self, outs, orig_img_shapes=None, targets=None, hand_poses=None, hand_boxes=None):
        return self.rcnn_model.dets_from_outs(outs, orig_img_shapes, targets=targets, hand_poses=hand_poses, hand_boxes=hand_boxes)

    def forward_w_dets(self, x, targets=None):
        outs = self(x, targets)
        original_image_shapes = [tuple(img.shape[1:]) for img in x["image"]]
        dets = self.dets_from_outs(outs, orig_img_shapes=original_image_shapes, targets=targets, hand_poses=x.get("hand_poses"), hand_boxes=x.get("hand_boxes"))

        #for image_idx in range(len(dets)):
        #    frame_id = dets[image_idx]["frame_ids"]
        #    video_id = frame_id.rsplit("_", 1)[0]
        #    output_path = f"/local/home/agavryushin/transfusion_feature_outputs_wonton/v2/{video_id}/{frame_id}.pt"
        #    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #    torch.save({"detections": {k: (v.detach().cpu()[:10].half() if k == "box_features" else v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in dets[image_idx].items()},
        #                "targets": {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in targets[image_idx].items()},
        #                # "box_features": outs["roi_outputs"]["box_features"][:10].detach().cpu().half()
        #                # box features incorporated into "detections" to account for postprocessing
        #                },
        #                output_path)

        return dets

    def postprocess_detections(self, detections, proposals, image_sizes, original_image_shapes):
        return self.rcnn_model.postprocess_detections(detections, proposals, image_sizes, original_image_shapes)

    def compute_rpn_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        return self.rcnn_model.compute_rpn_loss(objectness, pred_bbox_deltas, labels, regression_targets)

    def setup_patch_to_token(self, patch_norm, patch_dim, token_dim, in_channels=None, patch_h=None, patch_w=None):
        if in_channels is not None:
            linear = nn.Conv2d(
                in_channels=in_channels,
                out_channels=token_dim,
                kernel_size=(patch_h, patch_w),
                stride=(patch_h, patch_w),
                bias=False,
            )
        else:
            linear = nn.Linear(patch_dim, token_dim)

        if patch_norm["visual"]:
            if patch_norm["visual"] == "layer1d":
                norm_layer = nn.LayerNorm(token_dim)
                module = nn.Sequential(linear, norm_layer)
            elif patch_norm["visual"] == "layer2d":
                raise NotImplementedError()
                # norm_layer = nn.LayerNorm((no_patches, token_dim))
                # module = nn.Sequential(linear, norm_layer)
            elif patch_norm["visual"] == "batch1d":
                norm_layer = nn.BatchNorm1d(token_dim)
                module = nn.Sequential(linear, TransposeLayer(1, 2), norm_layer, TransposeLayer(1, 2))
            else:
                raise ValueError(f"{patch_norm['visual']} strategy not known.")
            return module

        else:
            return linear

    def setup_lang_prenorm(self, patch_norm, token_dim):
        if patch_norm["language"] == "layer1d":
            self.lang_f_input_ln = nn.LayerNorm(token_dim)
        elif patch_norm["language"] is None:
            self.lang_f_input_ln = nn.Identity()
        else:
            raise ValueError(f"{patch_norm['language']} not recognized")


class CrossFusionBoxWrapperShared(CrossFusionBoxWrapper):
    def __init__(self, rcnn_model, cross_layer_args=CROSS_LAYER_ARGS, narr_embed_args=DEF_NARR_EMBED_ARGS):
        super().__init__(rcnn_model, cross_layer_args, narr_embed_args)

        self.cross_fusion_encoders = None
        del self.cross_fusion_encoders

        max_num_patches = -1
        for i, shape in enumerate(self.dsampled_shapes):
            feature_h = shape[0]
            feature_w = shape[1]
            patch_h = self.cross_encoder_args["patch_h"][i]
            patch_w = self.cross_encoder_args["patch_w"][i]
            num_patches = (feature_h // patch_h) * (feature_w // patch_w)
            max_num_patches = max(max_num_patches, num_patches)

        cross_encoder_clzz = get_cross_box_encoder(
            self.cross_encoder_args["type"], class_token_only=self.cross_encoder_args["narr_out_mode"] == "embedding"
        )
        pos_embedding_layer = PositionalEmbeddingLayer(
            self.cross_encoder_args["pos_embedding"], max_num_patches, self.token_dim
        )
        self.cross_encoder_args["args"]["classif_token"] = False
        self.cross_fusion_encoder = cross_encoder_clzz(
            no_patches=max_num_patches,
            pos_embedding_layer=pos_embedding_layer,
            **self.cross_encoder_args["args"],
        )

    def forward(self, x, targets=None):
        visual_data = x[self.vis_input_key]

        features_dict = self.rcnn_model.forward_features(visual_data, targets)

        if self.cross_encoder_args["narr_out_mode"] == "embedding":
            language_f, att_w, _ = self.narr_pooling_layer(x["language_f"], pad_mask=False)
        else:
            language_f, att_w, att_mask = self.narr_pooling_layer(x["language_f"], pad_mask=True)

        for i, key in enumerate(self.fpn_features_idx):
            key = str(key)

            patches = patchify_image(
                features_dict["features"][key],
                self.cross_encoder_args["patch_w"][i],
                self.cross_encoder_args["patch_h"][i],
            )
            vis_tokens = self.patches_to_token[i](patches)

            # if the att_mask is coming from huggingface we have to invert it
            if self.cross_encoder_args["narr_out_mode"] == "embedding":
                fused_features, fused_l_features, atts, _ = self.cross_fusion_encoder(
                    vis_tokens, language_f.unsqueeze(1), None
                )[0]
            else:
                fused_features, fused_l_features, atts, _ = self.cross_fusion_encoder(
                    vis_tokens, language_f, ~(att_mask.type(torch.bool))
                )

            fused_features = self.tokens_to_features[i](fused_features)
            if self.cross_encoder_args["replace_fpn_features"]:
                features_dict["features"][key] = fused_features

        if "hand_boxes" in x:
            features_dict["hand_boxes"] = x["hand_boxes"]
        if "hand_poses" in x:
            features_dict["hand_boxes"] = x["hand_poses"]
        features_dict = self.rcnn_model.apply_fpn(features_dict)
        return self.rcnn_model.apply_rpn_roi_on_features(features_dict), fused_l_features
