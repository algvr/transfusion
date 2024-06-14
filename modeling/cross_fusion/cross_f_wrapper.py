import torch
import torch.nn.functional as F
from torch import nn

from modeling.commons import NaoABC
from modeling.cross_fusion.cross_f_layers import CrossTransformerModule, CrossTransformerTokenModule
from modeling.cross_fusion.cross_qkv_layers import AsymmetricCrossFModule, AsymmetricCrossFTokenModule
from modeling.cross_fusion.utils import (
    BackProjectLayer,
    RegroupPatchesLayer,
    RegroupPatchesLayerExtra,
    RegroupPatchesGatedLayerExtra,
)
from modeling.narration_embeds.narr_pooling_layers import get_narr_pooling_layer

CROSS_LAYER_ARGS = {
    "type": "cross_transformer",
    "narr_out_mode": "embedding",
    # "narr_out_mode": "token_embeddings",
    "type": "asymmetric",
    "heatmap_upscale": 1,
    "class_upscale": 1,
    "extra": True,
    "args": {
        "patch_h": 2,
        "patch_w": 2,
        "patch_dropout": 0.2,
        # "num_layers": 2,
        # "token_dropout": 0.1,
        "vis_dropout": 0.1,
        "lang_dropout": 0.1,
        "lang_layers": 2,
        "vis_layers": 3,
        "num_heads": 2,
        "fforward_multiplier": 2,
        # "back_to_img_fn": "token",
        "back_to_img_fn": "regroup_gated",
        "pos_embedding": "learned",
        "activ_f": "gelu",
    },
}

DEF_DF_LAYER_ARGS = {
    "type": "attention",
    "args": {"input_f_size": 100, "filter_size": 128, "downscale": 2},
}

DEF_NARR_EMBED_ARGS = {
    "size": 300,
    "finetune": True,
    "num_heads": 2,
    "lang_dropout": 0.2,
    "text_pooling": "self_attention",
}


def get_cross_layer(cross_type, class_token_only):
    if cross_type == "cross_transformer":
        if class_token_only:
            return CrossTransformerTokenModule
        else:
            return CrossTransformerModule
    elif cross_type == "asymmetric":
        if class_token_only:
            return AsymmetricCrossFTokenModule
        else:
            return AsymmetricCrossFModule
    else:
        raise ValueError(f"{cross_type=} not implemented")


class CrossFNaoABC(NaoABC):
    def __init__(self):
        super().__init__()

    def last_stage(self, features, im_size):
        x = features[0]
        noun_logits, verb_logits, ttc_pred = self.classif_heads(features[1])

        if not self.multivar_n:
            x = self.upsample_layer(x, im_size)

        if self.kl_div:
            x_shape = x.size()
            x = F.log_softmax(x.view(x_shape[0], -1), dim=-1).reshape(x_shape)

        if self.multivar_n:
            dist = self.forward_multivar(x)
            return {"heatmap": dist, "noun_logits": noun_logits, "verb_logits": verb_logits, "ttc": ttc_pred}

        return {"heatmap": x.squeeze(dim=1), "noun_logits": noun_logits, "verb_logits": verb_logits, "ttc": ttc_pred}

    def get_final_dsampled_size(self):
        return self.to_wrap.get_final_dsample_size()

    def get_hmap_token_postprocess(
        self,
        back_to_img_fn,
        token_dim,
        feature_h,
        feature_w,
        backproj_dropout,
        patch_h,
        patch_w,
        extra,
        h_upscale_dim,
        elu_last,
    ):
        if back_to_img_fn == "token":
            if self.back_to_img:
                hmap_token_postprocess = BackProjectLayer(token_dim, feature_h, feature_w, backproj_dropout)
            else:
                hmap_token_postprocess = nn.Identity()

        elif back_to_img_fn == "regroup":
            if extra:
                hmap_token_postprocess = RegroupPatchesLayerExtra(
                    token_dim, h_upscale_dim, feature_h, feature_w, patch_h, patch_w, backproj_dropout, elu_last
                )
            else:
                hmap_token_postprocess = RegroupPatchesLayer(
                    token_dim, feature_h, feature_w, patch_h, patch_w, backproj_dropout
                )
        elif back_to_img_fn == "regroup_gated":
            if extra:
                hmap_token_postprocess = RegroupPatchesGatedLayerExtra(
                    token_dim, h_upscale_dim, feature_h, feature_w, patch_h, patch_w, backproj_dropout
                )
            else:
                pass

        elif not self.back_to_img:
            hmap_token_postprocess = nn.Identity()
        else:
            raise ValueError(f"{self.back_to_img=} strategy not recognized")

        return hmap_token_postprocess


class CrossFusionWrapper(CrossFNaoABC):
    def __init__(
        self, nao_model, back_to_img=True, cross_layer_args=CROSS_LAYER_ARGS, narr_embed_args=DEF_NARR_EMBED_ARGS
    ):
        super().__init__()

        dsampled_shape = nao_model.get_final_dsampled_shape()
        cross_layer_args["args"]["feature_h"] = dsampled_shape[0]
        cross_layer_args["args"]["feature_w"] = dsampled_shape[1]

        cross_layer_args["args"]["token_dim"] = narr_embed_args["size"]
        cross_layer_args["args"]["classif_token"] = nao_model.is_classifying()

        nao_model.setup_multivar(in_features=cross_layer_args["args"]["token_dim"])

        self.cross_layer_args = cross_layer_args
        self.narr_embed_args = narr_embed_args
        self.nao_model = nao_model
        self.vis_input_key = self.nao_model.get_vis_input_key()
        self.back_to_img = back_to_img
        self.token_dim = self.cross_layer_args["args"]["token_dim"]
        h_upscale_dim = int(cross_layer_args["heatmap_upscale"] * self.token_dim)
        c_upscale_dim = int(cross_layer_args["class_upscale"] * self.token_dim)

        self.narr_pooling_layer = get_narr_pooling_layer(narr_embed_args["text_pooling"])(
            narr_embed_args, cross_layer_args["narr_out_mode"]
        )
        self.cross_layer = get_cross_layer(
            cross_layer_args["type"], class_token_only=self.cross_layer_args["narr_out_mode"] == "embedding"
        )(
            in_channels=nao_model.get_features_out_channels(),
            back_to_img=back_to_img,
            backproj_dropout=self.nao_model.run_config["hmap_dropout"],
            **cross_layer_args["args"],
        )
        del self.nao_model.heatmap_head

        self.back_to_img_fn = cross_layer_args["args"]["back_to_img_fn"] if self.back_to_img else "identity"
        self.hmap_token_postprocess = self.get_hmap_token_postprocess(
            self.back_to_img_fn,
            token_dim=cross_layer_args["args"]["token_dim"],
            feature_h=cross_layer_args["args"]["feature_h"],
            feature_w=cross_layer_args["args"]["feature_w"],
            backproj_dropout=self.nao_model.run_config["hmap_dropout"],
            patch_h=cross_layer_args["args"]["patch_h"],
            patch_w=cross_layer_args["args"]["patch_w"],
            extra=cross_layer_args["extra"],
            h_upscale_dim=h_upscale_dim,
            elu_last=cross_layer_args["elu_last"],
        )

        self.class_token_postprocess = nn.Identity()
        nao_model.setup_classifiers(in_features=self.token_dim)

        if cross_layer_args["extra"]:
            if nao_model.is_classifying() or nao_model.is_ttc_pred_on():
                self.class_token_postprocess = nn.Sequential(
                    nn.Linear(self.token_dim, c_upscale_dim),
                    nn.GELU(),
                    nn.LayerNorm(c_upscale_dim),
                )
                self.nao_model.setup_classifiers(in_features=c_upscale_dim)

    def call_model_epoch_triggers(self, epoch):
        if epoch >= self.narr_embed_args["train_ep"] and self.narr_embed_args["train_ep"] != -1:
            self.unfreeze_embeddings()
        self.nao_model.call_model_epoch_triggers(epoch)

    def unfreeze_embeddings(self):
        self.narr_pooling_layer.unfreeze_embeddings()

    def set_get_attentions(self, get_attentions):
        self.cross_layer.get_attentions = get_attentions

    def forward(self, x):
        visual_data = x[self.vis_input_key]
        im_size = visual_data.shape[-2:]
        vis_features = self.nao_model.forward_features(visual_data)

        language_f = x["language_f"]
        # if the att_mask is coming from hugging face we have to invert it
        if self.cross_layer_args["narr_out_mode"] == "embedding":
            language_f, att_w, _ = self.narr_pooling_layer(language_f, pad_mask=False)
            fused_features = self.cross_layer(vis_features, language_f.unsqueeze(1), None)[:2]
        else:
            language_f, att_w, att_mask = self.narr_pooling_layer(language_f, pad_mask=True)
            fused_features = self.cross_layer(vis_features, language_f, ~(att_mask.type(torch.bool)))[:2]

        x, noun_logits, verb_logits, ttc_pred = super().classif_branch_and_heatmap(fused_features)
        return self.nao_model.last_stage(x, noun_logits, verb_logits, im_size, ttc_pred)

    def heatmap_from_features(self, x):
        heatmap = self.hmap_token_postprocess(x[0], x[1])
        return heatmap

    def classif_branch(self, x):
        return self.class_token_postprocess(x[1])

    def classif_heads(self, x):
        return self.nao_model.classif_heads(x)

    def get_heatmap_channels(self):
        return self.nao_model.get_features_out_channels()

    def is_classifying(self):
        return self.nao_model.is_classifying()

    def is_ttc_pred_on(self):
        return self.nao_model.is_ttc_pred_on()

    def is_heatmap_pred_on(self):
        return self.nao_model.is_heatmap_pred_on()

    def get_classify_noun(self):
        return self.nao_model.classify_noun

    def get_classify_verb(self):
        return self.nao_model.classify_verb
