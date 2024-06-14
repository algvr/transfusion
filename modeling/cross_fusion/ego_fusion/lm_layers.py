import torch
from torch import nn


def get_lm_layer(cross_fusion_box_wrapper):
    cross_encoder_args = cross_fusion_box_wrapper.cross_encoder_args
    # subtract 1 for the bg class
    no_nouns = cross_fusion_box_wrapper.rcnn_model.noun_classes - 1
    # if cross_fusion_box_wrapper.rcnn_model.verb_classes == (IGNORE_VERB_IDX_BG +1)
    # skip the bg verb class, use only 74 classes instead of 75 (idx 74 is bg)
    no_verbs = cross_fusion_box_wrapper.rcnn_model.verb_classes - 1

    if cross_encoder_args["lm_args"]["pooling"]["type"] in {"mean", "max"}:
        if cross_encoder_args["lm_args"].get("multi", False) == True:
            return MultiPoolPredictor(
                cross_encoder_args["lm_args"]["pooling"], cross_fusion_box_wrapper.token_dim, no_nouns, no_verbs
            )
        elif cross_encoder_args["lm_args"].get("multi", False) == "sep":
            return MultiPoolPredictorSep(
                cross_encoder_args["lm_args"]["pooling"], cross_fusion_box_wrapper.token_dim, no_nouns, no_verbs
            )
        else:
            return PoolPredictor(
                cross_encoder_args["lm_args"]["pooling"], cross_fusion_box_wrapper.token_dim, no_nouns, no_verbs
            )
    else:
        raise NotImplementedError


class PoolPredictor(nn.Module):
    def __init__(
        self,
        pooling_args,
        token_dim,
        no_nouns,
        no_verbs,
    ):
        super().__init__()

        self.pooling_args = pooling_args
        self.token_dim = token_dim
        self.repr_size = token_dim
        self.ln = None
        self.repr_mlp = None
        self.mlp_verb = None

        if self.pooling_args.get("ln", None):
            self.ln = nn.LayerNorm(token_dim)

        if self.pooling_args.get("repr_size", None):
            self.repr_mlp = nn.Sequential(nn.GELU(), nn.Linear(self.token_dim, pooling_args["repr_size"]))
            self.repr_size = pooling_args["repr_size"]

        # subtract 1 for the bg noun class
        self.mlp_noun = nn.Linear(self.repr_size, no_nouns)
        if no_verbs:
            self.mlp_verb = nn.Linear(self.repr_size, no_verbs)

    def forward(self, fused_l_tokens, att_mask=None):
        if att_mask is not None:
            fused_l_tokens = fused_l_tokens * att_mask.unsqueeze(2)

        if self.pooling_args["type"] == "max":
            features = fused_l_tokens.max(dim=1)[0]
        elif self.pooling_args["type"] == "mean":
            features = fused_l_tokens.mean(dim=1)

        if self.ln:
            features = self.ln(features)

        if self.repr_mlp:
            features = self.repr_mlp(features)

        noun_logits = self.mlp_noun(features)

        if self.mlp_verb:
            verb_logits = self.mlp_verb(features)
        else:
            verb_logits = None

        return {"noun_logits": noun_logits, "verb_logits": verb_logits}


class MultiPoolPredictor(PoolPredictor):
    """We performn the LM classification using the fused language features from all FPN scales"""

    def __init__(self, pooling_args, token_dim, no_nouns, no_verbs):
        super().__init__(pooling_args, token_dim, no_nouns, no_verbs)

    def forward(self, x, att_mask=None):
        no_scales = len(x)
        outs = []
        for i in range(no_scales):
            outs.append(super(MultiPoolPredictor, self).forward(x[i], att_mask))

        noun_logits = torch.stack([outs[i]["noun_logits"] for i in range(no_scales)]).mean(axis=0)
        if self.mlp_verb:
            verb_logits = torch.stack([outs[i]["verb_logits"] for i in range(no_scales)]).mean(axis=0)

        return {"noun_logits": noun_logits, "verb_logits": verb_logits}


class MultiPoolPredictorSep(nn.Module):
    """We perform the LM classification using the fused language features from all FPN scales"""

    def __init__(self, pooling_args, token_dim, no_nouns, no_verbs):
        super().__init__()
        predictors = []
        self.no_fpns = 3
        for _ in range(self.no_fpns):
            predictors.append(PoolPredictor(pooling_args, token_dim, no_nouns, no_verbs))
        self.predictors = nn.ModuleList(predictors)
        self.no_verbs = no_verbs

    def forward(self, x, att_mask=None):
        no_scales = len(x)
        outs = []
        for i in range(no_scales):
            outs.append(self.predictors[i].forward(x[i], att_mask))

        noun_logits = torch.stack([outs[i]["noun_logits"] for i in range(no_scales)]).mean(axis=0)
        if self.no_verbs:
            verb_logits = torch.stack([outs[i]["verb_logits"] for i in range(no_scales)]).mean(axis=0)

        return {"noun_logits": noun_logits, "verb_logits": verb_logits}
