import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from modeling.layers.upsample_layers import get_upsample_l
from data_preprocessing.utils.dataset_utils import MAX_STD
from data_preprocessing.utils.math_utils import get_lin_space


def get_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def get_dnn(in_features, nr_classes, dropout=0.2, activ_f=None):
    seq = [
        nn.Dropout(dropout),
        nn.Linear(in_features, nr_classes),
    ]

    if activ_f == "relu":
        seq.append(nn.ReLU())

    elif activ_f == "softplus":
        seq.append(nn.Softplus())

    return nn.Sequential(*seq)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not isinstance(m, torch.nn.modules.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)
    else:
        for param in m.parameters():
            param.requires_grad_(True)


class NaoABC(nn.Module):
    def __init__(self):
        super().__init__()

    def get_final_dsampled_size(self):
        raise NotImplementedError()

    def get_final_dsampled_shape(self):
        raise NotImplementedError()

    def get_orig_size(self):
        raise NotImplementedError()

    def get_vis_input_key(self):
        return self.vis_input_key

    def forward_features(self, x):
        raise NotImplementedError()

    def classif_branch(self, x):
        raise NotImplementedError()

    def heatmap_from_features(self, x):
        raise NotImplementedError()

    def classif_heads(self, x):
        raise NotImplementedError()

    def call_model_epoch_triggers(self, epoch):
        print(f"No epoch trigers for {type(self)}")

    def setup_classifiers(self, in_features):
        if self.classify_noun:
            self.noun_classifier = get_dnn(in_features, self.noun_classes, dropout=self.class_dropout)

        if self.classify_verb:
            self.verb_classifier = get_dnn(in_features, self.verb_classes, dropout=self.class_dropout)

        if self.ttc_pred_on:
            self.ttc_predictor = get_dnn(in_features, 1, dropout=self.ttc_dropout, activ_f="softplus")

    def setup_multivar(self):
        if self.multivar_n:
            self.final_dsampled_size = self.get_final_dsampled_size()

            h, w = np.array(self.run_config["resize_spec"]) // 8
            mapped_x, mapped_y = get_lin_space(w, h, MAX_STD * h / w, MAX_STD)
            mesh_x, mesh_y = np.meshgrid(mapped_x, mapped_y, indexing="xy")
            mesh_x, mesh_y = torch.from_numpy(mesh_x).type(torch.float32), torch.from_numpy(mesh_y).type(torch.float32)
            self.register_buffer("mesh_x_8", mesh_x, persistent=True)
            self.register_buffer("mesh_y_8", mesh_y, persistent=True)

            self.dist_dropout = nn.Dropout(self.run_config["hmap_dropout"])
            self.dist_mlp = nn.Linear(in_features=self.final_dsampled_size, out_features=5)
            self.offset_stds = nn.Parameter(torch.FloatTensor([1, 1]))

    def forward_multivar(self, x):
        x = self.dist_dropout(x)
        outputs = self.dist_mlp(x.view([x.size(0), -1]))
        loc, tril, diag = outputs[:, :2], outputs[:, 3], outputs[:, 3:]
        diag = 1 + self.offset_stds + nn.functional.elu(diag)
        z = torch.zeros(size=[loc.size(0)], device=x.device)
        scale_tril = torch.stack([diag[:, 0], z, tril, diag[:, 1]], dim=-1).view(-1, 2, 2)
        # scale_tril is a tensor of size [batch, 2, 2]
        dist = MultivariateNormal(loc=loc, scale_tril=scale_tril)
        return dist

    def forward(self, x):
        x = x[self.vis_input_key]
        im_size = x.shape[-2:]
        features = self.forward_features(x)
        x, noun_logits, verb_logits, ttc_pred = self.classif_branch_and_heatmap(features)
        return self.last_stage(x, noun_logits, verb_logits, im_size, ttc_pred)

    def classif_branch_and_heatmap(self, features):
        x = None
        if self.is_heatmap_pred_on():
            x = self.heatmap_from_features(features)

        if self.is_classifying() or self.is_ttc_pred_on():
            features = self.classif_branch(features)

        noun_logits, verb_logits, ttc_pred = self.classif_heads(features)

        return x, noun_logits, verb_logits, ttc_pred

    def last_stage(self, x, noun_logits, verb_logits, im_size, ttc_pred):
        if self.w_sigmoid:
            x = torch.sigmoid(x)

        if self.is_heatmap_pred_on():
            if not self.multivar_n:
                x = self.upsample_layer(x, im_size)
            else:
                dist = self.forward_multivar(x)
                return {"heatmap": dist, "noun_logits": noun_logits, "verb_logits": verb_logits, "ttc": ttc_pred}

            return {
                "heatmap": x.squeeze(dim=1),
                "noun_logits": noun_logits,
                "verb_logits": verb_logits,
                "ttc": ttc_pred,
            }

        else:
            return {"heatmap": x, "noun_logits": noun_logits, "verb_logits": verb_logits, "ttc": ttc_pred}

    def is_classifying(self):
        return self.classifying

    def is_heatmap_pred_on(self):
        return self.heatmap_pred_on

    def is_ttc_pred_on(self):
        return self.ttc_pred_on

    def get_classify_noun(self):
        return self.classify_noun

    def get_classify_verb(self):
        return self.classify_verb


class NaoWrapperBase(NaoABC):
    """Base Nao wrapper class that holds functionalities for all models that will be used in the nao experiments
    -> requires heatmap prediction, with different output function (sigmoid, clipping, raw etc)
    -> run classification heads for noun or verb
    """

    def __init__(self, to_wrap, noun_classes, verb_classes, criterion, w_sigmoid, finetune, run_config) -> None:
        super().__init__()
        self.to_wrap = to_wrap
        self.classify_noun = criterion["noun"] > 0
        self.classify_verb = criterion["verb"] > 0
        self.ttc_pred_on = criterion["ttc"] > 0
        self.classifying = self.classify_noun or self.classify_verb
        self.noun_classes = noun_classes
        self.verb_classes = verb_classes
        self.w_sigmoid = w_sigmoid
        self.finetune = finetune
        self.kl_div = "kl_div" in criterion
        self.run_config = run_config
        self.class_dropout = run_config["class_dropout"]
        self.ttc_dropout = run_config["ttc_dropout"]
        self.max_norm = run_config.get("heatmap_type", "const") != "gaussian_dist"
        self.multivar_n = criterion.get("multivar_n", 0)
        self.epoch = 0
        self.heatmap_pred_on = is_heatmap_pred_on(criterion)
        self.upsample_layer = get_upsample_l(run_config.get("upsample_kind", "bilinear"))(1)

        self.setup_multivar()

    def classif_heads(self, features):
        if self.classify_noun:
            noun_logits = self.noun_classifier(features)
        else:
            noun_logits = None

        if self.classify_verb:
            verb_logits = self.verb_classifier(features)
        else:
            verb_logits = None

        if self.ttc_pred_on:
            ttc_pred = torch.squeeze(self.ttc_predictor(features), 1)
        else:
            ttc_pred = None

        return noun_logits, verb_logits, ttc_pred

    def heatmap_from_features(self, features):
        return self.heatmap_head(features)


def get_heatmap_norm(norm_type):
    if norm_type == None:
        return lambda x: x
    if norm_type == "l2":
        return lambda x: F.normalize(
            x.view(x.shape[0], -1),
            p=2.0,
        ).view(x.shape)
    else:
        raise ValueError(f"{norm_type=} not implemented")


def is_heatmap_pred_on(criterion):
    return bool(criterion.get("mae", 0) or criterion.get("mse", 0) or criterion.get("multivar_n", 0))
