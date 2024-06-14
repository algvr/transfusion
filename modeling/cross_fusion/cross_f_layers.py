import torch
from torch import nn

from modeling.cross_fusion.utils import (
    get_sin1d_embed,
    get_sin2d_embed,
    patchify_image,
)


class TransposeLayer(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class CrossTransformerModule(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_w,
        feature_h,
        patch_w,
        patch_h,
        patch_dropout,
        input_f_size,
        num_layers=2,
        num_heads=4,
        classif_token=False,
        back_to_img=True,
        fforward_multiplier=1,
        token_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        backproj_dropout=0.1,
        pos_embedding="learned",
        lang_to_hmap=True,
        patch_norm=False,
        final_ln=False,
    ):
        super().__init__()

        assert (
            feature_h % patch_h == 0 and feature_w % patch_w == 0
        ), "Image dimensions must be divisible by the patch size."
        self.num_patches = (feature_h // patch_h) * (feature_w // patch_w)
        patch_dim = in_channels * patch_w * patch_h
        self.feature_h = feature_h
        self.feature_w = feature_w
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.in_channels = in_channels
        self.classif_token = classif_token
        self.back_to_img = back_to_img
        self.back_to_img_fn = back_to_img_fn
        self.backproj_dropout = backproj_dropout
        self.pos_embedding_type = pos_embedding
        self.lang_to_hmap = lang_to_hmap
        self.patch_norm = patch_norm
        self.final_ln = final_ln
        self.no_pos_embeddings = self.num_patches + 1 + classif_token
        self.token_dim = input_f_size
        self.setup_pos_embedding()
        self.image_kind_embedding = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.lang_kind_embedding = nn.Parameter(torch.randn(1, 1, self.token_dim))

        self.heatmap_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        if self.classif_token:
            self.class_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.patch_dropout = nn.Dropout(patch_dropout)
        self.patches_to_token = self.setup_patch_to_token(self.patch_norm, patch_dim, self.token_dim, self.num_patches)
        self.register_buffer("padding_mask", torch.zeros(size=(1,), dtype=torch.bool))

        t_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            batch_first=True,
            dropout=token_dropout,
            activation=activ_f,
        )
        self.t_encoder = nn.TransformerEncoder(t_encoder_layer, num_layers)

        if self.final_ln:
            self.vis_ln = nn.LayerNorm((self.no_pos_embeddings - 1, self.token_dim))
            self.lang_ln = nn.LayerNorm(self.token_dim)

    def setup_patch_to_token(self, patch_norm, patch_dim, token_dim, no_patches):
        linear = nn.Linear(patch_dim, token_dim)
        patch_norm_type = patch_norm["type"]

        if patch_norm["both"]:
            self.lang_f_input_ln = nn.LayerNorm(token_dim)
        else:
            self.lang_f_input_ln = nn.Identity()

        if patch_norm_type:
            if patch_norm_type == "layer1d":
                norm_layer = nn.LayerNorm(token_dim)
                module = nn.Sequential(linear, norm_layer)
            elif patch_norm_type == "layer2d":
                norm_layer = nn.LayerNorm((no_patches, token_dim))
                module = nn.Sequential(linear, norm_layer)
            elif patch_norm_type == "batch1d":
                norm_layer = nn.BatchNorm1d(token_dim)
                module = nn.Sequential(linear, TransposeLayer(1, 2), norm_layer, TransposeLayer(1, 2))
            else:
                raise ValueError(f"{patch_norm_type} strategy not known.")
            return module
        else:
            return linear

    def setup_pos_embedding(self):
        if self.pos_embedding_type == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, self.no_pos_embeddings, self.token_dim))
        elif self.pos_embedding_type == "sin1d":
            pos_embedding = get_sin1d_embed(self.no_pos_embeddings, self.token_dim)
            self.register_buffer("pos_embedding", pos_embedding)
        elif self.pos_embedding_type == "sin2d":
            pos_embedding = get_sin2d_embed(
                self.feature_h // self.patch_h, self.feature_w // self.patch_w, self.token_dim
            )
            self.register_buffer("pos_embedding", pos_embedding)
        else:
            raise ValueError(f"{self.pos_embedding_type=} is not recognized")

    def forward(self, vis_features, language_f, language_f_att_maks):
        x = self.patches_to_token(patchify_image(vis_features, self.patch_w, self.patch_h))
        bs, n, _ = x.shape

        heatmap_token = torch.repeat_interleave(self.heatmap_token, repeats=bs, dim=0)
        x = torch.cat((heatmap_token, x), dim=1)
        if self.classif_token:
            class_token = torch.repeat_interleave(self.class_token, repeats=bs, dim=0)
            x = torch.cat((x, class_token), dim=1)

        x = x + self.pos_embedding
        x = x + self.image_kind_embedding
        x = self.patch_dropout(x)

        language_f = language_f + self.lang_kind_embedding
        x = torch.cat([x, language_f], dim=1)

        padding_mask = self.padding_mask.repeat(bs, x.shape[1])
        padding_mask = torch.cat([padding_mask, language_f_att_maks], dim=1)

        x, attentions = self.t_encoder(x, src_key_padding_mask=padding_mask, get_attentions=True)

        if self.back_to_img_fn == "token":
            hmap_token = x[:, 0]
        else:
            hmap_token = x[:, 1 : self.num_patches + 1]

        class_token = x[:, self.num_patches + 1]

        return hmap_token, class_token, attentions, None


class CrossTransformerTokenModule(CrossTransformerModule):
    """Used for cross fusion using just the CLS token from the bert encoder"""

    def __init__(
        self,
        in_channels,
        feature_w,
        feature_h,
        patch_w,
        patch_h,
        patch_dropout,
        input_f_size,
        num_layers=2,
        num_heads=4,
        classif_token=False,
        back_to_img=True,
        fforward_multiplier=1,
        token_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        backproj_dropout=0.1,
        pos_embedding="learned",
        lang_to_hmap=True,
    ):
        super().__init__(
            in_channels,
            feature_w,
            feature_h,
            patch_w,
            patch_h,
            patch_dropout,
            input_f_size,
            num_layers,
            num_heads,
            classif_token=False,
            back_to_img=back_to_img,
            fforward_multiplier=fforward_multiplier,
            token_dropout=token_dropout,
            back_to_img_fn=back_to_img_fn,
            activ_f=activ_f,
            backproj_dropout=backproj_dropout,
            pos_embedding=pos_embedding,
            lang_to_hmap=lang_to_hmap,
        )

    def forward(self, vis_features, language_f, language_f_att_maks):
        x = self.patches_to_token(patchify_image(vis_features, self.patch_w, self.patch_h))
        bs, n, _ = x.shape

        heatmap_token = torch.repeat_interleave(self.heatmap_token, repeats=bs, dim=0)
        x = torch.cat((heatmap_token, x), dim=1)
        x = x + self.pos_embedding
        x = x + self.image_kind_embedding
        x = self.patch_dropout(x)

        language_f = language_f + self.lang_kind_embedding
        x = torch.cat([x, language_f], dim=1)

        # No need for padding mask, we attend at everything
        x, attentions = self.t_encoder(x, get_attentions=True)

        if self.back_to_img_fn == "token":
            hmap_token = x[:, 0]
        else:
            hmap_token = x[:, 1 : self.num_patches + 1]
        class_token = x[:, -1]

        return hmap_token, class_token, attentions, None
