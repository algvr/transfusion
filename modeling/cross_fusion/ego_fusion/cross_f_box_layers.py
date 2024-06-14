import torch
from torch import nn

from modeling.cross_fusion.ego_fusion.torch18_adapters import TransformerEncoderLayerBFirst
from modeling.obj_detection.wrapper_utils import is_torch_18v

if is_torch_18v(torch.__version__):
    transformer_encoder_clzz = TransformerEncoderLayerBFirst
else:
    transformer_encoder_clzz = nn.TransformerEncoderLayer


class CrossTransformerModuleBox(nn.Module):
    def __init__(
        self,
        no_patches,
        patch_dropout,
        input_f_size,
        pos_embedding_layer,
        num_layers=2,
        num_heads=4,
        classif_token=False,
        fforward_multiplier=2,
        token_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        patch_norm=False,
        final_norm=False,
        lang_pos_embedding=None,
    ):
        super().__init__()
        self.no_patches = no_patches
        self.classif_token = classif_token
        self.back_to_img_fn = back_to_img_fn
        self.patch_norm = patch_norm
        self.final_norm = final_norm
        self.token_dim = input_f_size
        self.pos_embedding_layer = pos_embedding_layer
        self.image_kind_embedding = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.lang_kind_embedding = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.lang_pos_embedding = lang_pos_embedding

        self.heatmap_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        if self.classif_token:
            self.class_token = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.patch_dropout = patch_dropout

        self.register_buffer("padding_mask", torch.zeros(size=(1,), dtype=torch.bool))

        t_encoder_layer = transformer_encoder_clzz(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            batch_first=True,
            dropout=token_dropout,
            activation=activ_f,
        )
        self.t_encoder = nn.TransformerEncoder(t_encoder_layer, num_layers)
        if self.final_norm == "ln":
            self.final_norm_layer = nn.LayerNorm(self.token_dim)
        elif self.final_norm == "bn":
            raise ValueError("not implemented")
            self.final_norm_layer = nn.BatchNorm1d(self.no_patches)
        elif self.final_norm is False:
            self.final_norm_layer = nn.Identity()
        else:
            raise ValueError("not implemented")

    def forward(self, x, language_tokens, language_tokens_att_maks, vis_tokens_mask=None):
        bs, n, _ = x.shape

        x = self.pos_embedding_layer(x)
        x = x + self.image_kind_embedding
        x = nn.functional.dropout(x, self.patch_dropout, self.training)

        language_tokens = language_tokens + self.lang_kind_embedding
        if self.lang_pos_embedding:
            language_tokens = self.lang_pos_embedding(language_tokens)

        if language_tokens_att_maks is not None:
            padding_mask = self.padding_mask.repeat(bs, x.shape[1])
            padding_mask = torch.cat([padding_mask, language_tokens_att_maks], dim=1)
        else:
            padding_mask = None

        x = torch.cat([x, language_tokens], dim=1)
        if vis_tokens_mask is None:
            mask = None
        else:
            no_total_tokens = x.shape[1]
            no_lang_tokens = language_tokens.shape[1]
            no_vis_tokens = vis_tokens_mask.shape[0]
            vis_tokens_mask = torch.cat([vis_tokens_mask, torch.zeros(no_vis_tokens, no_lang_tokens)], dim=1)
            mask = torch.cat([vis_tokens_mask, torch.zeros((no_lang_tokens, no_total_tokens))])
            mask = mask.to(x.device).type(torch.bool)
        
        t_encoder_output = self.t_encoder(x, mask=mask, src_key_padding_mask=padding_mask)
        if isinstance(t_encoder_output, tuple):
            x, attentions = t_encoder_output
        else:
            x = t_encoder_output
            attentions = None

        vis_tokens = x[:, :n]

        # return x, class_token, attentions, None
        vis_tokens = self.final_norm_layer(vis_tokens)
        return vis_tokens, x[:, n:], attentions, None


class CrossTransformerTokenModule(CrossTransformerModuleBox):
    """Used for cross fusion using just the CLS token from the bert encoder"""

    def __init__(
        self,
        no_patches,
        patch_dropout,
        input_f_size,
        pos_embedding_layer,
        num_layers=2,
        num_heads=4,
        classif_token=False,
        fforward_multiplier=2,
        token_dropout=0.1,
        back_to_img_fn="regroup",
        activ_f="relu",
        patch_norm=False,
        final_ln=False,
    ):
        super().__init__(
            no_patches,
            patch_dropout,
            input_f_size,
            pos_embedding_layer,
            num_layers,
            num_heads,
            classif_token=False,
            fforward_multiplier=fforward_multiplier,
            token_dropout=token_dropout,
            back_to_img_fn=back_to_img_fn,
            activ_f=activ_f,
            patch_norm=patch_norm,
            final_ln=final_ln,
        )

    def forward(self, x, language_tokens, language_tokens_att_maks):
        bs, n, _ = x.shape
        x = self.pos_embedding_layer(x)
        x = x + self.image_kind_embedding

        x = nn.functional.dropout(x, self.patch_dropout, self.training)

        language_tokens = language_tokens + self.lang_kind_embedding
        x = torch.cat([x, language_tokens], dim=1)

        # No need for padding mask, we attend to everything
        t_encoder_output = self.t_encoder(x)
        if isinstance(t_encoder_output, tuple):
            x, _ = t_encoder_output
        else:
            x = t_encoder_output

        x = x[:, :n]

        x = self.final_norm_layer(x)
        return x, None, None


class SpaceTimeFusionLayer(nn.Module):
    def __init__(self, token_dim, num_heads, dim_feedforward, batch_first, token_dropout, activ_f):
        super().__init__()

        self.spatial_encoder = transformer_encoder_clzz(
            token_dim,
            num_heads,
            dim_feedforward,
            token_dropout,
            activ_f,
            batch_first=batch_first,
        )

        self.temporal_encoder = transformer_encoder_clzz(
            token_dim,
            num_heads,
            dim_feedforward,
            token_dropout,
            activ_f,
            batch_first=batch_first,
        )

    def forward(self, x, src_mask, src_key_padding_mask, get_attentions=False):
        b, n, s, d = x.shape
        x = torch.flatten(x, start_dim=0, end_dim=1)  # 1×nt·nh·nw·d --> nt×nh·nw·d

        outy = self.spatial_encoder(x)
        if isinstance(outy, tuple):
            res, _ = outy
        else:
            res = outy
        x = res + x

        x = x.reshape(b, n, s, d).transpose(1, 2)
        x = torch.flatten(x, start_dim=0, end_dim=1)  # nt×nh·nw·d --> nh·nw×nt·d

        outy = self.temporal_encoder(x)
        if isinstance(outy, tuple):
            res, _ = outy
        else:
            res = outy
        x = res + x
        x = x.reshape(
            b, n, s, d
        )  # reshaping because this block is used for several depths in ViViTEncoder class and Next layer will expect the x in proper shape
        # None for the attentions
        return x, None


class SpaceTimeFusionModule(CrossTransformerModuleBox):
    def __init__(
        self,
        no_patches,
        patch_dropout,
        input_f_size,
        pos_embedding_layer,
        num_layers=2,
        num_heads=4,
        classif_token=False,
        fforward_multiplier=2,
        token_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        pos_embedding="learned",
        patch_norm=False,
        final_norm=False,
    ):
        super().__init__(
            no_patches,
            patch_dropout,
            input_f_size,
            pos_embedding_layer,
            num_layers,
            num_heads,
            classif_token,
            fforward_multiplier,
            token_dropout,
            back_to_img_fn,
            activ_f,
            patch_norm,
            final_norm,
        )

        t_encoder_layer = SpaceTimeFusionLayer(
            token_dim=self.token_dim,
            num_heads=num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            batch_first=True,
            token_dropout=token_dropout,
            activ_f=activ_f,
        )
        self.t_encoder = nn.TransformerEncoder(t_encoder_layer, num_layers)

    def forward(self, x, flow_tokens_att_maks=None):
        out, atts = self.t_encoder(x)

        return out, atts, None
