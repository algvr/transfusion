from typing import Any, Optional

import torch
import torch.functional as F
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn, _get_clones

from modeling.cross_fusion.cross_f_layers import CrossTransformerModule
from modeling.cross_fusion.utils import patchify_image
from modeling.obj_detection.wrapper_utils import is_torch_18v
from modeling.cross_fusion.ego_fusion.torch18_adapters import MultiheadAttentionBFirst

if is_torch_18v(torch.__version__):
    mha_module_clzz = MultiheadAttentionBFirst
else:
    mha_module_clzz = nn.MultiheadAttention


class QKVEncoder(nn.Module):
    def __init__(
        self,
        vdim,
        qdim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
    ) -> None:
        super().__init__()
        if vdim == None:
            vdim = qdim
        self.self_attn = mha_module_clzz(qdim, nhead, dropout=dropout, kdim=qdim, vdim=vdim, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(vdim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, vdim)

        self.norm1 = nn.LayerNorm(vdim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(vdim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(QKVEncoder, self).__setstate__(state)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        get_attentions=False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        q2, attentions, vs = self.self_attn(
            q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=get_attentions
        )
        q = q + self.dropout1(q2)
        q = self.norm1(q)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(q))))
        q = q + self.dropout2(q2)
        q = self.norm2(q)

        # attention shape: [BS, N_Heads, Q_len, Seq_len]
        # values shape: torch.Size([BS, N_Heads, Seq_len, embed_dim])
        return q, attentions, vs


class AsymmetricCrossFTokenModule(CrossTransformerModule):
    """Used for cross fusion using just the CLS token from the BERT encoder"""

    def __init__(
        self,
        in_channels,
        feature_w,
        feature_h,
        patch_w,
        patch_h,
        patch_dropout,
        token_dim,
        vis_layers=3,
        lang_layers=2,
        num_heads=4,
        classif_token=False,
        back_to_img=True,
        fforward_multiplier=1,
        vis_dropout=0.1,
        lang_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        backproj_dropout=0.1,
        pos_embedding="learned",
        lang_to_hmap=False,
        patch_norm=False,
        final_ln=False,
    ):
        super().__init__(
            in_channels,
            feature_w,
            feature_h,
            patch_w,
            patch_h,
            patch_dropout,
            token_dim,
            vis_layers,
            num_heads,
            classif_token=False,
            back_to_img=back_to_img,
            fforward_multiplier=fforward_multiplier,
            token_dropout=vis_dropout,
            back_to_img_fn=back_to_img_fn,
            activ_f=activ_f,
            backproj_dropout=backproj_dropout,
            pos_embedding=pos_embedding,
            lang_to_hmap=lang_to_hmap,
            patch_norm=patch_norm,
            final_ln=final_ln,
        )
        del self.t_encoder
        self.no_vis_layers = vis_layers
        self.no_lang_layers = lang_layers
        self.get_attentions = False

        vis_c_encoder_layer = QKVEncoder(
            token_dim,
            token_dim,
            num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            dropout=vis_dropout,
            activation=activ_f,
        )
        lang_c_encoder_layer = QKVEncoder(
            token_dim,
            token_dim,
            num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            dropout=lang_dropout,
            activation=activ_f,
        )
        self.cross_vis_layers = _get_clones(vis_c_encoder_layer, vis_layers)
        self.cross_lang_layers = _get_clones(lang_c_encoder_layer, lang_layers)

    def forward(self, vis_features, language_f, language_f_att_maks):
        x = self.patches_to_token(patchify_image(vis_features, self.patch_w, self.patch_h))
        bs, n, _ = x.shape

        heatmap_token = torch.repeat_interleave(self.heatmap_token, repeats=bs, dim=0)
        x = torch.cat((heatmap_token, x), dim=1)
        x = x + self.pos_embedding
        x = x + self.image_kind_embedding
        x = self.patch_dropout(x)

        language_f = self.lang_f_input_ln(language_f)
        language_f = language_f + self.lang_kind_embedding

        attentions = [[], []]
        values = [[], []]

        # No need for padding mask, we attend to everything
        v_k = torch.cat((x, language_f), dim=1)
        if self.lang_to_hmap:
            language_f, atts, vs = self.cross_lang_layers[0](language_f, v_k, v_k, get_attentions=self.get_attentions)
        else:
            language_f, atts, vs = self.cross_lang_layers[0](
                language_f, v_k[:, 1:], v_k[:, 1:], get_attentions=self.get_attentions
            )
        values[0].append(vs)
        attentions[0].append(atts)
        x, atts, vs = self.cross_vis_layers[0](x, v_k, v_k, get_attentions=self.get_attentions)
        values[1].append(vs)
        attentions[1].append(atts)

        for i in range(1, self.no_lang_layers):
            v_k = torch.cat((x, language_f), dim=1)
            x, atts, vs = self.cross_vis_layers[i](x, v_k, v_k, get_attentions=self.get_attentions)
            attentions[1].append(atts)
            values[1].append(vs)

            if self.lang_to_hmap:
                language_f, atts, vs = self.cross_lang_layers[i](
                    language_f, v_k, v_k, get_attentions=self.get_attentions
                )
            else:
                language_f, atts, vs = self.cross_lang_layers[i](
                    language_f, v_k[:, 1:], v_k[:, 1:], get_attentions=self.get_attentions
                )
            attentions[0].append(atts)
            values[0].append(vs)

        for i in range(self.no_lang_layers, self.no_vis_layers):
            v_k = torch.cat((x, language_f), dim=1)
            x, atts, vs = self.cross_vis_layers[i](x, v_k, v_k, get_attentions=self.get_attentions)
            attentions[1].append(atts)
            values[1].append(vs)

        if self.back_to_img_fn == "token":
            hmap_token = x[:, 0]
        else:
            hmap_token = x[:, 1:]

        if self.final_ln:
            hmap_token = self.vis_ln(hmap_token)
            language_f = self.lang_ln(language_f)

        return hmap_token, language_f.squeeze(), attentions, values


class AsymmetricCrossFModule(AsymmetricCrossFTokenModule):
    """Used for cross fusion with all tokens from the BERT encoder"""

    def __init__(
        self,
        in_channels,
        feature_w,
        feature_h,
        patch_w,
        patch_h,
        patch_dropout,
        token_dim,
        vis_layers=3,
        lang_layers=2,
        num_heads=4,
        classif_token=False,
        back_to_img=True,
        fforward_multiplier=1,
        vis_dropout=0.1,
        lang_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        backproj_dropout=0.1,
        pos_embedding="learned",
        lang_to_hmap=False,
        patch_norm=False,
        final_ln=False,
    ):
        super().__init__(
            in_channels,
            feature_w,
            feature_h,
            patch_w,
            patch_h,
            patch_dropout,
            token_dim,
            vis_layers=vis_layers,
            lang_layers=lang_layers,
            num_heads=num_heads,
            classif_token=classif_token,
            back_to_img=back_to_img,
            fforward_multiplier=fforward_multiplier,
            vis_dropout=vis_dropout,
            lang_dropout=lang_dropout,
            back_to_img_fn=back_to_img_fn,
            activ_f=activ_f,
            backproj_dropout=backproj_dropout,
            pos_embedding=pos_embedding,
            lang_to_hmap=lang_to_hmap,
            patch_norm=patch_norm,
            final_ln=final_ln,
        )

    def forward(self, vis_features, language_f, language_f_att_maks):
        x = self.patches_to_token(patchify_image(vis_features, self.patch_w, self.patch_h))
        bs, n, _ = x.shape

        heatmap_token = torch.repeat_interleave(self.heatmap_token, repeats=bs, dim=0)
        x = torch.cat((heatmap_token, x), dim=1)
        x = x + self.pos_embedding
        x = x + self.image_kind_embedding
        x = self.patch_dropout(x)

        padding_mask = torch.repeat_interleave(self.padding_mask, repeats=bs, dim=0)
        padding_mask = torch.cat([padding_mask, language_f_att_maks], dim=1)

        language_f = self.lang_f_input_ln(language_f)
        language_f = language_f + self.lang_kind_embedding

        attentions = [[], []]
        values = [[], []]

        v_k = torch.cat((x, language_f), dim=1)
        if self.lang_to_hmap:
            language_f, atts, vs = self.cross_lang_layers[0](
                language_f, v_k, v_k, src_key_padding_mask=padding_mask, get_attentions=self.get_attentions
            )
        else:
            language_f, atts, vs = self.cross_lang_layers[0](
                language_f,
                v_k[:, 1:],
                v_k[:, 1:],
                src_key_padding_mask=padding_mask[:, 1:],
                get_attentions=self.get_attentions,
            )
            attentions[0].append(atts)
            values[0].append(vs)

        x, atts, vs = self.cross_vis_layers[0](
            x, v_k, v_k, src_key_padding_mask=padding_mask, get_attentions=self.get_attentions
        )
        attentions[1].append(atts)
        values[1].append(vs)

        for i in range(1, self.no_lang_layers):
            v_k = torch.cat((x, language_f), dim=1)
            x, atts, vs = self.cross_vis_layers[i](x, v_k, v_k, get_attentions=self.get_attentions)
            attentions[1].append(atts)
            values[1].append(vs)
            if self.lang_to_hmap:
                language_f, atts, vs = self.cross_lang_layers[i](
                    language_f, v_k, v_k, src_key_padding_mask=padding_mask, get_attentions=self.get_attentions
                )
            else:
                language_f, atts, vs = self.cross_lang_layers[i](
                    language_f,
                    v_k[:, 1:],
                    v_k[:, 1:],
                    src_key_padding_mask=padding_mask[:, 1:],
                    get_attentions=self.get_attentions,
                )
            attentions[0].append(atts)
            values[0].append(vs)

        for i in range(self.no_lang_layers, self.no_vis_layers):
            v_k = torch.cat((x, language_f), dim=1)
            x, atts, vs = self.cross_vis_layers[i](
                x, v_k, v_k, src_key_padding_mask=padding_mask, get_attentions=self.get_attentions
            )
            attentions[1].append(atts)
            values[1].append(vs)

        if self.back_to_img_fn == "token":
            hmap_token = x[:, 0]
        else:
            hmap_token = x[:, 1:]

        language_f = language_f[:, 0]

        if self.final_ln:
            hmap_token = self.vis_ln(hmap_token)
            language_f = self.lang_ln(language_f)

        return (hmap_token, language_f, attentions, values)
