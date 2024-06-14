import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from modeling.cross_fusion.cross_qkv_layers import QKVEncoder
from modeling.cross_fusion.ego_fusion.cross_f_box_layers import CrossTransformerModuleBox



class AsymmetricCrossFModuleBox(CrossTransformerModuleBox):
    """Used for cross fusion using all tokens from the LM encoder"""

    def __init__(
        self,
        no_patches,
        patch_dropout,
        input_f_size,
        pos_embedding_layer,
        vis_layers=3,
        lang_layers=2,
        num_heads=4,
        fforward_multiplier=1,
        vis_dropout=0.1,
        lang_dropout=0.1,
        back_to_img_fn="token",
        activ_f="relu",
        pos_embedding="learned",
        patch_norm=False,
        final_ln=False,
    ):
        super().__init__(
            no_patches,
            patch_dropout,
            input_f_size,
            pos_embedding_layer,
            vis_layers,
            num_heads,
            classif_token=False,
            fforward_multiplier=fforward_multiplier,
            token_dropout=vis_dropout,
            back_to_img_fn=back_to_img_fn,
            activ_f=activ_f,
            pos_embedding=pos_embedding,
            patch_norm=patch_norm,
            final_ln=final_ln,
        )
        del self.t_encoder
        self.no_vis_layers = vis_layers
        self.no_lang_layers = lang_layers
        self.get_attentions = False

        vis_c_encoder_layer = QKVEncoder(
            self.token_dim,
            self.token_dim,
            num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            dropout=vis_dropout,
            activation=activ_f,
        )
        lang_c_encoder_layer = QKVEncoder(
            self.token_dim,
            self.token_dim,
            num_heads,
            dim_feedforward=int(self.token_dim * fforward_multiplier),
            dropout=lang_dropout,
            activation=activ_f,
        )

        self.cross_vis_layers = _get_clones(vis_c_encoder_layer, vis_layers)
        self.cross_lang_layers = _get_clones(lang_c_encoder_layer, lang_layers)

    def forward(self, x, language_f, language_tokens_att_maks):
        bs, n, _ = x.shape

        x = self.pos_embedding_layer(x)
        x = x + self.image_kind_embedding
        x = nn.functional.dropout(x, self.patch_dropout, self.training)

        # language_f = self.lang_f_input_ln(language_f)
        language_f = language_f + self.lang_kind_embedding

        attentions = [[], []]
        values = [[], []]

        padding_mask = self.padding_mask.repeat(bs, x.shape[1])
        padding_mask = torch.cat([padding_mask, language_tokens_att_maks], dim=1)
        v_k = torch.cat((x, language_f), dim=1)
        language_f, atts, vs = self.cross_lang_layers[0](language_f, v_k, v_k, get_attentions=self.get_attentions)
        values[0].append(vs)
        attentions[0].append(atts)

        # attend from visual to all features
        x, atts, vs = self.cross_vis_layers[0](x, v_k, v_k, get_attentions=self.get_attentions)
        values[1].append(vs)
        attentions[1].append(atts)

        for i in range(1, self.no_lang_layers):
            v_k = torch.cat((x, language_f), dim=1)
            x, atts, vs = self.cross_vis_layers[i](x, v_k, v_k, get_attentions=self.get_attentions)
            attentions[1].append(atts)
            values[1].append(vs)

            language_f, atts, vs = self.cross_lang_layers[i](language_f, v_k, v_k, get_attentions=self.get_attentions)

            attentions[0].append(atts)
            values[0].append(vs)

        # assume number of visual layers is larger than number of language layers
        for i in range(self.no_lang_layers, self.no_vis_layers):
            v_k = torch.cat((x, language_f), dim=1)
            x, atts, vs = self.cross_vis_layers[i](x, v_k, v_k, get_attentions=self.get_attentions)
            attentions[1].append(atts)
            values[1].append(vs)

        if self.back_to_img_fn == "token":
            hmap_token = x[:, 0]
        else:
            hmap_token = x[:, :n]

        return hmap_token, attentions, values


class AsymmetricCrossFTokenModuleBox:
    pass
