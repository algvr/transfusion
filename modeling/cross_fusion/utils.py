import math
from torch import nn
import torch
import torch.nn.functional as F

cache_masks = {}


def get_visual_token_mask(img_shape, mask_type):
    if mask_type == "global":
        return None
    
    # 0 is attended, 1 is blocked
    elif "local" in mask_type:
        if not str(img_shape) in cache_masks:
            no_local_tokens = int(mask_type.split("_")[-1])
            mask  = torch.ones((img_shape[0]*img_shape[1], img_shape[0], img_shape[1]))
            
            for i, sub_mask in enumerate(mask):
                true_c = i % img_shape[1]
                true_r = i // img_shape[1]
                for j1 in range(-no_local_tokens, no_local_tokens+1):
                    for j2 in range(-no_local_tokens, no_local_tokens+1):
                        c = max(0, min(true_c+j1, img_shape[1]-1))
                        r = max(0, min(true_r+j2, img_shape[0]-1))
                        sub_mask[r, c] = 0

            cache_masks[str(img_shape)] = mask.flatten(1)

        return cache_masks[str(img_shape)]
    else:
        raise NotImplementedError()
    

def patchify_image(image, patch_w, patch_h):
    B, C, H, W = image.shape
    patches = image.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * patch_h * patch_w)
    return patches


def regroup_patches(patches, init_h, init_w, patch_h, patch_w):
    """patches should be of shape B x F*num_p_h*num_p_w x Num_patches"""
    res = patches.transpose(-2, -1)
    res = torch.nn.functional.fold(res, (init_h, init_w), (patch_h, patch_w), stride=(patch_h, patch_w))
    return res


def get_regroup_acti_f(activ_f):
    if activ_f == "elu":
        activ_f = lambda x: 1 + F.elu(x)
    elif activ_f == "tanh":
        activ_f = torch.tanh
    elif activ_f == "h_swish":
        activ_f = torch.nn.Hardswish()
    elif activ_f is None:
        activ_f = nn.Identity()
    elif activ_f == "relu":
        activ_f = torch.nn.ReLU()
    else:
        raise NotImplementedError(f"{activ_f} not recognized")
    return activ_f


class RegroupPatchesLayer(nn.Module):
    def __init__(self, token_dim, init_h, init_w, patch_h, patch_w, backproj_dropout=0.1, activ_f="elu"):
        super().__init__()
        self.init_h = init_h
        self.init_w = init_w
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.back_dropout = nn.Dropout(backproj_dropout)
        self.linear = nn.Linear(token_dim, patch_h * patch_w)

        self.activ_f = get_regroup_acti_f(activ_f)

    def forward(self, x, cls_f):
        x = self.back_dropout(x)
        x = self.linear(x)
        x = self.activ_f(x)
        return regroup_patches(x, self.init_h, self.init_w, self.patch_h, self.patch_w)


class RegroupPatchesLayerBox(nn.Module):
    def __init__(
        self,
        token_dim,
        init_h,
        init_w,
        patch_h,
        patch_w,
        out_channels,
        backproj_dropout=0.1,
        activ_f=None,
        final_norm=False,
    ):
        super().__init__()
        self.init_h = init_h
        self.init_w = init_w
        self.patch_h = patch_h
        self.patch_w = patch_w

        self.back_dropout = nn.Dropout(backproj_dropout)
        self.linear = nn.Linear(token_dim, patch_h * patch_w * out_channels)
        self.activ_f = get_regroup_acti_f(activ_f)
        self.final_norm = final_norm
        if self.final_norm == "ln":
            self.final_norm_layer = nn.LayerNorm(patch_h * patch_w * out_channels)
        elif self.final_norm == "bn":
            self.final_norm_layer = nn.BatchNorm1d(patch_h * patch_w * out_channels)
        else:
            self.final_norm_layer = nn.Identity()

    def forward(self, x, cls_f=None):
        x = self.back_dropout(x)
        x = self.linear(x)
        x = self.activ_f(x)
        x = self.final_norm_layer(x)
        return regroup_patches(x, self.init_h, self.init_w, self.patch_h, self.patch_w)


class RegroupPatchesLayerExtra(nn.Module):
    def __init__(self, token_dim, fat_dim, init_h, init_w, patch_h, patch_w, backproj_dropout=0.1, elu_last=False):
        super().__init__()
        self.init_h = init_h
        self.init_w = init_w
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.linear_1 = nn.Linear(token_dim, fat_dim)
        self.back_dropout = nn.Dropout(backproj_dropout)
        self.linear_2 = nn.Linear(fat_dim, patch_h * patch_w)
        self.elu_last = elu_last

    def forward(self, x, cls_f):
        x = self.linear_1(x)
        if not self.elu_last:
            x = 1 + F.elu(x)

        x = self.back_dropout(x)
        x = self.linear_2(x)

        if self.elu_last:
            x = 1 + F.elu(x)
        return regroup_patches(x, self.init_h, self.init_w, self.patch_h, self.patch_w)


class RegroupPatchesGatedLayerExtra(nn.Module):
    def __init__(self, token_dim, upscale_dim, init_h, init_w, patch_h, patch_w, backproj_dropout=0.1):
        super().__init__()
        self.init_h = init_h
        self.init_w = init_w
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.linear_1 = nn.Linear(token_dim, upscale_dim)
        self.back_dropout = nn.Dropout(backproj_dropout)
        self.linear_2 = nn.Linear(upscale_dim, patch_h * patch_w)
        self.cls_f_linear = nn.Linear(token_dim, upscale_dim)

    def forward(self, x, cls_f):
        cls_f = self.cls_f_linear(cls_f)
        cls_f = F.sigmoid(cls_f)

        x = self.linear_1(x)
        x = torch.mul(x, cls_f.unsqueeze(1))
        x = self.back_dropout(x)
        x = self.linear_2(x)
        x = 1 + F.elu(x)

        return regroup_patches(x, self.init_h, self.init_w, self.patch_h, self.patch_w)


class PositionalEmbeddingLayer(nn.Module):
    def __init__(self, embedding_type, num_patches, token_dim, temporal_dim=0):
        super().__init__()
        self.embedding_type = embedding_type
        self.num_patches = num_patches
        self.token_dim = token_dim
        self.temporal_dim = temporal_dim

        if not temporal_dim:
            if self.embedding_type == "learned":
                self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.token_dim))
            elif self.embedding_type == "zero":
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.token_dim))
            elif self.embedding_type == "sin1d":
                pos_embedding = get_sin1d_embed(self.num_patches, self.token_dim)
                self.register_buffer("pos_embedding", pos_embedding)
            elif self.embedding_type == "sin2d":
                pos_embedding = get_sin2d_embed(
                    self.feature_h // self.patch_h, self.feature_w // self.patch_w, self.token_dim
                )
                self.register_buffer("pos_embedding", pos_embedding)
            else:
                raise ValueError(f"{self.embedding_type=} is not recognized for {temporal_dim}")
        else:
            if self.embedding_type == "learned":
                self.temporal_embedding = nn.Parameter(torch.randn(1, temporal_dim, 1, self.token_dim))
                self.pos_embedding = nn.Parameter(torch.randn(1, 1, self.num_patches, self.token_dim))
            elif self.embedding_type == "zero":
                self.temporal_embedding = nn.Parameter(torch.zeros(1, temporal_dim, 1, self.token_dim))
                self.pos_embedding = nn.Parameter(torch.zeros(1, 1, self.num_patches, self.token_dim))
            elif self.embedding_type == "sin1d":
                self.temporal_embedding = nn.Parameter(torch.zeros(1, temporal_dim, 1, self.token_dim))
                pos_embedding = get_sin1d_embed(self.num_patches, self.token_dim).unsqueeze(0)
                self.register_buffer("pos_embedding", pos_embedding)
            else:
                raise ValueError(f"{self.embedding_type=} is not recognized for {temporal_dim}")

    def forward(self, x):
        _, np, _ = x.shape
        if np == self.num_patches:
            x = x + self.pos_embedding
        else:
            x = x + self.pos_embedding[:, :np, :]

        if self.temporal_dim:
            x = x + self.temporal_embedding
        return x


# taken from https://github.com/SforAiDl/vformer/blob/main/vformer/encoder/embedding/pos_embedding.py
class PosEmbedding(nn.Module):
    """
    Generalised Positional Embedding class
    """

    def __init__(self, shape, dim, drop=None, sinusoidal=False, std=0.02):
        super(PosEmbedding, self).__init__()

        if not sinusoidal:
            if isinstance(shape, int):
                shape = [1, shape, dim]
            else:
                shape = [1] + list(shape) + [dim]
            self.pos_embed = nn.Parameter(torch.zeros(shape))

        else:
            pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(shape)])
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            self.pos_embed = pe
            self.pos_embed.requires_grad = False
        nn.init.trunc_normal_(self.pos_embed, std=std)
        self.pos_drop = nn.Dropout(drop) if drop is not None else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed
        return self.pos_drop(x)


class BackProjectLayer(nn.Module):
    def __init__(self, token_dim, init_h, init_w, backproj_dropout=0.1):
        super().__init__()
        self.init_h = init_h
        self.init_w = init_w
        self.back_dropout = nn.Dropout(backproj_dropout)
        self.linear = nn.Linear(token_dim, init_h * init_w)

    def forward(self, x, cls_f):
        bs = x.size(0)
        x = self.back_dropout(x)
        x = self.linear(x)
        x = 1 + F.elu(x)
        return x.reshape((bs, 1, self.init_h, self.init_w))


def get_sin1d_embed(no_embeds, dim):
    position = torch.arange(no_embeds).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    pe = torch.zeros(no_embeds, 1, dim)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe.permute((1, 0, 2))


def get_sin2d_embed(height, width, m_dim, w_hmap_emb=True):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """

    if m_dim % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(dim))
    pe = torch.zeros(m_dim, height, width)
    # Each dimension use half of dim
    dim = int(m_dim / 2)
    div_term = torch.exp(torch.arange(0.0, dim, 2) * -(math.log(10000.0) / dim))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[dim + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.reshape((m_dim, -1)).transpose(0, 1)

    if w_hmap_emb:
        hmap_emb = get_sin1d_embed(height * width + 1, m_dim)[:, -1]
        pe = torch.cat((hmap_emb, pe), dim=0)

    return pe.unsqueeze(0)


if __name__ == "__main__":
    sin2d = get_sin2d_embed(3, 6, 768)
