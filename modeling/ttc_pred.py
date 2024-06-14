import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class BoxEmbedder(nn.Module):
    def __init__(self, feat_dim, num_steps, pos_encoder):
        super().__init__()
        self.feat_dim = feat_dim
        self.pos_encoder = pos_encoder
        self.num_steps = num_steps

        self.coord_embed = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.SiLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, boxes):
        # boxes.shape: bs, num_boxes, 2 * num_steps, 4
        idxs = torch.floor(boxes.flatten() * self.num_steps).long()  # bs * num_boxes * 2(hands) * num_steps * 4(coords)
        pes = self.pos_encoder.pe[idxs]
        # every dimension other than the batch and feature dimension collapses into the token dimension
        return self.coord_embed(pes).reshape(boxes.shape[0], -1, self.feat_dim)


class HandPoseEmbedder(nn.Module):
    def __init__(self, feat_dim, hand_feat_dim, pos_encoder):
        super().__init__()
        self.feat_dim = feat_dim
        self.pos_encoder = pos_encoder
        self.hand_feat_dim = hand_feat_dim

        self.hand_embed = nn.Sequential(
            nn.Linear(self.hand_feat_dim, self.feat_dim),
            nn.SiLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    def forward(self, poses):
        # poses.shape: bs, num_boxes, 2 * num_steps, 63
        # every dimension other than the batch and feature dimension collapses into the token dimension
        return self.hand_embed(poses).reshape(poses.shape[0], -1, self.feat_dim)


class TTCPredictionHead(nn.Module):
    def __init__(self, num_steps, feat_dim, num_heads, num_layers, ff_dim, dropout, emb_steps_hand, emb_steps_object, hand_feat_dim, object_feat_dim):
        super().__init__()

        self.num_steps = num_steps
        self.feat_dim = feat_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb_steps_hand = emb_steps_hand
        self.emb_steps_object = emb_steps_object
        self.hand_feat_dim = hand_feat_dim
        self.object_feat_dim = object_feat_dim

        self.cls_token = nn.Parameter(torch.randn(1, self.feat_dim))
        # self.missing_enc = nn.Parameter(torch.randn(1, self.feat_dim))
        self.hand_side_enc = nn.Parameter(torch.randn(2, self.feat_dim))
        self.x0_type_enc = nn.Parameter(torch.randn(1, self.feat_dim))
        self.y0_type_enc = nn.Parameter(torch.randn(1, self.feat_dim))
        self.x1_type_enc = nn.Parameter(torch.randn(1, self.feat_dim))
        self.y1_type_enc = nn.Parameter(torch.randn(1, self.feat_dim))

        self.pos_encoder = PositionalEncoding(self.feat_dim, self.dropout)

        if self.num_layers > 0:
            self.enc_layer = nn.TransformerEncoderLayer(self.feat_dim, self.num_heads, self.ff_dim, self.dropout,
                                                        batch_first=True)
            self.enc = nn.TransformerEncoder(self.enc_layer, num_layers=self.num_layers)

        # box encoders
        self.hand_box_embedder = BoxEmbedder(self.feat_dim, self.emb_steps_hand, self.pos_encoder)
        if self.emb_steps_object > 0:
            self.object_box_embedder = BoxEmbedder(self.feat_dim, self.emb_steps_object, self.pos_encoder)

        if self.object_feat_dim > 0:
            self.object_feat_embedder = nn.Linear(self.object_feat_dim, self.feat_dim)
        if self.hand_feat_dim > 0:
            self.hand_pose_embedder = HandPoseEmbedder(self.feat_dim, self.hand_feat_dim, self.pos_encoder)

        self.ttc_out = nn.Linear(self.feat_dim, 1)

    def forward(self, x):
        # box_features, target_ttcs, frame_ids, object_boxes, hand_poses, hand_boxes, orig_pred_boxes, orig_pred_nounsm orig_pred_verbs
        # embedded box features, 4 for box x0,x1,y0,y1; left then right hand boxes (4 per step); left then right embedded hand poses
        if self.object_feat_dim > 0:
            object_feat = self.object_feat_embedder(x["box_features"].float())[:, None, :]
        else:
            object_feat = None
        
        B = x["box_features"].shape[0]

        if self.emb_steps_object > 0:
            object_box_feat = self.object_box_embedder(x["object_boxes"])
            object_box_feat += torch.cat((self.x0_type_enc, self.y0_type_enc, self.x1_type_enc, self.y1_type_enc))[None].repeat((B, object_box_feat.shape[1] // 4, *([1] * (len(object_box_feat.shape)-2))))
        else:
            object_box_feat = None

        if self.emb_steps_hand > 0:
            hand_box_feat = self.hand_box_embedder(x["hand_boxes"])
            mid = hand_box_feat.shape[-2] // 2
            hand_box_feat[:mid] += self.hand_side_enc[0:1][None]
            hand_box_feat[mid:] += self.hand_side_enc[1:2][None]
            hand_box_feat += torch.cat((self.x0_type_enc, self.y0_type_enc, self.x1_type_enc, self.y1_type_enc))[None].repeat((B, hand_box_feat.shape[1] // 4, *([1] * (len(hand_box_feat.shape)-2))))
        else:
            hand_box_feat = None

        if self.hand_feat_dim > 0:
            hand_pose_feat = self.hand_pose_embedder(x["hand_poses"])
            mid = hand_pose_feat.shape[-2] // 2
            hand_pose_feat[:mid] += self.hand_side_enc[0:1][None]
            hand_pose_feat[mid:] += self.hand_side_enc[1:2][None]
        else:
            hand_pose_feat = None

        if self.emb_steps_hand > 0:
            pes_hand = self.pos_encoder.pe[range(self.emb_steps_hand, self.emb_steps_hand + self.num_steps)].squeeze(1)[None].repeat((B, 1, 1)).repeat((1, 2, 1))  # last repeat: hand side
            hand_box_feat += pes_hand.repeat_interleave(4, dim=1)  # x0, y0, x1, y1
            if self.hand_feat_dim > 0:
                hand_pose_feat += pes_hand

        enc_features = torch.cat((*([object_feat] if object_feat is not None else []),
                                  *([object_box_feat] if object_box_feat is not None else []),
                                  *([hand_box_feat] if hand_box_feat is not None else []),
                                  *([hand_pose_feat] if hand_pose_feat is not None else [])), dim=1)
        if self.num_layers > 0:
            enc_in = torch.cat((self.cls_token[None].repeat(enc_features.shape[0], 1, 1), enc_features), dim=1)
            # enc_in += self.pos_encoder.pe[range(enc_in.shape[1])].squeeze(1)[None].repeat((B, 1, 1))
            enc_out = self.enc(enc_in)
            ttc_pre_act = self.ttc_out(enc_out[:, 0, :])[:, 0]
        else:
            ttc_pre_act = self.ttc_out(F.gelu(enc_features.view(enc_features.shape[0], -1)))[:, 0]
        final_out = F.softplus(ttc_pre_act)
        return final_out

