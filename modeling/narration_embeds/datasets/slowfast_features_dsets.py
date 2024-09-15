import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from data_preprocessing.datasets.readers import SFastFeaturesReader


slowfast_data_dirs = {
    "epic": Path(os.path.expandvars("$DATA/EK/data/slowfast_features")),
    "epicv": Path(os.path.expandvars("$DATA/EK/data/slowfast_features")),
    "egtea": Path(os.path.expandvars("$DATA/EGTEAp/slowfast_features")),
    "epicv": Path(os.path.expandvars("$DATA/EK/data/slowfast_features")),
}


class PrevSlowfastFeatures(Dataset):
    def __init__(
        self,
        to_wrap_dset,
        embed_args,
        all_annots,
        w_leak=False,
        sort_by="video_id",
        stop_by="video_id",
        device="cpu",
        feature_key ="language_f"
    ) -> None:
        super().__init__()
        self.to_wrap_dset = to_wrap_dset
        self.embed_args = embed_args
        self.nao_annots = copy.deepcopy(to_wrap_dset.get_nao_annots())
        self.all_annots = (
            copy.deepcopy(all_annots).reset_index().set_index("episode_id").sort_values([sort_by, "start_frame"])
        )
        self.readers = self.to_wrap_dset.readers
        self.no_prev = int(embed_args["strategy"].split("_")[-1])
        self.w_leak = w_leak
        self.feature_key = feature_key
        self.source = self.to_wrap_dset.source

        if self.source in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            self.data_dir = Path(os.path.expandvars("$DATA/Ego4d/data/slowfast_features/"))
            if not self.data_dir.exists():
                self.data_dir = Path("/data/rpasca/shared/slowfast_features/")
            with open(self.data_dir / "ego4d.json", "r") as fp:
                self.ego4d_metadata = json.loads(fp.read())

            self.video_id_to_num_frames = {}
            for entry in self.ego4d_metadata["videos"]:
                self.video_id_to_num_frames[entry["video_uid"]] = entry["video_metadata"]["num_frames"]

            self.frame_to_f_idxs = self.setup_slowfast_cache()

        else:
            self.data_dir = slowfast_data_dirs[self.source]
            videos = self.to_wrap_dset.nao_annots["video_id"].unique()
            self.sfast_readers = {}
            for video in videos:
                self.sfast_readers[video] = SFastFeaturesReader(self.data_dir, video, no_segs_to_read=self.no_prev)


    def setup_slowfast_cache(
        self,
    ):
        print("Setup SlowFast features, might take a while")
        frame_to_f_idxs = {}
        # features are in windows of [0,31], [16, 47], [32,63], [38,70] say for a video of length 70
        # -> uses backpading
        for idx, nao_annot in tqdm(self.nao_annots.iterrows()):
            video_id = nao_annot["video_id"]
            if video_id not in frame_to_f_idxs:
                frame_to_f_idxs[video_id] = {}

            movie_len = self.video_id_to_num_frames[video_id]
            # idx = 16 -> win 0
            # idx = 37 -> win 1
            # idx = 47 -> win 1
            # idx = 48 -> win 2
            window = (idx - 16) / 16
            frac_p, int_p = math.modf(window)
            # # also subtract 1 because we index by 0
            # int_p = max(int_p-1, 0)

            # if we need frame 40, 40 /16 is 2.5 , if we read window 2 above we will have 7 frames in the future
            # if we need frame 35, 35/16 is 2.18 -> will have 12 frames in the future, about 0.4s, can extend over NAO
            w_end = min(math.ceil(window) * 16 + 31, movie_len)
            if not self.w_leak:
                # let's use the ttc entry to choose the window, if the window contains the contact frame, not good
                contact_frame = idx + nao_annot["det_diff"] * 30
                if contact_frame - 5 < w_end:
                    int_p -= 1


            if int_p < 0:
                print("rau taking with leak")
                int_p += 1
                

            frame_to_f_idxs[video_id][idx] = [int(int_p - i) for i in range(self.no_prev)][::-1]

        return frame_to_f_idxs

    def get_features_for_idx(self, idx):
        annot = self.to_wrap_dset.get_annot_by_idx(idx)
        if self.source in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            slowfast_features = torch.load(self.data_dir / "v1" / "slowfast8x8_r101_k400" / f"{annot['video_id']}.pt")

            # the idxs we want to grab from the extracted features, can be -1, 0, 1 for the 1st clip with prev 3
            # -> we will return only the 0, and 1
            # and attention mask accordingly
            idxs = self.frame_to_f_idxs[annot["video_id"]][annot.name]

            good_f = [slowfast_features[idx] for idx in idxs if idx >= 0]
            if len(good_f) != self.no_prev:
                return None
            features = torch.stack(good_f, dim=0)

        else:
            frame = annot.name
            if self.source == "epicv":
                frame = annot["orig_frame"]
            features = torch.from_numpy(self.sfast_readers[annot["video_id"]].get_frame(frame))

        return features

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        features = self.get_features_for_idx(idx)

        if features is None:
            return self.__getitem__(idx + 1)

        return {**example, self.feature_key: features.type(torch.float32)}

    def __len__(self):
        return len(self.to_wrap_dset)

    def get_nao_annots(self):
        return self.nao_annots

    def get_raw_item(self, idx):
        return self.to_wrap_dset.get_raw_item(idx)

    def get_ignore_labels(self, col):
        return self.to_wrap_dset.get_ignore_labels(col)

    def get_label_cutoff(self):
        return self.to_wrap_dset.get_label_cutoff()

    def drop_cutoff_classes(self):
        return self.to_wrap_dset.drop_cutoff_classes()

    def get_verb_mapping(self):
        return self.to_wrap_dset.get_verb_mapping()

    def get_noun_mapping(self):
        return self.to_wrap_dset.get_noun_mapping()

    def set_input_transforms(self, transforms):
        self.to_wrap_dset.set_input_transforms(transforms)

    def set_hmap_transforms(self, hmap_transforms):
        self.to_wrap_dset.set_hmap_transforms(hmap_transforms)

    def set_resize_fn(self, resize_fn):
        self.to_wrap_dset.set_resize_fn(resize_fn)

    def get_narration_embeds(self):
        nar_embeds = np.array([self.narration_embeds[k] for k in sorted(self.narration_embeds.keys())])
        nar_embeds = np.append(nar_embeds, np.zeros((1, nar_embeds.shape[-1])), axis=0)
        return nar_embeds

    def get_no_nouns(self):
        return self.to_wrap_dset.get_no_nouns()

    def get_no_verbs(self):
        return self.to_wrap_dset.get_no_verbs()

    def get_classes(self, clzz):
        return self.to_wrap_dset.get_classes(clzz)

    def set_resize_fn(self, resize_fn):
        self.to_wrap_dset.set_resize_fn(resize_fn)

    def set_input_transforms(self, transforms):
        self.to_wrap_dset.set_input_transforms(transforms)

    def set_localiz_transforms(self, transforms):
        self.to_wrap_dset.set_localiz_transforms(transforms)

    def set_flow_transform(self, transform):
        self.to_wrap_dset.set_flow_transform(transform)

    def get_annot_by_idx(self, idx):
        return self.to_wrap_dset.get_annot_by_idx(idx)


class SlowFastPooling(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.no_prev = args[0]["strategy"]
        out_mlp = args[0]["out_mlp"]
        self.size = args[0]["size"]
        self.out_dropout = nn.Dropout(args[0]["out_dropout"])
        self.use_out_tanh = args[0]["out_tanh"]
        if out_mlp:
            self.out_mlp = nn.Linear(self.size, out_mlp)
        else:
            self.out_mlp = None

    def unfreeze_embeddings(self):
        pass

    def forward(self, tensor, *args, **kwargs):
        tensor = torch.stack(tensor, dim=0)

        if self.out_mlp:
            tensor = self.out_mlp(tensor)

        if self.use_out_tanh:
            tensor = torch.tanh(tensor)

        if tensor.shape[1] > 1:
            tensor = F.normalize(tensor, p=2, dim=1)

        tensor = self.out_dropout(tensor)

        # must be HFace style, i.e. 0 is canceled and 1 is ok
        att_mask = torch.ones(tensor.shape[:2], device=tensor.device)

        return tensor, None, att_mask
