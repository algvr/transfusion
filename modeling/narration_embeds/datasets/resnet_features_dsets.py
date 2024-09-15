import copy
import json
import numpy as np
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision.models import resnet50, resnet18, resnet34
import torchvision.transforms as transforms
from tqdm import tqdm

from data_preprocessing.datasets.readers import (Ego4dDataReader,
                                                 SFastFeaturesReader)      
from data_preprocessing.datasets.video_readers import get_clip_frame_idxs, get_clip_frame_idxs_block


def setup_idx_fn(visual_sampling_args):
    if visual_sampling_args["block"] == 1:
        return get_clip_frame_idxs
    elif visual_sampling_args["block"] > 1:
        return get_clip_frame_idxs_block
    else:
        raise ValueError("Cannot have flow block argument smaller than 1")


res50_data_dirs = {
    "ego4d": Path(os.path.expandvars("$DATA/Ego4d/data/lmdb")),
    "ego4djpg": Path(os.path.expandvars("$DATA/Ego4d/data/lmdb")),
    "ego4djpgv2": Path(os.path.expandvars("$DATA/Ego4d/data/lmdb"))
}


class PrevRes50Features(Dataset):
    def __init__(
        self,
        to_wrap_dset,
        visual_sampling_args,
        device="cpu",
    ) -> None:
        super().__init__()
        self.to_wrap_dset = to_wrap_dset
        self.visual_sampling_args = visual_sampling_args
        self.nao_annots = copy.deepcopy(to_wrap_dset.get_nao_annots())
      
        self.device = device
        self.readers = self.to_wrap_dset.readers
        self.get_vis_idxs = setup_idx_fn(self.visual_sampling_args)
        self.source= self.to_wrap_dset.source

        self.model = resnet50()
        self.model.eval()
        self.model.to(device)

        videos = []
        self.image_readers = {}
        self.data_dir = res50_data_dirs[self.source]

        self.cache_path = self.data_dir.parent

        if self.source in {"ego4d", "ego4djpg"}:
           videos = [video.name for video in self.data_dir.glob("*/**") if "flow" not in video.name and "lmdb" not in video.name]

        else:
            videos = self.to_wrap_dset.nao_annots["video_id"].unique()
        
        for v in videos:
            self.image_readers[v] = Ego4dDataReader(self.data_dir/v, v)

        self.transform =transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        self.precompute_res50_features()
        

    def precompute_res50_features(self):
        no_examples = range(len(self.to_wrap_dset))

        print("Precomputing Res50 features")
       
        no_exs = len(no_examples)
        batch_f = []
        batch_ids = []

        bs = 48

        for i, idx in enumerate(tqdm(no_examples)):
            annot = self.to_wrap_dset.get_annot_by_idx(idx)
            frame = annot.name
            idy = f"{annot['video_id']}_{str(frame)}"
            batch_ids.append(idy)

            if os.path.isfile(f"{self.cache_path/idy}.pt"):
                continue

            imgs_idxs = self.get_vis_idxs(annot.name - 1, 
                                    self.visual_sampling_args["stride"], 
                                    self.visual_sampling_args["num_frames"], 
                                    self.visual_sampling_args["block"])

            reader = self.image_readers[annot.video_id]
            try:
                f_frames = np.array(reader.get_clip(imgs_idxs)).transpose(0, 3,1,2)/255.0
                inp_features = torch.from_numpy(f_frames)
                inp_features = self.transform(inp_features)

            except ValueError:
                print(f"Failed at {annot.name}- {annot.video_id}, {imgs_idxs=}")
                imgs_idxs = self.get_vis_idxs(annot.name - 1, 
                                    2, 
                                    self.visual_sampling_args["num_frames"], 
                                    self.visual_sampling_args["block"])
                f_frames = np.array(reader.get_clip(imgs_idxs)).transpose(0, 3,1,2)/255.0
                inp_features = torch.from_numpy(f_frames)
                inp_features = self.transform(inp_features)

            inp_features = F.interpolate(inp_features, (480,640))

            batch_f.append(inp_features)

            if i % bs == bs-1 or i == no_exs-1:
                no_samples_in_b = len(batch_f)
                
                with torch.no_grad():
                        batch_f = torch.cat(batch_f, axis=0)
                        batch_f = self.model._forward_impl(batch_f.float().to(self.device))
                
                batch_f = torch.chunk(batch_f, no_samples_in_b)
                to_save = zip(batch_f, batch_ids)
                for save_entry in to_save:
                    torch.save(save_entry[0], f"{self.cache_path/save_entry[1]}.pt")

                batch_f = []
                batch_ids = []

    def get_features_for_idx(self, idx):
        annot = self.to_wrap_dset.get_annot_by_idx(idx)

        features = torch.load(f"{self.cache_path/annot['video_id']}_{str(annot.name)}.pt", map_location="cpu")[-self.visual_sampling_args["num_frames"]:]

        if features.shape[-1] != 2048:
            imgs_idxs = self.get_vis_idxs(annot.name - 1, 
                                    self.visual_sampling_args["stride"], 
                                    self.visual_sampling_args["num_frames"], 
                                    self.visual_sampling_args["block"])

            reader = self.image_readers[annot.video_id]
            f_frames = np.array(reader.get_clip(imgs_idxs)).transpose(0, 3,1,2)/255.0
            inp_features = torch.from_numpy(f_frames)
            inp_features = self.transform(inp_features)
            inp_features = F.interpolate(inp_features, (480,640))

            with torch.no_grad():
                features = self.model._forward_impl(inp_features.float().to(self.device)).cpu()

        if features.shape[0] != self.visual_sampling_args["num_frames"]:
            dif = self.visual_sampling_args["num_frames"]- features.shape[0] 
            features = torch.cat([torch.zeros((dif, 2048)), features])

        return features

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        features = self.get_features_for_idx(idx)

        return {**example, "visual_features": features.type(torch.float32)}

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
