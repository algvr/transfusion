import copy
import json
import logging
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data.dataset import Dataset
from typing import Iterable

from data_preprocessing.utils.dataset_utils import get_label_mapping


compare_cols = [
    "Bboxes",
    "nao_clip_id",
    "nao_narration",
]


class SnaoIdSlicer(Dataset):
    def __init__(self, slice_ids, to_slice_dset):
        super().__init__()
        self.slice_ids = slice_ids

        self.to_slice_dset = to_slice_dset
        if slice_ids == None:
            slice_ids = to_slice_dset.nao_annots["nao_clip_id"].unique()

        self.idx_mapping = np.flatnonzero(to_slice_dset.get_nao_annots()["nao_clip_id"].isin(slice_ids))

        self.nao_annots = copy.deepcopy(to_slice_dset.get_nao_annots())[
            to_slice_dset.get_nao_annots()["nao_clip_id"].isin(slice_ids)
        ]

        self.verb_mapping = self.to_slice_dset.get_verb_mapping()
        self.noun_mapping = self.to_slice_dset.get_noun_mapping()

        self.noun_verb_frequencies = None

    def get_nao_annots(self):
        return self.nao_annots

    def set_resize_fn(self, resize_fn):
        self.to_slice_dset.set_resize_fn(resize_fn)

    def set_input_transforms(self, transforms):
        self.to_slice_dset.set_input_transforms(transforms)

    def set_localiz_transforms(self, transforms):
        self.to_slice_dset.set_localiz_transforms(transforms)

    def get_annot_by_idx(self, idx):
        annot = self.to_slice_dset.get_annot_by_idx(self.idx_mapping[idx])
        return annot

    def __len__(self):
        return len(self.nao_annots)

    def __getitem__(self, idx):
        orig_idx = self.idx_mapping[idx]
        to_return = self.to_slice_dset[orig_idx]

        return to_return

    def get_raw_item(self, idx):
        return self.to_slice_dset.get_raw_item(self.idx_mapping[idx])

    def get_b_class_weights(self, clazz):
        y = self.to_slice_dset.get_classes(clazz)
        target_ds = self.to_slice_dset.to_wrap_dset
        while not hasattr(target_ds, "noun_mapping"):
            target_ds = target_ds.to_wrap_dset
        full_mapping = {
            "noun": target_ds.noun_mapping.copy(),
            "verb": target_ds.verb_mapping.copy(),
            "action_class": getattr(target_ds, "action_mapping", target_ds.verb_mapping).copy()
        }[clazz]
        max_val = max(full_mapping.values())
        final_weights = np.ones(max_val + 1)


        # for verb classes I am missing some entries in train/val i.e. id 61:swing and 28:lock
        not_in = [idx for k, idx in full_mapping.items() if k not in y.unique()]
        mapping = sorted(list(set(full_mapping.keys()).intersection(y.unique())))
        weights = compute_class_weight("balanced", classes=mapping, y=y)
        # if self.to_slice_dset.get_label_cutoff()["dampen"]:

        for i in range(len(weights)):
            clss_name = mapping[i]
            clss_id = full_mapping[clss_name]
            final_weights[clss_id] = weights[i]

        final_weights[not_in] = np.mean(final_weights)

        # For verb we will have all classes entered, noun has BG on pos zero
        cutoff = self.to_slice_dset.get_label_cutoff()
        final_weights = final_weights ** cutoff.get("dampen", cutoff.get("dampen_" + clazz))

        return final_weights

    def get_no_verbs(self):
        return self.to_slice_dset.get_no_verbs()

    def get_no_nouns(self):
        return self.to_slice_dset.get_no_nouns()

    def get_nouns(self):
        return set(self.noun_mapping.keys())

    def get_verbs(self):
        return set(self.verb_mapping.keys())

    def get_samples(self, no_samples, seed, collate_fn=None):
        rng = np.random.default_rng(seed)
        samples_idxs = rng.integers(low=0, size=min(no_samples, len(self)), high=len(self))
        samples = [self[idx] for idx in samples_idxs]
        sample_annots = self.nao_annots.iloc[samples_idxs]

        dicty = {}
        if collate_fn:
            samples = collate_fn(samples)

        dicty["inputs"] = samples
        try:
            dicty["nao_narration"] = sample_annots["nao_narration"]
        except:
            dicty["nao_narration"] = sample_annots["narration"]

        dicty["nao_clip_id"] = sample_annots["nao_clip_id"]
        dicty["frame"] = sample_annots.index

        try:
            dicty["eta"] = sample_annots["det_diff"].str[0]
        except:
            try:
                dicty["eta"] = sample_annots["det_diff"]
            except:
                dicty["eta"] = "-1"

        if "boxes" not in dicty["inputs"]:
            try:
                dicty["boxes"] = sample_annots["Bboxes"]
            except:
                dicty["boxes"] = [[0.0, 0.0, 1.0, 1.0]]
        else:
            dicty["boxes"] = [box.astype(int) for box in dicty["inputs"]["boxes"]]

        return dicty

    def set_flow_transform(self, transform):
        self.to_slice_dset.set_flow_transform(transform)

    def get_noun_verb_frequencies(self, allow_cache=True):
        if self.noun_verb_frequencies is not None and allow_cache:
            return self.noun_verb_frequencies
        
        ret = {}
        for row_idx, annot in self.nao_annots.iterrows():
            noun_ids = [self.noun_mapping[i] for i in annot["all_nouns"]]
            verb_ids = [self.verb_mapping[i] for i in annot["all_verbs"]]

            for verb_id, noun_id in zip(verb_ids, noun_ids):
                if noun_id not in ret:
                    ret[noun_id] = {}
                ret[noun_id][verb_id] = ret[noun_id].get(verb_id, 0) + 1
            
        if allow_cache:
            self.noun_verb_frequencies = ret
        return ret


class MergedNaoDataset(Dataset):
    def __init__(self, datasets):
        super().__init__()

        self.datasets = datasets
        for source, dataset in self.datasets.items():
            dataset.nao_annots["source"] = source

        self.label_cutoff = list(self.datasets.values())[0].get_label_cutoff()

        self.nao_annots = pd.concat([dataset.get_nao_annots() for dataset in self.datasets.values()], axis=0)
        self.readers = {}
        for dataset in self.datasets.values():
            self.readers.update(dataset.readers)

        use_external_label_mapping = any(dataset.use_external_label_mapping for dataset in self.datasets)

        self.verb_mapping = get_label_mapping(self.nao_annots["verb"], "verb", list(self.datasets.keys()),
                                              use_external_label_mapping)
        self.noun_mapping = get_label_mapping(self.nao_annots["noun"], "noun", list(self.datasets.keys()),
                                              use_external_label_mapping)

    def drop_cutoff_classes(self):
        return self.label_cutoff["drop"]

    def get_label_cutoff(self):
        return self.label_cutoff

    def get_verb_mapping(self):
        return self.verb_mapping

    def get_noun_mapping(self):
        return self.noun_mapping

    def get_img_shape(self, source):
        return self.datasets[source].get_img_shape()

    def get_no_verbs(self):
        return len(self.verb_mapping)

    def get_nao_annots(self):
        return self.nao_annots

    def get_annotations_df(self):
        return pd.concat([dataset.get_annotations_df() for dataset in self.datasets.values()])

    def get_annot_by_idx(self, idx):
        annot = self.nao_annots.iloc[idx]
        return annot

    def get_no_nouns(self):
        return len(self.noun_mapping)

    def set_resize_fn(self, resize_fn):
        for dataset in self.datasets.values():
            dataset.set_resize_fn(resize_fn)

    def set_input_transforms(self, input_transforms):
        for dataset in self.datasets.values():
            dataset.set_input_transforms(input_transforms)

    def set_hmap_transforms(self, hmap_transforms):
        for dataset in self.datasets.values():
            dataset.set_hmap_transforms(hmap_transforms)

    def set_flow_transform(self, transform):
        for dataset in self.datasets.values():
            dataset.set_flow_transform(transform)

    def get_narration_embeds(self):
        my_datasets = list(self.datasets.values())
        com_ds = my_datasets[0]
        # get number of embeddings from 1st dataset
        nr_embeds_1st = len(com_ds.narr_to_idx)

        # make second to be indexed from where we left of
        narr_dataset_2nd = my_datasets[1]
        narr_dataset_2nd.set_narr_to_idx_mapping(idx_offset=nr_embeds_1st)

        # skip the padding embed,we will take it from the second one
        embeds_1st = com_ds.get_narration_embeds()[:-1]
        embeds_2nd = narr_dataset_2nd.get_narration_embeds()
        embeds = np.concatenate([embeds_1st, embeds_2nd], axis=0)
        return embeds


    def get_raw_item(self, idx):
        annot = self.nao_annots.iloc[idx]
        annot_nao_clip_id = annot["nao_clip_id"]

        dataset = self.datasets[annot["source"]]
        idx = np.argwhere((dataset.nao_annots["nao_clip_id"] == annot_nao_clip_id).values)[0][0]
        outs = dataset.get_raw_item(idx)
        return outs

    def __len__(self):
        return len(self.nao_annots)

    def __getitem__(self, idx):
        annot = self.nao_annots.iloc[idx]
        annot_nao_clip_id = annot["nao_clip_id"]

        dataset = self.datasets[annot["source"]]
        idx = np.argwhere((dataset.nao_annots["nao_clip_id"] == annot_nao_clip_id).values)[0][0]
        outs = dataset[idx]

        return outs

