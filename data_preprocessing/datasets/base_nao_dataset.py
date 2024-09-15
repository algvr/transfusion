import copy
from joblib.parallel import Parallel, delayed
import json
import logging
import numpy as np
import os
import pandas as pd
import torch
from typing import Iterable

from data_preprocessing.utils.annotations_df_utils import timestamp_to_ms
from data_preprocessing.utils.dataset_utils import _filter_nao_annotations, drop_labels, get_label_mapping
from data_preprocessing.utils.math_utils import get_img_heatmap
from data_preprocessing.utils.path_utils import (apply_annot_structure,
                                                 get_actors,
                                                 get_annotations_df,
                                                 get_full_nao_annotations)


class NaoBase:
    def __init__(
        self,
        root_data_path,
        subset,
        offset_s,
        actors,
        source,
        heatmap_type,
        label_merging={},
        label_cutoff=None,
        nao_version=1,
        take_double=True,
        coarse=False,
        nao_annots_keep_cols=None,
        action_rec=False,
        narr_structure="{gt_narr}",
        narr_external_paths=[],
        use_external_label_mapping=False,
    ):
        self.root_data_path = root_data_path
        self.action_rec = action_rec
        self.source = source
        self.take_double = take_double
        if self.source == "ego4djpgv2":
            self.uid_col = "video_uid"
        else:
            self.uid_col = "video_id"
        self.annotations = get_annotations_df(
            self.source,
            self.root_data_path,
            self.source == "epicv",
            coarse=coarse,
            action_rec=self.action_rec,
            narr_structure=narr_structure,
            narr_external_paths=narr_external_paths,
        )
        self.offset_s = offset_s
        self.label_merging = label_merging
        self.label_cutoff = label_cutoff
        self.nao_version = nao_version
        self.heatmap_fn = get_img_heatmap(heatmap_type)
        self.heatmap_type = heatmap_type
        self.coarse = coarse
        self.heatmap_on = self.source not in ("ego4d", "ego4djpg", "ego4djpgv2")

        if not actors:
            self.actors = get_actors(self.root_data_path, self.source, "all")
        else:
            self.actors = actors
        assert isinstance(self.actors, Iterable)

        self.clip_subset = subset
        if self.clip_subset:
            self.annotations = self.annotations[self.annotations["episode_id"].isin(self.clip_subset)]

        if self.source not in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            self.nao_annots = self.load_nao_annotations(self.actors)
            self.nao_annots = self.filter_nao_annotations(self.nao_annots, offset_s)
            self.nao_annots = pd.concat(self.nao_annots.values())
            self.nao_annots = (
                pd.merge(
                    self.nao_annots.reset_index(),
                    self.annotations.reset_index()[["episode_id", "video_id", "verb", "noun", "narration"]],
                    left_on="nao_clip_id",
                    right_on="episode_id",
                )
                .drop("episode_id", axis=1)
                .set_index("Frame_no")
            )

            narr_external_dicts = []
            for narr_external_path in narr_external_paths:
                if not os.path.exists(narr_external_path):
                    raise ValueError(f"{narr_external_path} does not exist")
                with open(narr_external_path, "r") as narr_external_file:
                    narr_external_dicts.append(json.loads(narr_external_file.read()))

            self.nao_annots["narration"] = self.nao_annots.apply(
            lambda row: apply_annot_structure(row, narr_structure, narr_external_dicts), axis=1)

            if self.source == "epicv":
                # structure annotations
                const = 24 / 60
                self.nao_annots["orig_frame"] = self.nao_annots.index
                ratio = [456/640, 256/480, 456/640, 256/480] 
                self.nao_annots["Bboxes"] = self.nao_annots["Bboxes"].apply(lambda x: [[box[i]/ratio[i] for i in range(4)] for box in x])
                self.nao_annots.index = (self.nao_annots.index * const).astype(int)
               
                self.annotations["start_frame"] = np.floor(24 *  self.annotations["starting_ms"] / 1000).astype(int)
                self.annotations["end_frame"] = np.floor(
                    24 * self.annotations["stop_timestamp"].apply(timestamp_to_ms) / 1000
                ).astype(int)
             
        else:
            self.actors = self.annotations[self.uid_col].unique()
            self.annotations = self.annotations.reset_index().set_index("Frame_no")
            self.nao_annots = self.annotations

        if self.label_cutoff and self.label_cutoff["drop"]:
            self.nao_annots = drop_labels(self.nao_annots, self.label_cutoff)

        if self.source not in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            use_external_label_mapping = False

        self.noun_mapping = get_label_mapping(self.nao_annots["noun"], "noun", self.source, use_external_label_mapping)
        self.verb_mapping = get_label_mapping(self.nao_annots["verb"], "verb", self.source, use_external_label_mapping)

        assert len(self.nao_annots) > 0

        self.localiz_transform = lambda x: torch.from_numpy(x)
        self.input_transform = lambda x: x
        self.resize_fn = None

        if source in {"ego4d","ego4djpg","ego4djpgv2"}:
            nao_annots_keep_cols.append("episode_action_id")
            nao_annots_keep_cols.append("clip_id")
        else:
            nao_annots_keep_cols.append("video_id")
            if source == "epicv":
                nao_annots_keep_cols.append("orig_frame")

        if action_rec:
            nao_annots_keep_cols.append("nao_frame")

        if nao_annots_keep_cols:
            self.nao_annots_keep_cols = list(set(nao_annots_keep_cols))
            self.nao_annots = self.nao_annots[self.nao_annots_keep_cols]

    def drop_cutoff_classes(self):
        return self.label_cutoff["drop"]

    def get_verb_mapping(self):
        return self.verb_mapping

    def get_noun_mapping(self):
        return self.noun_mapping

    def get_label_cutoff(self):
        return self.label_cutoff

    def get_ignore_labels(self, col):
        counts = self.nao_annots[col].value_counts()
        ignore_counts = counts[counts < self.label_cutoff[col]].index.tolist()
        return ignore_counts

    def set_fg_bg_percs(self):
        self.nao_annots["fg_perc"] = 0.5
        self.nao_annots["bg_perc"] = 1 - self.nao_annots["fg_perc"]

    def set_resize_fn(self, resize_fn):
        self.resize_fn = resize_fn

    def set_input_transforms(self, transforms):
        self.input_transform = transforms

    def set_localiz_transforms(self, transforms):
        self.localiz_transform = transforms

    def get_classes(self, clazz):
        return self.nao_annots[clazz]

    def get_no_verbs(self):
        return len(self.verb_mapping)

    def get_no_nouns(self):
        return len(self.noun_mapping)

    def get_nao_annots(self):
        return copy.deepcopy(self.nao_annots)

    def get_annotations_df(self):
        return self.annotations

    def get_annot_by_idx(self, idx):
        annot = self.nao_annots.iloc[idx]
        return annot

    def load_nao_annotations(self, actors):
        nao_dfs = {}
        # set_loky_pickler("pickle")

        def read(actor):
            logging.info(f"Loading annotations for {actor}")
            actor_dfs = get_full_nao_annotations(self.root_data_path, self.source, actor, self.nao_version)
            return actor_dfs

        dfs = Parallel(n_jobs=4)(delayed(read)(actor) for actor in actors)
        for actor_df in dfs:
            nao_dfs.update(actor_df)

        return nao_dfs

    def filter_nao_annotations(self, nao_dfs, offset):
        for video, v_naos in nao_dfs.items():
            nao_dfs[video] = _filter_nao_annotations(v_naos, offset, self.take_double)

        return nao_dfs

    def get_img_shape(self):
        return list(self.readers.values())[0]["reader"].get_img_shape()

    def set_flow_transform(self, transform):
        pass

    def get_samples(self, no_samples, seed, collate_fn=None):
        rng = np.random.default_rng(seed)
        samples_idxs = rng.integers(low=0, size=min(no_samples, len(self)), high=len(self))
        samples = [self[idx] for idx in samples_idxs]
        sample_annots = self.nao_annots.iloc[samples_idxs]

        dicty = {}
        if collate_fn:
            samples = collate_fn(samples)

        dicty["inputs"] = samples
        dicty["nao_narration"] = sample_annots["nao_narration"]
        dicty["nao_clip_id"] = sample_annots["nao_clip_id"]
        dicty["frame"] = sample_annots.index
        try:
            dicty["eta"] = sample_annots["det_diff"].str[0]
        except:
            dicty["eta"] = sample_annots["det_diff"]

        if "boxes" not in dicty["inputs"]:
            dicty["boxes"] = sample_annots["Bboxes"]
        else:
            dicty["boxes"] = [box.astype(int) for box in dicty["inputs"]["boxes"]]

        return dicty
