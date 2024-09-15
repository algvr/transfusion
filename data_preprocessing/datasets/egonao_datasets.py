from detectron2.data import transforms as T
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data.dataset import Dataset


from data_preprocessing.datasets.base_nao_dataset import NaoBase
from data_preprocessing.utils.dataset_utils import _get_readers


EGO_NAO_ANNOT_COLS_TO_KEEP = [
    "all_nouns",
    "all_verbs",
    "Bboxes",
    "det_diff",
    "orig_split",
    "nao_clip_id",
    "nao_narration",
    "episode_id",
    "start_frame",
    "narration",
    "video_id",
]

EGO_NAO_ANNOT_COLS_TO_KEEP_v2 = [
    "all_nouns",
    "all_verbs",
    "Bboxes",
    "det_diff",
    "orig_split",
    "nao_clip_id",
    "nao_narration",
    "episode_id",
    "start_frame",
    "narration",
    "video_uid",
]

NAO_ANNOT_COLS_TO_KEEP = {
    "ego4d": EGO_NAO_ANNOT_COLS_TO_KEEP,
    "ego4djpg": EGO_NAO_ANNOT_COLS_TO_KEEP,
    "ego4djpgv2": EGO_NAO_ANNOT_COLS_TO_KEEP_v2,
}


class EgoNaoDataset(Dataset, NaoBase):
    def __init__(
        self,
        root_data_path,
        subset,
        offset_s,
        actors,
        source,
        label_merging=None,
        label_cutoff=None,
        nao_version=1,
        take_double=False,
        coarse=False,
        action_rec=False,
        narr_structure="{gt_narr}",
        narr_external_paths=[],
        use_external_label_mapping=False,
        verb_bg=True,
    ):
        Dataset.__init__(self)
        NaoBase.__init__(
            self,
            root_data_path,
            subset,
            offset_s,
            actors=actors,
            source=source,
            label_merging=label_merging,
            label_cutoff=label_cutoff,
            nao_version=nao_version,
            heatmap_type="const",
            take_double=take_double,
            coarse=coarse,
            nao_annots_keep_cols=NAO_ANNOT_COLS_TO_KEEP[source],
            action_rec=action_rec,
            narr_structure=narr_structure,
            narr_external_paths=narr_external_paths,
            use_external_label_mapping=use_external_label_mapping,
        )

        self.readers = _get_readers(self.actors, self.source, self.root_data_path)
        self.verb_bg = verb_bg

    def __len__(self):
        return len(self.nao_annots)

    def get_nao_annots(self):
        return super().get_nao_annots()

    def get_no_nouns(self):
        return 1 + super().get_no_nouns()

    def get_no_verbs(self):
        no_verbs = super().get_no_verbs()
        if self.verb_bg:
            no_verbs = 1 + no_verbs
        return no_verbs

    def get_b_class_weights(self, clazz):
        mapping = self.noun_mapping.copy() if clazz == "noun" else self.verb_mapping.copy()
        mapping = list(mapping.keys())

        y = self.to_slice_dset.nao_annots[f"all_{clazz}s"].explode()
        weights = compute_class_weight("balanced", classes=mapping, y=y)
        cutoff = self.to_slice_dset.get_label_cutoff()
        weights = weights ** cutoff.get("dampen", cutoff.get("dampen_" + clazz))

        # set the weights on classes which are to be ignored to 0 and renormalized
        # done such that we still have the heatmap prediction while classification will not impact with noisy samples
        if self.to_slice_dset.get_label_cutoff()[clazz] and self.to_slice_dset.get_label_cutoff()["drop"] is True:
            ignore_labels = self.to_slice_dset.get_ignore_labels(clazz)
            ignore_idxs = [i for i, x in enumerate(mapping) if x in ignore_labels]
            if ignore_idxs:
                weights[np.array(ignore_idxs)] = 0
                weights = weights / weights.sum()

        return weights

    def get_classes(self, clazz):
        if self.source in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            return self.nao_annots[f"all_{clazz}s" if f"all_{clazz}s" in self.nao_annots else "all_verbs"].explode()
        else:
            return self.nao_annots[clazz].explode()

    def get_raw_item(self, idx):
        annot = self.nao_annots.iloc[idx]
        reader_dict = self.readers[annot[self.uid_col]]

        try:
            img = reader_dict["reader"].get_frame(annot.name)
        except:
            return self.get_raw_item(idx+1)
        orig_shape = img.shape
        bboxes = annot["Bboxes"]

        if self.resize_fn:
            inpy = T.AugInput(img)
            transf = self.resize_fn(inpy)
            bboxes = transf.apply_box(bboxes)
            img = inpy.image
        else:
            transf = None

        return {
            "image": img,
            "orig_shape": orig_shape[:2],
            "bboxes": bboxes,
            "annot": annot,
            "labels": np.array([self.noun_mapping[noun] for noun in annot["all_nouns"]]),
            "verb": np.array([self.verb_mapping[verb] for verb in annot["all_verbs"]]),
            "ttc": np.array([annot["det_diff"] for _ in annot["all_verbs"]]),
            "transf": transf,
            "id": annot["nao_clip_id"],
        }

    def convert_example(self, example):
        image = torch.Tensor(self.input_transform(example["image"].copy())).type(torch.float32)
        bboxes = self.localiz_transform(example["bboxes"]).type(torch.int32).squeeze(dim=0)
        bboxes = torch.from_numpy(example["bboxes"]).type(torch.float32)

        labels = torch.from_numpy(example["labels"]).type(torch.long)
        verb = torch.from_numpy(example["verb"]).type(torch.long)
        ttc = torch.from_numpy(example["ttc"]).type(torch.float32)

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "verb": verb,
            "ttc": ttc,
            "id": example["id"],
            "orig_shape": example["orig_shape"],
        }

    def __getitem__(self, idx):
        example = self.get_raw_item(idx)
        return self.convert_example(example)
