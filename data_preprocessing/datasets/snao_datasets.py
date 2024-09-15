import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import swifter
import torch
from detectron2.data import transforms as T
from torch.utils.data.dataset import Dataset

from data_preprocessing.datasets.base_nao_dataset import NaoBase
from data_preprocessing.train_test_splits.utils import subsample_split
from data_preprocessing.utils.cfg_utils import setup_logger
from data_preprocessing.utils.dataset_utils import _filter_nao_annotations, _get_readers, get_label_mapping
from data_preprocessing.utils.path_constants import data_roots, paper_actors
from data_preprocessing.utils.path_utils import data_roots, get_paper_nao
from data_preprocessing.train_test_splits.utils import load_train_test_split
from runner.utils.data_transforms import get_norm_mean_std


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
        # if isinstance(com_ds, PrevNarrEmbedDataset) or isinstance(com_ds, PrevNarrIdxDataset):
        # get number of emebddings from 1st dataset
        nr_embeds_1st = len(com_ds.narr_to_idx)

        # make second to be indexed from where we left of
        narr_dataset_2nd = my_datasets[1]
        narr_dataset_2nd.set_narr_to_idx_mapping(idx_offset=nr_embeds_1st)

        # skip the padding embed,we will take it from the second one
        embeds_1st = com_ds.get_narration_embeds()[:-1]
        embeds_2nd = narr_dataset_2nd.get_narration_embeds()
        embeds = np.concatenate([embeds_1st, embeds_2nd], axis=0)
        return embeds
        # else:
        # raise ValueError()

    def get_pad_idx(self):
        com_ds = list(self.datasets.values())[0]
        # if isinstance(com_ds, PrevNarrEmbedDataset):
        return com_ds.get_pad_idx() + list(self.datasets.values())[1].get_pad_idx()
        # else:
        # raise ValueError()

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


class SNaoDataset(Dataset, NaoBase):
    def __init__(
        self,
        root_data_path,
        subset,
        offset_s,
        actors,
        source,
        heatmap_type="gaussian",
        label_merging=None,
        label_cutoff=None,
        nao_version=1,
        take_double=True,
        coarse=False,
        action_rec=False,
        narr_structure="{gt_narr}",
        narr_external_paths=[],
        use_external_label_mapping=False
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
            heatmap_type=heatmap_type,
            take_double=take_double,
            coarse=coarse,
            nao_annots_keep_cols=[
                "noun",
                "verb",
                "Bboxes",
                "det_diff",
                "orig_split",
                "nao_clip_id",
                "nao_narration",
                "video_id",
            ],
            action_rec=action_rec,
            narr_structure=narr_structure,
            narr_external_paths=narr_external_paths,
            use_external_label_mapping=use_external_label_mapping
        )
        self.readers = _get_readers(self.actors, self.source, self.root_data_path)
        self.set_fg_bg_percs()

    def __len__(self):
        return len(self.nao_annots)

    def get_nao_annots(self):
        return super().get_nao_annots()

    def get_raw_item(self, idx):
        annot = self.nao_annots.iloc[idx]
        reader_dict = self.readers[annot["video_id"]]

        img = reader_dict["reader"].get_frame(annot.name)

        if self.heatmap_on:
            heatmap = self.heatmap_fn(reader_dict["mapped_x"], reader_dict["mapped_y"], boxes=annot["Bboxes"])
        else:
            heatmap  = np.ones_like(img)

        if self.resize_fn:
            inpy = T.AugInput(img)
            transf = self.resize_fn(inpy)
            heatmap = transf.apply_image(heatmap)
            if self.heatmap_type == "const":
                heatmap = np.round(heatmap)

            img = inpy.image

        else:
            transf = None

        return {
            "image": img,
            "heatmap": heatmap,
            "annot": annot,
            "noun": self.noun_mapping[annot["noun"]],
            "verb": self.verb_mapping[annot["verb"]],
            "fg_perc": annot["fg_perc"],
            "bg_perc": annot["bg_perc"],
            "ttc": annot["det_diff"],
            "transf": transf,
        }

    def convert_example(self, example):
        image = torch.Tensor(self.input_transform(example["image"].copy())).type(torch.float32)
        heatmap = self.localiz_transform(example["heatmap"]).type(torch.float32).squeeze(dim=0)

        noun = torch.Tensor([example["noun"]]).type(torch.long)
        verb = torch.Tensor([example["verb"]]).type(torch.long)
        fg_perc = torch.tensor(example["fg_perc"]).type(torch.float32)
        bg_perc = torch.tensor(example["bg_perc"]).type(torch.float32)
        ttc = torch.tensor(example["ttc"]).type(torch.float32)

        return {
            "image": image,
            "heatmap": heatmap,
            "noun": noun,
            "verb": verb,
            "fg_perc": fg_perc,
            "bg_perc": bg_perc,
            "ttc": ttc,
        }

    def __getitem__(self, idx):
        example = self.get_raw_item(idx)
        return self.convert_example(example)


class PaperSNaoDataset(SNaoDataset):
    def __init__(
        self,
        root_data_path,
        subset,
        offset_s,
        actors,
        source,
        label_merging=None,
        label_cutoff=None,
        nao_version="paper_full",
        heatmap_type="gaussian",
        frame_diff=None,
    ):
        assert source in {"epic", "epic_v"}, "Paper dataset can only be used with EPIC data"
        assert nao_version in {"paper", "paper_full"}, "Paper dataset requires specific nao version"
        self.frame_diff = frame_diff

        super().__init__(
            root_data_path,
            subset,
            offset_s,
            actors=paper_actors,
            source=source,
            label_merging=label_merging,
            label_cutoff=label_cutoff,
            nao_version=nao_version,
            heatmap_type=heatmap_type,
        )

    def load_nao_annotations(self, actors):
        return {
            "all": get_paper_nao(
                self.annotations, frame_diff=self.frame_diff, epic_video=False, nao_version=self.nao_version
            )
        }

    def filter_nao_annotations(self, nao_dfs, offset):
        for video, v_naos in nao_dfs.items():
            nao_dfs[video] = _filter_nao_annotations(v_naos, offset, take_double=False)

        return nao_dfs


def run_test(dataset_name):
    setup_logger()
    nao_version, dataset = get_test_snao(dataset_name)
    print(f"No examples {len(dataset)}")

    img, heatmap, annot, noun, verb, _ = dataset[2].values()
    plt.imshow(img.numpy().astype(int))
    plt.imshow(heatmap.numpy(), alpha=0.6)
    plt.savefig(f"dummy_{dataset_name}.jpg")
    print(annot)

    run_cfg = {"heatmap_type": "gaussian", "hmap_scaling": 1, "normalization": "own", "dataset": dataset_name}
    aug_cfg = {
        "resize_spec": [192, 384],
        "crop_spec": [1, 1],
        "brightness": 0.0,
        "contrast": 0.0,
        "saturation": 0.0,
        "hue": 0.0,
    }
    mean, std = get_norm_mean_std(run_cfg["normalization"], dataset_name)
    denorm = get_denormalize(mean, std)
    set_transforms_to_dsets(dataset, [], run_cfg, aug_cfg, "snao")
    img, heatmap, annot, noun, verb, _ = dataset[2].values()
    img = denorm(img)
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.imshow(heatmap.numpy(), alpha=0.6)
    plt.savefig(f"dummy_{dataset_name}_transform.jpg")
    print(annot)

    split_cfg = {"subset": 0, "version": 42, "type": "group_stratified", "strat_col": None}
    train_test_split_df = load_train_test_split(split_cfg, dataset_name, nao_version)
    train_test_split_df = subsample_split(train_test_split_df, None, 42)

    for video_id in train_test_split_df["video_id"]:
        no_subsets = train_test_split_df[train_test_split_df["video_id"] == video_id]["subset"].nunique()
        assert no_subsets == 1


def get_test_paper_snao():
    label_merging_path = os.path.expandvars("$CODE/data_preprocessing/configs/label_merging.json")
    with open(label_merging_path, "r") as fp:
        label_merging_dict = json.load(fp)

    dataset_name = "epic"
    actors = ["P29"]
    nao_version = "paper_full"

    dataset = PaperSNaoDataset(
        data_roots[dataset_name],
        subset=None,
        offset_s=0.5,
        actors=actors,
        source=dataset_name,
        label_cutoff={"drop": False},
        label_merging=label_merging_dict[dataset_name],
        nao_version=nao_version,
    )

    return nao_version, dataset


def get_test_snao(dataset_name):
    label_merging_path = os.path.expandvars("$CODE/data_preprocessing/configs/label_merging.json")
    with open(label_merging_path, "r") as fp:
        label_merging_dict = json.load(fp)

    actors = {
        # "ego4d": ["9c59e912-2340-4400-b2df-7db3d4066723"],
        "ego4d": None,
        "epic": None,
        "egtea": None,
    }

    actors = actors[dataset_name]
    nao_version = "1"

    dataset = SNaoDataset(
        data_roots[dataset_name],
        subset=None,
        offset_s=0.0,
        actors=actors,
        source=dataset_name,
        heatmap_type="gaussian",
        label_cutoff={"drop": False},
        label_merging=label_merging_dict[dataset_name],
        nao_version=nao_version,
        coarse=True,
    )

    return nao_version, dataset


if __name__ == "__main__":
    from runner.utils.data_transforms import get_denormalize, set_transforms_to_dsets

    dataset = "ego4d"
    run_test(dataset)
