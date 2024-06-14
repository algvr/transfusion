import argparse
from functools import reduce

import json
import numpy as np
import pandas as pd
from data_preprocessing.utils.path_utils import (
    get_actors,
    get_annotations_df,
    get_full_nao_annotations,
    get_paper_nao,
)
    
from data_preprocessing.train_test_splits.utils import get_snao_split_name
from runner.utils.envyaml_wrapper import EnvYAMLWrapper
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import wandb
from data_preprocessing.utils.dataset_utils import merge_labels
from data_preprocessing.utils.path_constants import SPLITS_DIR, data_roots
from data_preprocessing.train_test_splits.utils import read_egtea_splits, splits_to_df

STRATIFY_COL = "noun"
GROUP_COL = "video_id"


def get_train_test_splits(samples, seed, type="group_stratified", nr_splits=1, y_col=STRATIFY_COL, group_col=GROUP_COL):
    if type == "group_stratified":
        return stratified_group_split(samples, y_col=y_col, group_col=group_col, seed=seed, nr_splits=nr_splits)
    elif type == "stratified":
        return stratified_split(samples, y_col=y_col, seed=seed, nr_splits=nr_splits)
    else:
        raise ValueError(f"Split {type=} not implemented")


def stratified_split(samples, y_col, seed, nr_splits=1):
    """Perform simple stratified split. Y_col proportion should be equal in the splits, clips can be mixed s.t.
    clips from a video appear in both splits."""
    x = np.arange(len(samples))
    y = samples[y_col]
    cv = StratifiedKFold(random_state=seed, n_splits=7, shuffle=True)
    split = cv.split(x, y)

    return [(x[train_index], x[test_index]) for train_index, test_index in list(split)[:nr_splits]]


def stratified_group_split(samples, y_col, group_col, seed, nr_splits=3):
    """Perform group stratified split. Y_col proportion should be equal in the splits, with the constraint that all clips
    of a video must belong to the same split (or what group col represents)"""
    x = np.arange(len(samples))
    y = samples[y_col]
    group = samples[group_col]
    cv = StratifiedGroupKFold(random_state=seed, n_splits=6, shuffle=True)
    split = cv.split(x, y, groups=group)

    return [(x[train_index], x[test_index]) for train_index, test_index in list(split)[:nr_splits]]


def get_all_nao(dataset, data_root, label_version=""):

    annotations = get_annotations_df(dataset, data_root)
    if label_version in {"paper", "paper_full"}:
        all_nao_annots = get_paper_nao(annotations, frame_diff=None)
    else:
        actors = get_actors(data_root, dataset, "all")
        all_nao_annots = {}
        for actor in actors:
            all_nao_annots.update(get_full_nao_annotations(data_root, dataset, actor, label_version))
        all_nao_annots = pd.concat(all_nao_annots.values())

    all_nao_annots = (
        pd.merge(
            all_nao_annots.reset_index(),
            annotations.reset_index()[["narration_id", "video_id", "noun", "verb"]],
            left_on="nao_clip_id",
            right_on="narration_id",
        )
        .drop("narration_id", axis=1)
        .set_index("video_id")
    )
    return all_nao_annots


def egtea_train_test_split(split_type, data_root, merging_dict, seed, label_version):
    splits = []
    all_nao_annots = get_all_nao("egtea", data_root, label_version=label_version)
    all_nao_annots = merge_labels(all_nao_annots, merging_dict)

    if split_type == "stratified":
        for split_no in [1, 2, 3]:
            egtea_train_split = read_egtea_splits(data_root.joinpath(f"annotations/train_split{split_no}.txt"))
            df_for_split = (
                all_nao_annots[all_nao_annots["nao_clip_id"].isin(egtea_train_split.index.tolist())]
                .drop_duplicates("nao_clip_id")
                .reset_index()
                .set_index("nao_clip_id")
            )

            x_train, x_val = get_train_test_splits(df_for_split, type=split_type, seed=seed, nr_splits=1)[0]
            snao_train_split, snao_val_split = df_for_split.iloc[x_train], df_for_split.iloc[x_val]

            egtea_test_split = read_egtea_splits(data_root.joinpath(f"annotations/test_split{split_no}.txt"))
            snao_test_split = (
                all_nao_annots[all_nao_annots["nao_clip_id"].isin(egtea_test_split.index.tolist())]
                .drop_duplicates("nao_clip_id")
                .set_index("nao_clip_id")
            )

            split_df = splits_to_df(snao_train_split, snao_val_split, snao_test_split)
            assert len(snao_train_split.index.intersection(snao_test_split.index)) == 0
            assert len(snao_train_split.index.intersection(snao_val_split.index)) == 0
            assert len(snao_val_split.index.intersection(snao_test_split.index)) == 0
            split_df.rename(columns={"subset": f"subset_{split_no-1}"}, inplace=True)
            splits.append(split_df)

    elif split_type == "group_stratified":
        egtea_train_split = read_egtea_splits(data_root.joinpath(f"annotations/train_split1.txt"))
        egtea_test_split = read_egtea_splits(data_root.joinpath(f"annotations/test_split1.txt"))
        egtea_clips = pd.concat([egtea_train_split, egtea_test_split])
        df_for_split = (
            pd.merge(
                egtea_clips[["video_id"]],
                all_nao_annots.reset_index(drop=True),
                left_index=True,
                right_on="nao_clip_id",
            )
            .drop_duplicates("nao_clip_id")
            .reset_index()
            .set_index("nao_clip_id")
        )

        nao_splits = get_train_test_splits(df_for_split, seed=seed, type=split_type, nr_splits=3)
        for i, (x_train_val, x_test) in enumerate(nao_splits):
            snao_test_split = df_for_split.iloc[x_test]
            train_val_df = df_for_split.iloc[x_train_val]
            x_train, x_val = get_train_test_splits(train_val_df, seed, type=split_type, nr_splits=1)[0]
            snao_train_split = train_val_df.iloc[x_train]
            snao_val_split = train_val_df.iloc[x_val]

            assert len(snao_train_split.index.intersection(snao_test_split.index)) == 0
            assert len(snao_train_split.index.intersection(snao_val_split.index)) == 0
            assert len(snao_val_split.index.intersection(snao_test_split.index)) == 0

            split_df = splits_to_df(snao_train_split, snao_val_split, snao_test_split)
            split_df.rename(columns={"subset": f"subset_{i}"}, inplace=True)
            splits.append(split_df)

    merge_save_splits(split_type, data_root, seed, label_version, splits)


def merge_save_splits(split_type, data_root, seed, label_version, splits):
    splits_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), splits)
    splits_df = splits_df.drop(columns=["nao_narration_x", "nao_narration_y"])
    splits_df.to_csv(
        data_root.joinpath(SPLITS_DIR, get_snao_split_name(seed, label_version, split_type, strat_col=STRATIFY_COL)),
        index=True,
    )


def epic_train_test_split(split_type, data_root, merging_dict, seed, label_version):
    splits = []
    epic_df = get_annotations_df("epic", data_root)

    epic_nao_annots = get_all_nao("epic", data_root, label_version=label_version)
    epic_nao_annots = merge_labels(epic_nao_annots, merging_dict)

    df_for_split = (
        epic_nao_annots[epic_nao_annots["nao_clip_id"].isin(epic_df.index.tolist())]
        .drop_duplicates("nao_clip_id")
        .reset_index()
        .set_index("nao_clip_id")
    )

    nao_splits = get_train_test_splits(df_for_split, seed=seed, type=split_type, nr_splits=3)
    for i, (x_train_val, x_test) in enumerate(nao_splits):
        snao_test_split = df_for_split.iloc[x_test]
        train_val_df = df_for_split.iloc[x_train_val]
        x_train, x_val = get_train_test_splits(train_val_df, seed, type=split_type, nr_splits=1)[0]
        snao_train_split = train_val_df.iloc[x_train]
        snao_val_split = train_val_df.iloc[x_val]

        assert len(snao_train_split.index.intersection(snao_test_split.index)) == 0
        assert len(snao_train_split.index.intersection(snao_val_split.index)) == 0
        assert len(snao_val_split.index.intersection(snao_test_split.index)) == 0

        split_df = splits_to_df(snao_train_split, snao_val_split, snao_test_split)
        split_df.rename(columns={"subset": f"subset_{i}"}, inplace=True)
        splits.append(split_df)

    merge_save_splits(split_type, data_root, seed, label_version, splits)


def ego4d_train_test_split(split_type, data_root, merging_dict, seed, label_version):
    splits = []
    ego_df = get_annotations_df("ego4d", data_root).reset_index().set_index("nao_clip_id")
    ego_df_test = ego_df[ego_df["orig_split"] == "test"]
    ego_df = ego_df[ego_df["orig_split"] != "test"]

    # ego_df = merge_labels(ego_df, merging_dict)
    nao_splits = get_train_test_splits(ego_df, seed=seed, type=split_type, nr_splits=3, group_col="clip_uid")
    for i, (x_train, x_val) in enumerate(nao_splits):
        snao_test_split = ego_df_test
        snao_train_split = ego_df.iloc[x_train]
        snao_val_split = ego_df.iloc[x_val]

        assert len(snao_train_split.index.intersection(snao_test_split.index)) == 0
        assert len(snao_train_split.index.intersection(snao_val_split.index)) == 0
        assert len(snao_val_split.index.intersection(snao_test_split.index)) == 0

        split_df = splits_to_df(snao_train_split, snao_val_split, snao_test_split)
        split_df.rename(columns={"subset": f"subset_{i+1}"}, inplace=True)
        splits.append(split_df)

    merge_save_splits(split_type, data_root, seed, label_version, splits)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obtain nao labeling from already extracted bboxes with UniDet model.")
    parser.add_argument(
        "--config",
        type=str,
        help="config file to run the program",
        default="../configs/snao_train_test_split.yml",
    )

    args = parser.parse_args()

    snao_split_cfg = EnvYAMLWrapper(args.config)
    datasets = snao_split_cfg["dataset"]
    split_types = snao_split_cfg["split_type"]
    seed = snao_split_cfg["seed"]
    label_version = snao_split_cfg["label_version"]
    label_merging_path = snao_split_cfg["label_merging_path"]
    with open(label_merging_path) as fp:
        merging_dict = json.load(fp)

    wandb_run = wandb.init(
        project="thesis",
        config=snao_split_cfg.yaml_config,
        entity="razvanp",
        name=get_snao_split_name(seed, label_version, split_types, STRATIFY_COL),
    )

    for dset in datasets:
        data_root = data_roots[dset]
        for split_type in split_types:
            if dset == "egtea":
                egtea_train_test_split(split_type, data_root, merging_dict["egtea"], seed, label_version)
            elif dset == "epic":
                epic_train_test_split(split_type, data_root, merging_dict["epic"], seed, label_version)
            elif dset == "ego4d":
                ego4d_train_test_split(split_type, data_root, merging_dict["ego4d"], seed, label_version)
            else:
                raise NotImplementedError()

            split_name = get_snao_split_name(seed, label_version, split_type)
            split_path = data_root.joinpath(SPLITS_DIR, split_name)
            artifact = wandb.Artifact(f"{dset}_{split_name}", type="split")
            artifact.add_file(split_path)
            wandb_run.log_artifact(artifact)

    artifact = wandb.Artifact("label_merging", type="label_merging")
    artifact.add_file(label_merging_path)
    wandb_run.log_artifact(artifact)
