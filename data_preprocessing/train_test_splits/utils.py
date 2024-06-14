import numpy as np
import pandas as pd
from data_preprocessing.utils.annotations_df_utils import get_ego4d_annotations_df
from data_preprocessing.utils.path_constants import SPLITS_DIR, data_roots


def read_egtea_splits(split_path):
    data = [["video_id", "clip_id", "action_id", "verb_id", "noun_ids"]]
    with open(split_path, "r") as fp:
        lines = fp.readlines()
        for line in lines:
            splity = line.strip("\n").split(" ")
            video_id = "-".join(line.split("-")[:3])
            clip_id = "-".join(line.split("-")[:5])
            data.append([video_id, clip_id, splity[1], splity[2], ",".join(splity[3:])])

    df = pd.DataFrame(data=data[1:], columns=data[0]).set_index("clip_id")
    return df


def splits_to_df(train_df, val_df, test_df):
    train_df["subset"] = "train"
    val_df["subset"] = "val"
    test_df["subset"] = "test"
    df = pd.concat(
        [
            train_df[["subset", "nao_narration"]],
            val_df[["subset", "nao_narration"]],
            test_df[["subset", "nao_narration"]],
        ]
    )
    return df


def get_subsample_ratios(subsample_spec):
    if isinstance(subsample_spec, float) or subsample_spec == 1:
        return {"train": subsample_spec, "val": subsample_spec, "test": subsample_spec}
    elif isinstance(subsample_spec, str):
        splits = subsample_spec.split("_")
        return {"train": splits[0], "val": splits[1], "test": splits[2]}


def subsample_split(split_df, subsample_spec, seed, stratify="nao_narration"):
    train_split = split_df[split_df["subset"] == "train"]
    val_split = split_df[split_df["subset"] == "val"]
    test_split = split_df[split_df["subset"] == "test"]

    if subsample_spec == None:
        return pd.concat([train_split, val_split, test_split])

    if isinstance(subsample_spec, float) or subsample_spec == 1:
        no_train_samples = subsample_spec * len(train_split)
        no_val_samples = subsample_spec * len(val_split)
        no_test_samples = subsample_spec * len(test_split)
    else:
        splits = subsample_spec.split("_")
        no_train_samples, no_val_samples, no_test_samples = int(splits[0]), int(splits[1]), int(splits[2])

    train_split = stratified_sample(train_split, no_train_samples, seed, stratify)
    val_split = stratified_sample(val_split, no_val_samples, seed, stratify)

    if len(test_split) == 0:
        test_split = val_split
        no_test_samples = no_val_samples

    test_split = stratified_sample(test_split, no_test_samples, seed, stratify)
    return pd.concat([train_split, val_split, test_split])


def stratified_sample(to_split_df, no_samples, seed, stratify_col="nao_narration"):
    cls_counts = to_split_df[stratify_col].value_counts()
    cls_weights = (
        (cls_counts / len(to_split_df))
        .rename_axis("nao_narration")
        .reset_index(name="weights")
        .set_index("nao_narration")
    )
    cls_weights = to_split_df.merge(cls_weights, left_on="nao_narration", right_index=True)["weights"]
    cls_weights = cls_weights / cls_weights.sum()

    np.random.seed(seed)
    samples = np.random.choice(len(to_split_df), size=no_samples, replace=False, p=cls_weights)
    train_split = to_split_df.iloc[samples]
    return train_split


## loading utils


def get_snao_split_name(seed, label_version, split_type, strat_col=None):
    if strat_col:
        return f"snao_{split_type}_{label_version}_{seed}_{strat_col}.csv"
    else:
        return f"snao_{split_type}_{label_version}_{seed}.csv"


def load_train_test_split(split_cfg, dataset_name, nao_version):
    data_root = data_roots[dataset_name]
    split_version = split_cfg["version"]
    subset = split_cfg["subset"]
    split_type = split_cfg["type"]
    strat_col = split_cfg["strat_col"]

    if dataset_name not in {"ego4d", "ego4djpg", "ego4djpgv2"}:
        
        if split_type == "original":
            split_path_train = "/mnt/scratch/rpasca/Daatasets/EK/annotations/EPIC_100_train.csv"
            split_path_val = "/mnt/scratch/rpasca/Daatasets/EK/annotations/EPIC_100_validation.csv"

            split_df_val = pd.read_csv(split_path_val).rename(columns={"narration_id":"nao_clip_id"}).set_index("nao_clip_id")
            split_df_train = pd.read_csv(split_path_train).rename(columns={"narration_id":"nao_clip_id"}).set_index("nao_clip_id")
            split_df_val["subset"] = "val"
            split_df_train["subset"] = "train"
            split_df = pd.concat([split_df_train[["subset"]], split_df_val[["subset"]]])
        else:
            split_path = data_root.joinpath(
                SPLITS_DIR, get_snao_split_name(split_version, nao_version, split_type, strat_col)
            )
            split_df = pd.read_csv(split_path, index_col="nao_clip_id")[[f"subset_{subset}", "nao_narration"]]
            split_df = split_df.rename(mapper={f"subset_{subset}": "subset"}, axis=1)
            split_df["video_id"] = split_df.index.map(lambda x: " ".join(x.split("_")[:2]))

        if split_cfg.get("full_ek", False):
            test_pos = split_df["subset"] == "test"
            split_df.loc[test_pos, "subset"] = "train" 

    else:
        split_path = data_root.joinpath(
            SPLITS_DIR, get_snao_split_name(split_version, nao_version, split_type, strat_col)
        )
        annot_df = get_ego4d_annotations_df(data_root)

        if subset != 0:
            split_df = pd.read_csv(split_path, index_col="nao_clip_id")[[f"subset_{subset}", "nao_narration"]]
            split_df = split_df.rename(mapper={f"subset_{subset}": "subset"}, axis=1)
            split_df["video_id"] = annot_df.index.map(lambda x: "-".join(x.split("_")[:2]))

        else:
            split_df = annot_df[["orig_split", "nao_narration"]]
            split_df = split_df.rename(mapper={"orig_split": "subset"}, axis=1)
            split_df["video_id"] = split_df.index.map(lambda x: "-".join(x.split("_")[:-1]))

    split_df["source_ds"] = dataset_name
    return split_df
