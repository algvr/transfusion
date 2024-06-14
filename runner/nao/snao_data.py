import logging
import pandas as pd

from data_preprocessing.datasets.snao_datasets import MergedNaoDataset, SNaoDataset
from data_preprocessing.datasets.commons import SnaoIdSlicer
from data_preprocessing.utils.path_utils import data_roots
from data_preprocessing.train_test_splits.utils import load_train_test_split, subsample_split
from modeling.hand_pos_dataset import HandPosDataset
from modeling.narration_embeds.collate_wrapper_utils import get_narr_wrapper
from modeling.narration_embeds.datasets.slowfast_features_dsets import PrevSlowfastFeatures
from modeling.narration_embeds.datasets.resnet_features_dsets import PrevRes50Features
from runner.utils.utils import get_datasets_from_name, get_label_merging, ALL_ACTORS, DEBUG_ACTORS


def get_snao_datasets(config):
    dataset_cfg = config["dataset"]
    dataset_args = dataset_cfg["args"]
    run_args = config["run"]
    debug = config["debug"]
    label_merging = get_label_merging(dataset_args["label_merging"])
    hand_args = run_args.get("hand_args", {"use": False})
    if debug:
        actors = DEBUG_ACTORS
    else:
        actors = ALL_ACTORS

    dataset_names = get_datasets_from_name(dataset_cfg["name"])
    if len(dataset_names) > 1:
        datasets = {}
        for dataset_name in dataset_names:
            dataset = SNaoDataset(
                root_data_path=data_roots[dataset_name],
                subset=None,
                offset_s=dataset_args["offset_s"],
                actors=actors[dataset_name],
                source=dataset_name,
                heatmap_type=run_args["heatmap_type"],
                label_merging=label_merging[dataset_name],
                label_cutoff=dataset_args["label_cutoff"],
                nao_version=dataset_args["nao_version"],
                coarse=dataset_args["coarse"],
                take_double=dataset_args["take_double"],
                action_rec=dataset_args.get("action_rec", False),
                narr_structure=dataset_args.get("narr_structure", "{gt_narr}"),
                narr_external_paths=dataset_args.get("narr_external_paths", []),
            )
            datasets[dataset_name] = dataset

        for d_name in datasets.keys():
            datasets[d_name] = get_narr_dataset_wrapper(datasets[d_name], [datasets[d_name]], run_args, d_name)[0]

            if hand_args["use"]:
                datasets[d_name] = HandPosDataset(datasets[d_name], hand_args)

        raw_nao_dataset = MergedNaoDataset(datasets)
    else:  # if we load just 1 dataset
        dset_type = dataset_cfg["name"]
        dataset_args["label_merging"] = label_merging[dset_type]
        root_data_path = data_roots[dset_type]
        raw_nao_dataset = SNaoDataset(
            root_data_path,
            subset=None,
            offset_s=dataset_args["offset_s"],
            actors=actors[dset_type],
            source=dset_type,
            heatmap_type=run_args["heatmap_type"],
            label_merging=label_merging[dset_type],
            label_cutoff=dataset_args["label_cutoff"],
            nao_version=dataset_args["nao_version"],
            take_double=dataset_args["take_double"],
            coarse=dataset_args["coarse"],
            action_rec=dataset_args.get("action_rec", False),
            narr_structure=dataset_args.get("narr_structure", "{gt_narr}"),
            narr_external_paths=dataset_args.get("narr_external_paths", []),
        )
        raw_nao_dataset = get_narr_dataset_wrapper(raw_nao_dataset, [raw_nao_dataset], run_args, dset_type)[0]

        if hand_args["use"]:
            datasets[d_name] = HandPosDataset(datasets[d_name], hand_args)

    return raw_nao_dataset


def apply_train_test_split(train_test_split_df, raw_dataset, egtea_test):
    if not egtea_test:
        train_clips = train_test_split_df[train_test_split_df["subset"] == "train"].index.tolist()
        train_dataset = SnaoIdSlicer(train_clips, raw_dataset)

        val_clips = train_test_split_df[train_test_split_df["subset"] == "val"].index.tolist()
        val_dataset = SnaoIdSlicer(val_clips, raw_dataset)

        test_clips = train_test_split_df[train_test_split_df["subset"] == "test"].index.tolist()
        if len(test_clips) == 0:
            logging.warn("Current test set has no clips!! Using first 1k val clips")
            test_clips = val_clips[:1000]

        test_dataset = SnaoIdSlicer(test_clips, raw_dataset)
    else:
        # we use in train subset the train and test parts
        not_egtea_df = train_test_split_df[train_test_split_df["source_ds"] != "egtea"]
        train_clips = not_egtea_df[not_egtea_df["subset"] != "val"].index.tolist()
        train_dataset = SnaoIdSlicer(train_clips, raw_dataset)

        # we keep validation as it is, minus 20% of ego4d clips that we add to train
        val_clips = not_egtea_df[not_egtea_df["subset"] == "val"]
        ego4d_sampled = val_clips[val_clips["source_ds"] == "ego4d"].sample(frac=0.2).index.tolist()
        train_clips += ego4d_sampled
        val_clips = val_clips[~val_clips.index.isin(ego4d_sampled)].index.tolist()

        val_dataset = SnaoIdSlicer(val_clips, raw_dataset)

        # and test only on egtea
        test_clips = train_test_split_df[train_test_split_df["source_ds"] == "egtea"].index.tolist()[:10]
        test_dataset = SnaoIdSlicer(test_clips, raw_dataset)

    assert len(set(raw_dataset.get_noun_mapping().keys()) - train_dataset.get_nouns()) == 0
    assert len(val_dataset.get_nouns() - train_dataset.get_nouns()) == 0
    assert len(val_dataset.get_verbs() - train_dataset.get_verbs()) == 0
    assert len(test_dataset.get_nouns() - train_dataset.get_nouns()) == 0
    assert len(test_dataset.get_verbs() - train_dataset.get_verbs()) == 0

    return train_dataset, val_dataset, test_dataset, train_test_split_df


def get_train_test_split(raw_nao_dataset, config):
    split_cfg = config["split"]
    dataset_cfg = config["dataset"]
    dataset_args = dataset_cfg["args"]
    subsample_spec = dataset_cfg["subsample"]
    seed = config["run"]["seed"]

    dataset_names = get_datasets_from_name(dataset_cfg["name"])
    if len(dataset_names) > 1:
        train_test_split_dfs = [
            load_train_test_split(split_cfg, source, dataset_args["nao_version"]) for source in dataset_names
        ]
        merged = pd.concat(train_test_split_dfs)
        train_test_split_df = subsample_split(merged, subsample_spec, seed)
    else:
        source = dataset_cfg["name"]
        train_test_split_df = load_train_test_split(split_cfg, source, dataset_args["nao_version"])
        train_test_split_df = subsample_split(train_test_split_df, subsample_spec, seed)

    train_dataset, val_dataset, test_dataset, train_test_split_df = apply_train_test_split(train_test_split_df, raw_nao_dataset, split_cfg["egtea_test"])

    if split_cfg.get("all_samples_as_val", False):
        val_dataset = SnaoIdSlicer(train_test_split_df[train_test_split_df["subset"] == "train"].index.tolist(), raw_nao_dataset)
    elif split_cfg.get("all_samples_as_train", False):
        train_dataset = SnaoIdSlicer(train_test_split_df[train_test_split_df["subset"] != "test"].index.tolist(), raw_nao_dataset)
        val_dataset = SnaoIdSlicer(train_test_split_df[train_test_split_df["subset"] == "test"].index.tolist(), raw_nao_dataset)

    return train_dataset, val_dataset, test_dataset, train_test_split_df


def get_narr_dataset_wrapper(raw_dataset, datasets, run_cfg, dset_name):
    nar_embed_args = run_cfg["narration_embeds"]

    if nar_embed_args["use"]:
        all_annots = raw_dataset.get_annotations_df()
        if dset_name in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            sort_by = "episode_action_id"
            stop_by = "clip_id"
        else:
            sort_by = "video_id"
            stop_by = "video_id"
        device = None if isinstance(run_cfg["devices"]["devices"], int) else run_cfg["devices"]["devices"][0]

        if nar_embed_args.get("slowfast_f", False):
            datasets = [
                PrevSlowfastFeatures(
                    dataset,
                    nar_embed_args["args"],
                    all_annots,
                    nar_embed_args.get("w_leak", True),
                    sort_by=sort_by,
                    stop_by=stop_by,
                    device=device,
                )
                for dataset in datasets
            ]

        if nar_embed_args.get("slowfast_f_v", False):
            datasets = [
                PrevSlowfastFeatures(
                    dataset,
                    nar_embed_args["args"],
                    all_annots,
                    nar_embed_args.get("w_leak", True),
                    sort_by=sort_by,
                    stop_by=stop_by,
                    device=device,
                    feature_key="visual_features"
                )
                for dataset in datasets
            ]

        if nar_embed_args.get("res50_f", False):
            datasets = [
                PrevRes50Features(
                    dataset,
                    run_cfg["flow_args"],
                    device=device,
                )
                for dataset in datasets
            ]

        # if we don't use slowfast features
        narr_wrapper = get_narr_wrapper(nar_embed_args["args"])

        uniq_narrations = all_annots["narration"].unique()
        datasets =  [
            narr_wrapper(
                dataset,
                nar_embed_args["args"],
                all_annots,
                sort_by=sort_by,
                stop_by=stop_by,
                device=device,
                uniq_narrations=uniq_narrations,
            )
            for dataset in datasets
        ]

    return datasets
