import ast
import logging
from pathlib import Path

import json
from natsort.natsort import natsorted
import numpy as np
import os
import pandas as pd

from data_preprocessing.utils.path_constants import *
from data_preprocessing.utils.annotations_df_utils import (
    get_ego4d_annotations_df,
    ignore_labels
)


def get_path_to_actor(data_root, dataset_name, actor):
    if dataset_name == "epic" or dataset_name == "epicv":
        return Path(f"{data_root}/data/{actor}")
    elif dataset_name == "egtea":
        return Path(f"{data_root}/videos/{actor}")
    elif dataset_name == "ego4d":
        return Path(f"{data_root}/lmdb/{actor}")
    elif dataset_name == "ego4djpg":
        return Path(os.path.expanduser(f"{data_root}/object_frames")) 
    elif dataset_name == "ego4djpgv2":
        return Path(os.path.expanduser(f"{data_root}/object_frames"))
    else:
        raise ValueError(f"{dataset_name=} not recognized.")


def get_videos_for_actor(path_to_actor, dataset_name):
    videos = []

    if dataset_name.lower() == "epicv":
        path_to_videos = path_to_actor.joinpath("videos")
        for video in natsorted(list(path_to_videos.rglob("*.mp4"))):
            if video.name.startswith("P"):
                videos.append(video)

    elif dataset_name.lower() == "epic":
        path_to_videos = path_to_actor.joinpath("rgb_frames")
        for video in natsorted(list(path_to_videos.iterdir())):
            if video.name.startswith("P") and video.is_dir():
                videos.append(video)

    elif dataset_name.lower() == "egtea":
        for video in natsorted(list(path_to_actor.rglob("*.mp4"))):
            if video.name.startswith("P") or video.name.startswith("O"):
                videos.append(video)

    elif dataset_name.lower() == "ego4d":
        for video in natsorted(list(path_to_actor.iterdir())):
            videos.append(video)

    else:
        raise ValueError(f"{dataset_name=} not recognized")
    return videos


def apply_annot_structure(row, narr_structure, narr_external_dicts):
    frame_id = row.nao_clip_id
    if "P" in frame_id:
        frame_id = f'{row["video_id"]}_{"%07i" % row.name}'
        
    ret_annot = narr_structure
    replace_dict = {
        "gt_noun": row.noun,
        "gt_verb": row.verb,
        "gt_narr": row.narration,
        **{
            f"external_{narr_external_dict_idx}": narr_external_dict.get(frame_id, "")
            for narr_external_dict_idx, narr_external_dict in enumerate(narr_external_dicts)
        },
    }
    for k, v in replace_dict.items():
        ret_annot = ret_annot.replace("{" + k + "}", v)
    return " ".join(filter(str.__len__, ret_annot.split(" ")))  # eliminate multiple subsequent spaces


def get_annotations_df(
    dataset_name,
    data_root,
    video_epic=False,
    coarse=False,
    action_rec=False,
    narr_structure="{gt_narr}",
    narr_external_paths=[],
):
    """Loads the original data annotations for the dataset given in cfg file."""
    logging.info(f"Loading original annotations for {dataset_name=}, {data_root=}")

    if dataset_name == "ego4d":
        annotations_df = get_ego4d_annotations_df(data_root, action_rec=action_rec)

    elif dataset_name in {"ego4djpg", "ego4djpgv2"}:
        annotations_df = get_ego4d_annotations_df(data_root, resize_boxes=False, action_rec=action_rec)

    else:
        raise Exception(f"{dataset_name=} not recognized")

    # assert not annotations_df.isnull().values.any(), "Dataframe has null entries"
    assert not annotations_df["noun"].isnull().any()
    assert not annotations_df["verb"].isnull().any()

    # drop unwanted labels
    for key in ["noun", "verb"]:
        annotations_df = annotations_df[~annotations_df[key].isin(ignore_labels[dataset_name][key])]

    # structure annotations
    narr_external_dicts = []
    for narr_external_path in narr_external_paths:
        if not os.path.exists(narr_external_path):
            raise ValueError(f"{narr_external_path} does not exist")
            
        with open(narr_external_path, "r") as narr_external_file:
            narr_external_dicts.append(json.loads(narr_external_file.read()))

    if dataset_name in {"ego4d", "ego4djpg", "ego4djpgv2"}:
        annotations_df["narration"] = annotations_df.apply(
            lambda row: apply_annot_structure(row, narr_structure, narr_external_dicts), axis=1
        )

    return annotations_df


def read_detections_csv(labels_path, index="Frame_no"):
    """Loads Bboxes and object detection from UniDet model for the given actor/video combination"""
    logging.info(f"Loading extracted labels from {labels_path=}")

    df = pd.read_csv(
        labels_path,
        converters={
            "Scores": lambda x: np.array(ast.literal_eval(x)),
            "Classes": lambda x: np.array(ast.literal_eval(x)),
            "Bboxes": lambda x: np.array(ast.literal_eval(x)),
        },
    ).set_index(index)
    df = df[df["Classes"].str.len() > 0]
    df["Classes"] = df["Classes"].apply(lambda x: np.array([el.lower() for el in x]))
    return df


def read_detections_pkl(labels_path, index="Frame_no"):
    df = pd.read_pickle(labels_path).reset_index().set_index(index)
    df = df[df["Classes"].str.len() > 0]
    df["Classes"] = df["Classes"].apply(lambda x: np.array([el.lower() for el in x]))
    return df


def get_nao_path(path_to_actor, name, version, epic_video, ext="csv"):
    if version != "":
        if epic_video:
            csv_path = Path(f"{path_to_actor}/{name}_nao_{version}_video.{ext}")
        else:
            csv_path = Path(f"{path_to_actor}/{name}_nao_{version}.{ext}")
    else:
        csv_path = Path(f"{path_to_actor}/{name}_nao.{ext}")
    return csv_path

def get_actors(data_root, dataset_name, actors):
    if actors == "all":
        if dataset_name in {"ego4d", "ego4djpg", "ego4djpgv2"}:
            cache_path = {"ego4d": "${CODE}/data_preprocessing/configs/actor_cache_ego4d.json", "ego4djpg": "${CODE}/data_preprocessing/configs/actor_cache_ego4d.json", "ego4djpgv2": "${CODE}/data_preprocessing/configs/actor_cache_ego4dv2.json"}[dataset_name]
            cache_path = os.path.expandvars(cache_path)
            if os.path.isfile(cache_path):
                with open(cache_path, "r") as f:
                    actors = json.loads(f.read())
            else:
                try:
                    actors = [
                        actor_dir.stem
                        for actor_dir in Path(f"{data_root}/full_scale").iterdir()
                        if actor_dir.suffix == ".mp4"
                    ]
                except:
                    actors = [actor_dir.stem for actor_dir in Path(f"{data_root}/lmdb").iterdir()]
        else:
            actors = [
                actor_dir.name
                for actor_dir in Path(f"{data_root}/data").iterdir()
                if (actor_dir.name[0] == "P" or actor_dir.name[0] == "O")
            ]
        ret = natsorted(actors)

    else:
        ret = actors

    return ret


def get_full_nao_annotations(root_data_path, dataset_name, actor, label_version=""):
    actor_path = get_path_to_actor(root_data_path, dataset_name, actor)
    if label_version != "":
        pkl_path = actor_path.joinpath(f"{actor}_nao_{label_version}.pkl")
    else:
        pkl_path = actor_path.joinpath(f"{actor}_nao.pkl")
    nao_dfs = {}
    try:
        nao_dfs[actor] = pd.read_pickle(pkl_path)
        return nao_dfs
    except Exception as e:
        logging.warning(f"{e=} reading nao.pkl {actor=}")

    return nao_dfs


def get_matching_cols(row, annots_df, diff=100, nao_version="paper"):
    """Obtain the corresponding epic annotation metadata for the paper provided detections."""
    # first select only matching video
    match_videos = annots_df.loc[[row.name]]
    match_videos = match_videos[match_videos["start_frame"] > row["frame"]]
    if len(match_videos) == 0:
        return None

    ok_videos = match_videos[match_videos["old_narration"].map(lambda x: any(elem in x for elem in row["Classes"]))]
    if len(ok_videos) == 0:
        ok_videos = match_videos[
            match_videos["old_narration"].map(lambda x: any("".join(elem.split(" ")) in x for elem in row["Classes"]))
        ]
        if len(ok_videos) == 0:
            return None

    match_videos = ok_videos.iloc[0]
    if diff:
        if match_videos["start_frame"] - row["frame"] > diff:
            return None

    return pd.concat([row, match_videos], axis=0)


def get_short_term_annots(epic_video):
    short_term_annots = []
    annots_dir = data_roots["epic"].joinpath("annotations", "short_term_annots")
    for annot_csv in annots_dir.rglob("*.csv"):
        df = pd.read_csv(
            annot_csv,
            converters={
                "nao_bbox": lambda x: np.array([ast.literal_eval(x)]),
                "id": lambda x: x[3:],
                "label": lambda x: [str(x)],
            },
        )
        short_term_annots.append(df)

    short_term_annots = pd.concat(short_term_annots)
    short_term_annots = short_term_annots.rename(
        columns={"nao_bbox": "Bboxes", "id": "video_id", "label": "Classes"}
    ).set_index("video_id")

    if epic_video:
        const = 24 / 60
        short_term_annots["frame"] = np.floor(short_term_annots["frame"] * const).astype(int)

    return short_term_annots


def get_paper_nao(annots_df, frame_diff=150, epic_video=False, nao_version="paper"):
    assert nao_version in {"paper", "paper_full"}

    sorted_annots_df = annots_df.sort_values(by=["video_id", "start_frame"], ascending=[True, True])

    short_term_annots = get_short_term_annots(epic_video)
    short_term_annots.sort_values(by=["video_id", "frame"], ascending=[True, True], inplace=True)

    results = []
    for _, row in short_term_annots.iterrows():
        results.append(get_matching_cols(row, sorted_annots_df, frame_diff, nao_version))

    results = [x for x in results if x is not None]
    results_df = pd.concat(results, axis=1).T
    results_df = results_df.rename(
        columns={
            "narration": "nao_narration",
            "narration_id": "nao_clip_id",
            "frame": "Frame_no",
            "starting_ms": "starting_sec",
        }
    )
    results_df = results_df.drop(
        columns=[
            "verb",
            "participant_id",
            "narration_timestamp",
            "start_timestamp",
            "stop_timestamp",
            "verb_class",
            "noun",
            "noun_class",
            "all_nouns",
            "all_noun_classes",
        ]
    )

    # divide to get corresponding coords from 1920x1080 -> 456x256
    results_df["Bboxes"] = results_df["Bboxes"].map(lambda x: x / np.array([4.2187, 4.21, 4.2187, 4.21]))
    results_df["Bboxes"] = results_df["Bboxes"].map(lambda x: x.astype(int).tolist())
    results_df["det_sec"] = results_df["Frame_no"] / results_df["fps"]
    results_df["nao_start_sec"] = results_df["starting_sec"].values / 1000

    results_df = (
        results_df.reset_index()
        .groupby(["Frame_no", "nao_clip_id"])
        .agg(
            {
                "Classes": "sum",
                "Bboxes": "sum",
                # "video_id": "first",
                "start_frame": "first",
                "end_frame": "first",
                "nao_narration": "first",
                "nao_start_sec": "first",
                # "fps": "first",
                "det_sec": "first",
            }
        )
        .reset_index()
    )

    return results_df


def get_extracted_labels_actor_video(data_root, dataset_name, actor, video):
    path_to_actor = get_path_to_actor(data_root, dataset_name, actor)
    print(path_to_actor)
    if dataset_name == "egtea":
        labels_path = path_to_actor.joinpath(f"{actor}-{video}_detections.csv")
    else:
        labels_path = path_to_actor.joinpath(f"{actor}_{video}_detections.csv")

    labels_df = read_detections_csv(labels_path)
    return labels_df
