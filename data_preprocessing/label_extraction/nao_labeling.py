import argparse
import json
import logging
from pathlib import Path
from numpy import result_type

import pandas as pd
from Code.data_preprocessing.label_extraction.label_extraction import get_detections_csv_path
from Code.data_preprocessing.label_extraction.nao_labeling_cases import match_frame_in_annotation
from Code.data_preprocessing.label_extraction.utils import nms
from Code.data_preprocessing.utils.cfg_utils import setup_logger
from Code.data_preprocessing.utils.path_utils import (
    data_roots,
    get_actors,
    get_annotations_df,
    get_nao_path,
    get_path_to_actor,
    get_videos_for_actor,
    read_detections_csv,
)
from Code.runner.utils.envyaml_wrapper import EnvYAMLWrapper
from joblib import Parallel, delayed
from natsort import natsorted
from tqdm import tqdm


def get_nao_labels_for_video(detections_df, video_annotations, soft_matches, label_version):
    """Obtain NAO labels for a given video_id by going over all its annotations. Each annotation is compared
    to the detections obtained from the previous one to obtain the NAO labels."""
    video_annotations = video_annotations.sort_values(by=["start_frame"])
    annotation_fps = video_annotations["fps"].median()

    annotations_labels = []

    for ann_no, (_, annotation) in enumerate(video_annotations.iloc[1:].iterrows()):
        curr_start_frame = annotation["start_frame"]
        prev_start_frame = video_annotations.iloc[ann_no]["start_frame"]
        labels_slice = detections_df[prev_start_frame < detections_df.index]
        labels_slice = labels_slice[labels_slice.index < curr_start_frame]

        if len(labels_slice) == 0:
            continue

        nao_labels_annotation, matched_frames = get_nao_labels_for_annotation(
            labels_slice, annotation, soft_matches, label_version
        )
        nao_labels_annotation["nao_clip_id"] = annotation["narration_id"]
        nao_labels_annotation["nao_narration"] = annotation["narration"]
        nao_labels_annotation["nao_start_sec"] = annotation["starting_ms"] / 1000
        nao_labels_annotation["det_sec"] = nao_labels_annotation.index / annotation_fps

        annotations_labels.append(nao_labels_annotation)

    return pd.concat(annotations_labels)


def get_nao_labels_for_annotation(
    frames_slice_detections: pd.DataFrame, curr_annotation: pd.DataFrame, soft_matches: dict, version: int
):
    """Obtain NAO labels for an action annotation(segment) by searching for direct label correspondance and soft_matches.

    Args:
        frames_slice_detections (pd.DataFrame): Frames that correspond to the previous annotation, before
            curr_annotation annotation. If a frame has a detection that corresponds to a target label/soft match, it is
            considered a match.
        curr_annotation (pd.DataFrame): Current annotation. Holds narration, action, verb and noun classes.
        soft_matches (dict): Soft matches for most cooking classes. e.g. if cucumber should be found but is not, try to
            match with zuchinni.
        version (int): what version of label extraction to use. 1 is get first match, 2 also uses closest to hands,
        3 also uses take all for certain cases (furniture, vegetables etc)

    Returns:
        pd.DataFrame: DataFrame with frames that have matching/soft_matching detections. Considered as GT for the NAO objective.
    """
    if curr_annotation["narration_id"] in {"P16_01_11"}:
        print("debu")

    rows = [["Frame_no", "Classes", "Scores", "Bboxes"]]
    frames_pot_matches = pd.DataFrame(index=frames_slice_detections.index)
    frames_pot_matches["matches"] = frames_slice_detections.apply(
        match_frame_in_annotation, curr_annotation=curr_annotation, soft_matches=soft_matches, version=version, axis=1
    )

    non_empty_matches_idxs = frames_pot_matches["matches"].str.len() > 0
    non_empty_matches = frames_pot_matches[non_empty_matches_idxs]
    matched_frames = frames_slice_detections[non_empty_matches_idxs]

    for (index, labels_row), pair_row in zip(matched_frames.iterrows(), non_empty_matches["matches"]):
        # gives the entry frame coresp to this index with list of matching obj labels

        rows.append(
            [
                index,
                labels_row["Classes"][pair_row].tolist(),
                labels_row["Scores"][pair_row].tolist(),
                labels_row["Bboxes"][pair_row].tolist(),
            ]
        )

    df = pd.DataFrame(rows[1:], columns=rows[0]).set_index("Frame_no")
    return df, matched_frames


def get_nao_videos_left(path_to_actor, videos_to_filter, force, label_version):
    filtered_videos = []

    if force:
        for video in natsorted(videos_to_filter):
            detections_csv_path = get_detections_csv_path(path_to_actor, video.name)
            if detections_csv_path.exists():
                filtered_videos.append(video.stem)
            else:
                logging.warning(f"Skipping {video}. Detections were not extracted for it.")

    for video in natsorted(videos_to_filter):
        nao_labels_csv_path = get_nao_path(
            path_to_actor, video.name.replace(".mp4", ""), label_version, nao_labeling_cfg["epic_video"]
        )
        detections_csv_path = get_detections_csv_path(path_to_actor, video.name)
        if (not nao_labels_csv_path.exists()) and detections_csv_path.exists():
            logging.warning(f"Will compute labels for {video}. Csv {nao_labels_csv_path} does not exist.")
            filtered_videos.append(video.stem)
            continue

        if not detections_csv_path.exists():
            logging.warning(f"Skipping {video}. Detections were not extracted for it.")

    return filtered_videos


def get_nao_labels_for_actor_and_video(path_to_actor, actor, video_id, soft_matches, annotations_df, label_version):
    """Obtain NAO labels for all videos of a given actor"""
    annotations_df["all_nouns"] = annotations_df["all_nouns"].apply(
        lambda nouns: [x for x in nouns if x not in soft_matches["no_matches"]]
    )

    logging.info(f"Computing NAO labels for {actor=}, {video_id=}")
    labels_path = path_to_actor.joinpath(f"{video_id}_detections.csv")
    detections_df = read_detections_csv(labels_path)

    if label_version != 0:
        detections_df = detections_df.apply(nms, iou_lim=0.4, axis=1, result_type="expand")
        detections_df = detections_df.rename(columns={0: "Classes", 1: "Scores", 2: "Bboxes"})

    video_annotations = annotations_df.loc[[video_id]]
    # some videos have only one annotation in the offered splits.
    if len(video_annotations) > 1:
        nao_labels_video = get_nao_labels_for_video(detections_df, video_annotations, soft_matches, label_version)
    else:
        return

    csv_path = get_nao_path(path_to_actor, video_id.replace(".mp4", ""), label_version)
    nao_labels_video.to_csv(csv_path)
    pkl_path = f"{str(csv_path)[:-3]}pkl"
    nao_labels_video.to_pickle(pkl_path)
    logging.info(f"Saved NAO labels for {actor=}, {video_id=} in {csv_path=}")
    return nao_labels_video


def get_nao_labels_for_actor(data_root, nao_labeling_cfg, actor, force_videos, soft_matches, annotations_df):
    label_version = nao_labeling_cfg["parsing_version"]

    path_to_actor = get_path_to_actor(data_root, nao_labeling_cfg["dataset"], actor)
    all_actor_videos = get_videos_for_actor(path_to_actor, nao_labeling_cfg["dataset"])
    videos_left = get_nao_videos_left(path_to_actor, all_actor_videos, force_videos, label_version)

    if not videos_left:
        logging.info(f"Skipping {actor=}, all videos were extracted.")
        return

    # filter out videos which are in test set
    videos_left = [x for x in videos_left if x in set(annotations_df.index.unique())]
    if videos_left:
        for video_id in tqdm(videos_left):
            get_nao_labels_for_actor_and_video(
                path_to_actor, actor, video_id, soft_matches, annotations_df, label_version
            )

        ## read all video naos and concat them in a single pkl for faster loading
        all_actor_naos = []
        all_nao_pkl_path = get_nao_path(path_to_actor, actor, label_version, nao_labeling_cfg["epic_video"], ext="pkl")
        pkl_dfs_paths = path_to_actor.rglob(f"*_nao_{label_version}.pkl")
        for pkl_df_path in list(pkl_dfs_paths):
            if pkl_df_path.stem != all_nao_pkl_path.stem:
                all_actor_naos.append(pd.read_pickle(pkl_df_path))

        all_actor_naos = pd.concat(all_actor_naos)
        all_actor_naos.to_pickle(all_nao_pkl_path)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="Obtain nao labeling from already extracted bboxes with UniDet model.")
    parser.add_argument(
        "--config",
        type=str,
        help="config file to run the program",
        default="../configs/nao_labeling.yml",
    )
    parser.add_argument("--actors", nargs="+", help="actors for which to extract")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-compute nao labels even if they exist",
    )
    parser.add_argument("--no_jobs", type=int, default=1, help="number of jobs to parallelize")

    args = parser.parse_args()

    nao_labeling_cfg = EnvYAMLWrapper(args.config)
    data_root = data_roots[nao_labeling_cfg["dataset"]]
    with open(nao_labeling_cfg["soft_matches_file"], "r") as fp:
        soft_matches = json.load(fp)

    annotations_df = get_annotations_df(
        nao_labeling_cfg["dataset"], Path(data_root), video_epic=nao_labeling_cfg["epic_video"]
    )

    actors = args.actors
    if not args.actors:
        actors = nao_labeling_cfg["actors"]
    actors = get_actors(data_root, nao_labeling_cfg["dataset"], actors)
    logging.info(f"Computing on actors for {nao_labeling_cfg['dataset']}: {actors}")

    Parallel(n_jobs=args.no_jobs)(
        delayed(get_nao_labels_for_actor)(
            data_root,
            nao_labeling_cfg,
            actor,
            args.force,
            soft_matches[nao_labeling_cfg["dataset"]],
            annotations_df,
        )
        for actor in actors
    )
    print("Finished")
