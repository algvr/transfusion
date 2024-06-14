import argparse
import json
import logging
import os
import sys
from pathlib import Path


# base = os.path.dirname(sys.path[0])

# # sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

# sys.path.append(base)
# sys.path.append(os.path.dirname(base))
# sys.path.append(os.path.dirname(os.path.dirname(base)))
# sys.path.append(os.path.join(base, "detectron2"))
# sys.path.append(os.path.join(base, "detectron2", "utils"))
# sys.path.append(os.path.join(base, "detectron2", "projects", "UniDet", "demo"))
# sys.path.append(os.path.join(base, "detectron2", "projects", "UniDet", "predictor"))

import pandas as pd
import torch
from Code.data_preprocessing.utils.cfg_utils import my_setup_cfg, setup_logger, unidet_from_cfg
from Code.data_preprocessing.utils.dataset_utils import collate_unidet_input
from Code.data_preprocessing.utils.path_utils import data_roots, get_actors, get_path_to_actor, get_videos_for_actor
from Code.data_preprocessing.datasets.readers import ImageDataset, get_image_reader
from Code.data_preprocessing.label_extraction.utils import detections_to_pd_row
from Code.runner.utils.envyaml_wrapper import EnvYAMLWrapper
from natsort import natsorted
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_extraction_videos_left(path_to_actor, videos_to_filter, dataset_name, frame_stride):
    filtered_videos = []

    for video in natsorted(videos_to_filter):
        labels_csv_path = get_detections_csv_path(path_to_actor, video.name)
        if not labels_csv_path.exists():
            logging.warning(f"Extracting labels for {video}. Csv does not exist.")
            filtered_videos.append(video)
            continue

        # nr_frames_in_video = len(get_image_reader(dataset_name)(video.parent, video.name, frame_stride))
        # labels_pd = pd.read_csv(labels_csv_path, index_col=1)
        # if len(labels_pd) != nr_frames_in_video:
        #     filtered_videos.append(video)
        #     logging.warning(f"Extracting labels for {video}. Csv is incomplete")
        # else:
        #     logging.warning(f"Skippin labels for {video}. Already extracted")

    return filtered_videos


def get_detections_csv_path(path_to_actor, video_name):
    assert isinstance(video_name, str)
    video = video_name.replace(".mp4", "")
    csv_path = Path(f"{path_to_actor}/{video}_detections.csv")
    return csv_path


def extract_labels_for_actor(actor, label_extraction_cfg, unidet_cfg, model, force):
    labels_file = json.load(open(os.path.expandvars(unidet_cfg.MULTI_DATASET.UNIFIED_LABEL_FILE)))

    data_root = data_roots[label_extraction_cfg["dataset"]]

    path_to_actor = get_path_to_actor(data_root, label_extraction_cfg["dataset"], actor)
    all_actor_videos = get_videos_for_actor(path_to_actor, label_extraction_cfg["dataset"])

    if not force:
        videos_left = get_extraction_videos_left(
            path_to_actor, all_actor_videos, label_extraction_cfg["dataset"], label_extraction_cfg["frame_stride"]
        )
        if not videos_left:
            logging.info(f"Skipping {actor=}, all videos were extracted.")
            return
    else:
        videos_left = all_actor_videos

    for video in videos_left:
        logging.info(f"Extracting labels for {actor=}, {video=}")
        extract_labels_for_actor_and_video(
            path_to_actor=path_to_actor,
            video=video,
            label_extraction_cfg=label_extraction_cfg,
            unidet_cfg=unidet_cfg,
            labels_file=labels_file,
            model=model,
        )


def extract_labels_for_actor_and_video(path_to_actor, video, label_extraction_cfg, unidet_cfg, labels_file, model):
    csv_path = get_detections_csv_path(path_to_actor, video.name)

    img_reader = get_image_reader(label_extraction_cfg["dataset"])(
        path_to_actor, video, label_extraction_cfg["frame_stride"]
    )
    img_dataset = ImageDataset(unidet_cfg, img_reader)

    bs = label_extraction_cfg["bs"]
    num_workers = label_extraction_cfg["num_workers"]

    dataloader = DataLoader(
        img_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_unidet_input,
    )

    all_rows = [["Frame_no", "Classes", "Scores", "Bboxes"]]
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(img_dataset) // bs):
            img_batch, metadata = batch
            preds = model(img_batch)
            batch_pd_rows = detections_to_pd_row(preds, labels_file, metadata)
            all_rows.extend(batch_pd_rows)

            # if (i + 1) % 500 == 0:
            #     pd_labels = pd.DataFrame(all_rows[1:], columns=all_rows[0])
            #     pd_labels.to_csv(csv_path, index=False)

    pd_labels = pd.DataFrame(all_rows[1:], columns=all_rows[0])
    pd_labels.to_csv(csv_path, index=False)
    logging.info(f"Saved predictions csv to {csv_path}")


def main():
    setup_logger()
    parser = argparse.ArgumentParser(description="Extract bboxes using UniDet model.")
    parser.add_argument(
        "--config",
        type=str,
        help="config file to run the program",
        default="../configs/label_extraction.yml",
    )
    parser.add_argument("--actors", nargs="+", help="actors for which to extract")
    parser.add_argument("--videos", nargs="+", help="videos/recipes for which to extract")
    parser.add_argument("--force", action="store_true", default=False, help="to rerun bbox extraction")
    args = parser.parse_args()

    label_extraction_cfg = EnvYAMLWrapper(args.config, strict=False)
    data_root = data_roots[label_extraction_cfg["dataset"]]

    unidet_cfg = my_setup_cfg(label_extraction_cfg["model_cfg"], conf=0.4)
    model = unidet_from_cfg(unidet_cfg)

    actors = args.actors
    if not args.actors:
        actors = label_extraction_cfg["actors"]
    actors = get_actors(data_root, label_extraction_cfg["dataset"], actors)
    logging.info(f"Existing actors for {label_extraction_cfg['dataset']} , {actors}")

    for actor in actors:
        extract_labels_for_actor(
            actor, model=model, label_extraction_cfg=label_extraction_cfg, unidet_cfg=unidet_cfg, force=args.force
        )


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
