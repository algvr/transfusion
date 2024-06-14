import argparse
from collections import defaultdict
import json
import numpy as np
import os
from os.path import dirname, expandvars, isfile, join
from pathlib import Path
from PIL import Image
import sys
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from data_preprocessing.datasets.readers import Ego4dDataReaderMp4


if __name__ == "__main__":
    if not os.environ.get("DATA"):
        os.environ["DATA"] = join(dirname(dirname(__file__)), "datasets")

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="ego4dv2")
    parser.add_argument("--full-scale-dir", type=str, default=None)
    parser.add_argument("--annotation-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.version.lower() in {"ego4d", "v1", "1"}:
        args.version = "ego4d"
        print(f"Extracting JPG files for Ego4D version 1")
        if args.full_scale_dir in {None, ""}:
            args.full_scale_dir = expandvars("${DATA}/Ego4d/v1/full_scale")
        if args.annotation_dir in {None, ""}:
            args.annotation_dir = expandvars("${DATA}/Ego4d/v1/annotations")
        if args.output_dir in {None, ""}:
            args.output_dir = expandvars("${DATA}/Ego4d/v1/object_frames")
    elif args.version.lower() in {"ego4dv2", "v2", "2"}:
        args.version = "ego4dv2"
        print(f"Extracting JPG files for Ego4D version 2")
        if args.full_scale_dir in {None, ""}:
            args.full_scale_dir = expandvars("${DATA}/Ego4d/v2/full_scale")
        if args.annotation_dir in {None, ""}:
            args.annotation_dir = expandvars("${DATA}/Ego4d/v2/annotations")
        if args.output_dir in {None, ""}:
            args.output_dir = expandvars("${DATA}/Ego4d/v2/object_frames")
    else:
        raise NotImplementedError()

    args.full_scale_dir = expandvars(args.full_scale_dir)
    args.annotation_dir = expandvars(args.annotation_dir)
    args.output_dir = expandvars(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Full scale video dir: {args.full_scale_dir}")
    print(f"Annotation dir: {args.annotation_dir}")
    print(f"Output dir: {args.output_dir}")

    frame_ids = set()
    frame_idxs_by_video = defaultdict(list)
    json_paths = [join(args.annotation_dir, "fho_sta_train.json"), join(args.annotation_dir, "fho_sta_val.json")]

    print("Determining frame IDs...")
    for json_path in tqdm(json_paths):
        with open(json_path, "r") as f:
            data = json.load(f)
            new_annot_ids = [annot["uid"] for annot in data["annotations"]]
            frame_ids.update(new_annot_ids)
            for annot_id in new_annot_ids:
                video_id, frame_idx = annot_id.rsplit("_", 1)
                frame_idx = int(frame_idx)
                frame_idxs_by_video[video_id].append(frame_idx)
    
    print(f"Found {len(frame_ids)} frame IDs and {len(frame_idxs_by_video)} videos in total")
    print()
    print("Extracting frames...")
    progress = tqdm(frame_idxs_by_video.items())
    for video_id, frame_idxs in progress:
        progress.set_description(video_id)
        video_path = join(args.full_scale_dir, f"{video_id}.mp4")

        reader = Ego4dDataReaderMp4(actor_path=None, video=Path(video_path), overwrite_path=True)
        for frame_idx in frame_idxs:
            output_path = join(args.output_dir, f"{video_id}_{frame_idx:07d}.jpg")
            if not args.overwrite and isfile(output_path):
                continue
            frame_np = reader.get_frame(frame_idx)
            img = Image.fromarray(frame_np)
            img.save(output_path, quality=95)
    print("Done")
