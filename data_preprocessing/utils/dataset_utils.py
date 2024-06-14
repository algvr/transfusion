import cv2
import logging
import json
import numpy as np
import os
import pandas as pd
import torch
from tqdm import tqdm

from data_preprocessing.datasets.readers import Ego4dJpgReader, get_image_reader
from .math_utils import get_lin_space
from .path_utils import get_path_to_actor, get_videos_for_actor

MAX_STD = 5


def _filter_nao_annotations(df, offset, take_double=False):
    df["det_diff"] = df["nao_start_sec"] - df["det_sec"]
    df = df[df["det_diff"] > offset]
    if not take_double:
        min_diff = df.groupby("nao_clip_id")["det_diff"].min()
        df = df.reset_index().merge(min_diff, on="nao_clip_id", suffixes=("", "_min")).set_index("Frame_no")
        df = df[df["det_diff"] == df["det_diff_min"]].drop("det_diff_min", axis=1)
    else:
        gap = 0.35
        min_diff = df.groupby("nao_clip_id")["det_diff"].min()
        df1 = df.reset_index().merge(min_diff, on="nao_clip_id", suffixes=("", "_min")).set_index("Frame_no")
        df1 = df1[df1["det_diff"] == df1["det_diff_min"]].drop("det_diff_min", axis=1)

        # setting new detection difference based on the other, make sure we don't double pick
        # e.g. take one frame with eta 0.8 and another with 0.9, would be too close
        df = (
            df.reset_index()
            .merge(df1[["nao_clip_id", "det_diff"]], on="nao_clip_id", suffixes=("", "_prev"))
            .set_index("Frame_no")
        )

        df = df[df["det_diff"] > df["det_diff_prev"] + gap]
        df = df.drop(columns=["det_diff_prev"])
        min_diff = df.groupby("nao_clip_id")["det_diff"].min()
        df2 = df.reset_index().merge(min_diff, on="nao_clip_id", suffixes=("", "_min")).set_index("Frame_no")
        df2 = df2[df2["det_diff"] == df2["det_diff_min"]].drop("det_diff_min", axis=1)

        df = pd.concat([df1, df2])

    return df


def _get_readers(actors, source_dataset, root_data_path, max_std=MAX_STD):
    readers = {}
    reader_cls = get_image_reader(source_dataset)
    logging.info(f"Loading readers {source_dataset}")
    mapped_spaces = {}
    for actor in tqdm(actors):
        actor_path = get_path_to_actor(root_data_path, source_dataset, actor)
        if source_dataset not in {"ego4djpg", "ego4djpgv2"}:
            actor_videos = get_videos_for_actor(actor_path, source_dataset)
            for video_path in actor_videos:
                reader = reader_cls(actor_path, video=video_path.name)
                h, w, c = reader.get_img_shape()
                mapped_x, mapped_y = mapped_spaces.get(
                    f"{h}_{w}", get_lin_space(w, h, max_std_h=max_std * h / w, max_std_w=max_std)
                )
                if f"{h}_{w}" not in mapped_spaces:
                    mapped_spaces[f"{h}_{w}"] = mapped_x, mapped_y
                if source_dataset != "ego4d":
                    readers[video_path.stem] = {"reader": reader, "mapped_x": mapped_x, "mapped_y": mapped_y}
                else:
                    readers[actor_path.name] = {"reader": reader, "mapped_x": mapped_x, "mapped_y": mapped_y}
        else:
            reader = Ego4dJpgReader(data_path=actor_path, actor_id=actor)
            readers[actor] = {"reader": reader}
    return readers


def collate_unidet_input(batch):
    return [{"image": x["image"], "height": x["height"], "width": x["width"]} for x in batch], {
        "frames": torch.stack([x["frames"] for x in batch])
    }


def get_label_mapping(label_col, word_type, dataset_list, use_external_label_mapping):
    if not use_external_label_mapping:
        uniques = sorted(label_col.unique())
        # need to keep 0 as a bg class, the domain classes are counted from 1
        return {k: idx + 1 for idx, k in enumerate(uniques)}
    
    if dataset_list == "ego4djpg":
        mapping_file_path = os.path.expandvars("${CODE}/data_preprocessing/configs/label_mappings_v1.json")
    elif dataset_list == "ego4djpgv2":
        mapping_file_path = os.path.expandvars("${CODE}/data_preprocessing/configs/label_mappings_v2.json")

    if os.path.isfile(mapping_file_path):
        with open(mapping_file_path, "r") as f:
            mapping_dict = json.loads(f.read())
    else:
        mapping_dict = {}
    
    if not isinstance(dataset_list, list):
        dataset_list = [dataset_list]

    uniques_to_match = sorted(label_col.unique())

    ret_mappings = {}
    for dataset_name in dataset_list:
        dataset_name = {"ego4djpg": "ego4d", "ego4djpgv2":"ego4d"}.get(dataset_name, dataset_name)
        if dataset_name not in mapping_dict or word_type not in mapping_dict[dataset_name]:
            continue
        
        for word, mapping in mapping_dict[dataset_name][word_type].items():
            if word not in ret_mappings:
                ret_mappings[word] = mapping
            else:
                print(f'Duplicate mapping for {word_type} "{word}": indices {mapping} and {ret_mappings[word]}; '
                      f'keeping {ret_mappings[word]}')

    highest_mapping = max(ret_mappings.values()) if len(ret_mappings) > 0 else 0

    # remap 0 for nouns
    if word_type == "noun":
        zero_mapping_words = [k for k, v in ret_mappings.items() if v == 0]
        if len(zero_mapping_words) > 0:
            highest_mapping += 1
            quot = '"'
            print(f'Remapping {word_type}s "{(quot+", "+quot).join(zero_mapping_words)}" from index 0 to {highest_mapping}'
                  f' (0 reserved for background)')
            for zero_mapping_word in zero_mapping_words:
                ret_mappings[zero_mapping_word] = highest_mapping

    # add any unmatched words from uniques_to_match
    uniques_to_match = [u for u in uniques_to_match if u not in ret_mappings]
    lu = len(uniques_to_match)
    if lu > 0:
        for u in uniques_to_match:
            highest_mapping += 1
            ret_mappings[u] = highest_mapping

        plur = "s" if lu != 1 else ""
        print(f"Generated additional matching{plur} for {lu} {word_type}{plur}")

    print(f"Label mappings for {word_type=}: {ret_mappings=}")
    
    return ret_mappings


def get_resize_fn(resize_spec):
    if resize_spec == None:
        return lambda x: x
    else:
        return lambda x: cv2.resize(x, dsize=resize_spec[::-1], interpolation=cv2.INTER_LANCZOS4)


def merge_labels(nao_annots, label_merging):
    """Replaces labels for each column-key entry with the one specified in dictionary.
    Used to decrease the size of classif space"""
    if label_merging:
        for category, category_syn in label_merging.items():
            nao_annots[category] = nao_annots[category].apply(lambda x: category_syn.get(x, x))

    return nao_annots


def drop_labels(nao_annots, labels_cutoff):
    """Drop entries whose labels have fewer appearances than label_cutoff. Both for verb and noun."""
    keys = ["noun", "verb"]
    if labels_cutoff:
        if labels_cutoff["drop"] is True:
            for key in keys:
                counts = nao_annots[key].value_counts()
                keep_entries = counts[counts >= labels_cutoff[key]].index
                nao_annots = nao_annots[nao_annots[key].isin(keep_entries)]
        elif labels_cutoff["drop"].lower() == "top":
            for key in keys:
                if labels_cutoff[key]:
                    counts = nao_annots[key].value_counts()
                    nao_annots = nao_annots[nao_annots[key].isin(counts[: labels_cutoff[key]].index)]

    return nao_annots


def get_resize_specs(resize_spec, epic_ds):
    if resize_spec == "epic":
        return {"epic": get_resize_fn(None), "egtea": get_resize_fn(epic_ds.get_img_shape()[:2])}
    elif isinstance(resize_spec, list):
        return {"epic": get_resize_fn(resize_spec), "egtea": get_resize_fn(resize_spec)}
    elif resize_spec == None:
        return {"epic": get_resize_fn(None), "egtea": get_resize_fn(None)}
