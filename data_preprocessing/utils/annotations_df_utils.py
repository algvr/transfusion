import ast
import itertools
import json
import numpy as np
import pandas as pd
# import swifter


from data_preprocessing.utils.math_utils import custom_round, get_boxes_area
from data_preprocessing.utils.path_constants import *


MAX_REC_OFFSET = 5
MAX_REC_EXTRACT_OFFSET = 20


ignore_labels = {
    "ego4d": {"noun": [], "verb": []},
    "ego4djpg": {"noun": [], "verb": []},
    "ego4djpgv2": {"noun": [], "verb": []},
    "epic": {
        "noun": [
            "clothes",
            "chair",
            "foil",
            "paper",
            "scissors",
            "cap",
            "bar",
            "floor",
            "path",
            "tablet",
            "ladder",
            "cover",
            "clip",
            "washing powder",
            "dust pan",
            "crisp",
            "wire",
            "plug",
            "phone",
            "ice",
            "sushi mat",
            "wall",
            "sous vide machine",
            "remote control",
            "watch",
            "egg scotch",
            "light",
            "hoover",
            "candle",
        ],
        "verb": [],
    },
    "egtea": {"noun": [], "verb": []},
}
ignore_labels["epicv"] = ignore_labels["epic"]


def timestamp_to_seconds(timestamp):
    if timestamp:
        return timestamp_to_ms(timestamp) / 1000
    return None


def timestamp_to_ms(timestamp):
    if timestamp:
        parts = [float(part) for part in timestamp.split(":")]
        ms = (parts[0] * 3600 + parts[1] * 60 + parts[2]) * 1000 + 1e-6
        return ms
    return None


def get_ego4d_annotations_df(data_root, resize_boxes=True, action_rec=False):
    train_annots = get_ego4d_annotations_subset(data_root, "train", resize_boxes, action_rec)
    val_annots = get_ego4d_annotations_subset(data_root, "val", resize_boxes, action_rec)

    test_annots_path = data_root / "annotations" / "fho_sta_test_unannotated.json"
    with open(test_annots_path, "r") as fp:
        test_annots_json = json.loads(fp.read())

    test_annots = pd.DataFrame.from_dict(test_annots_json["annotations"])
    test_annots = test_annots.rename({"frame": "Frame_no"}, axis=1).drop(["clip_frame", "clip_uid"], axis=1)
    test_annots["orig_split"] = "test"
    test_annots["nao_clip_id"] = test_annots["uid"]
    test_annots["episode_id"] = test_annots["uid"]
    test_annots["episode_action_id"] = test_annots["uid"]
    try:
        test_annots["fps"] = test_annots["video_uid"].apply(lambda x: test_annots_json["info"]["video_metadata"][x]["fps"])
    except:
        test_annots["fps"] = test_annots["video_id"].apply(lambda x: test_annots_json["info"]["video_metadata"][x]["fps"])
    test_annots["all_nouns"] = [["ball"]] * len(test_annots)
    test_annots["all_verbs"] = [["take"]] * len(test_annots)
    test_annots["noun"] = "ball"
    test_annots["verb"] = "take"
    test_annots["start_frame"] = test_annots["Frame_no"]
    test_annots["nao_narration"] = "test_set_nao_narration"
    test_annots["Bboxes"] = [np.array([[17, 3, 190, 960]]) for _ in range(len(test_annots))]
    test_annots["narration"] = "test_set_narration"
    test_annots["det_sec"] = test_annots["Frame_no"] / test_annots["fps"]
    test_annots = test_annots.set_index("uid")

    annots = pd.concat([train_annots, val_annots, test_annots], axis=0)
    annots = annots[
        ~annots.index.isin(
            [
                "77ed1624-f87b-4196-9a0a-95b7023b18e4_0000220",
                "d18ef16d-f803-4387-bb5e-7876f1522a63_0023565",
                "77ed1624-f87b-4196-9a0a-95b7023b18e4_0000205",
                "77ed1624-f87b-4196-9a0a-95b7023b18e4_0000190",
                "d18ef16d-f803-4387-bb5e-7876f1522a63_0023520",
            ]
        )
    ]

    if action_rec:
        lines = []
        with open(os.path.expandvars("$CODE/data_preprocessing/flow_extraction/missing_rgb.txt"), "r") as fp:
            lines = fp.read().splitlines()

        missing_ids = set(lines)
        annots = annots[~annots.index.isin(missing_ids)]

    return annots


def get_ego4d_annotations_subset(data_root, subset, resize_boxes, action_rec=False):
    v2 = "v2" in str(data_root)
    if v2:
        uid_col = "video_uid"
    else:
        uid_col = "video_id"
    json_path = data_root / "annotations" / f"fho_sta_{subset}.json"
    with open(json_path, "r") as fp:
        annot_json = json.load(fp)
    annot_df = pd.DataFrame.from_dict(annot_json["annotations"])

    for entry in annot_json["noun_categories"]:
        if entry["name"] == "indument":
            entry["name"] = "cloth"
            break

    def objects_to_entries(objects):
        """Structure of Ego4D entry:  {"uid": "ae7b6096-4f00-42af-857d-603c2cbfa940_0073648",
            "video_id": "ae7b6096-4f00-42af-857d-603c2cbfa940",
            "frame": 73648,
            "clip_id": 1654,
            "clip_uid": "381e7ae9-2eae-4534-8df8-2e7793e8c5e9",
            "clip_frame": 1889,
            "objects": [
                {
                    "box": [637.02,250.16,822.23,464.24],
                    "verb_category_id": 62,
                    "noun_category_id": 16,
                    "time_to_contact": 0.16666666666666666
                }
            ]
        }"""
        boxes = []
        verb_ids = []
        noun_ids = []
        ttc = 0

        for obj in objects:
            boxes.append(obj["box"])
            verb_ids.append(obj["verb_category_id"])
            noun_ids.append(obj["noun_category_id"])
            ttc = obj["time_to_contact"]

        return np.array(boxes), verb_ids, noun_ids, ttc

    orig_w_h = pd.DataFrame.from_dict(annot_json["info"]["video_metadata"], orient="index")
    # shape of lmdb extracted frames
    orig_w_h["new_h"] = 480.0
    orig_w_h["scale"] = orig_w_h["frame_height"] / orig_w_h["new_h"]

    assert np.all(annot_df[uid_col].isin(orig_w_h.index.tolist()))
    annot_df = annot_df.join(
        annot_df.apply(lambda x: objects_to_entries(x["objects"]), result_type="expand", axis=1)
    ).drop("objects", axis=1)

    annot_df = annot_df.rename({"frame": "Frame_no", 0: "Bboxes", 1: "verb_ids", 2: "noun_ids", 3: "det_diff"}, axis=1)
    annot_df = annot_df.merge(orig_w_h[["scale"]], left_on=uid_col, right_index=True)
    # we are resizing boxes if we use preextracted frames at lower resolution
    # if we use original jpgs we don't rescale
    if resize_boxes:
        annot_df["Bboxes"] = annot_df["Bboxes"] / annot_df["scale"]
    annot_df = annot_df[annot_df["Bboxes"].map(lambda x: get_boxes_area(x) > 1)]

    def category_to_val_noun(cat_ids, mapper):
        if not v2:
            return [mapper[cat_id]["name"].split("_")[0] for cat_id in cat_ids]
        else:
            to_return = []
            for cat_id in cat_ids:
                if cat_id == 46:
                    to_return.append("nut tool")
                elif cat_id == 101:
                    to_return.append("nut food")
                elif cat_id == 76:
                    to_return.append("measurement tape")
                elif cat_id == 121:
                    to_return.append("tape")
                else:
                    to_return.append(mapper[cat_id]["name"].split("_")[0])
            return to_return


    # turn: 67, turn_off:68, turn_on:69
    def category_to_val_verb(cat_ids, mapper):
        # this one is the same for v1 and v2
        to_return = []
        for cat_id in cat_ids:
            if cat_id == 68:
                to_return.append("turn-off")
            elif cat_id == 69:
                to_return.append("turn-on")
            else:
                to_return.append(mapper[cat_id]["name"].split("_")[0])
        return to_return
        

    annot_df["all_nouns"] = annot_df["noun_ids"].apply(lambda x: category_to_val_noun(x, annot_json["noun_categories"]))
    annot_df["all_verbs"] = annot_df["verb_ids"].apply(lambda x: category_to_val_verb(x, annot_json["verb_categories"]))
    annot_df["noun"] = annot_df["all_nouns"].apply(lambda x: x[0])
    annot_df["verb"] = annot_df["all_verbs"].apply(lambda x: x[0])
    annot_df["fps"] = annot_df[uid_col].apply(lambda x: annot_json["info"]["video_metadata"][x]["fps"])
    annot_df["nao_clip_id"] = annot_df["uid"]
    annot_df["episode_id"] = annot_df["uid"]

    # if we do action recognition, we keep the labels but we offset the frames with the required fps
    if action_rec:
        offset_frames = (annot_df["fps"] * annot_df["det_diff"]).astype(int)
        random_offset = np.random.randint(0, MAX_REC_OFFSET - 1, len(offset_frames))
        offset_frames += random_offset
        # offset_frames += MAX_REC_OFFSET
        annot_df["nao_frame"] = annot_df["Frame_no"].copy()
        annot_df["Frame_no"] += offset_frames

    annot_df["start_frame"] = annot_df["Frame_no"]
    annot_df = annot_df.set_index("uid")

    cur_id = 0
    prev_row = annot_df.iloc[0]
    ep_ids = [f"{prev_row['clip_id']}_{cur_id:04d}"]
    for idx, row in annot_df[1:].iterrows():
        if row["clip_id"] == prev_row["clip_id"]:
            if row["det_diff"] > prev_row["det_diff"]:
                cur_id += 1
        else:
            cur_id = 0
        ep_ids.append(f"{row['clip_id']}_{cur_id:04d}")
        prev_row = row
    annot_df["episode_action_id"] = ep_ids

    def verb_noun_to_narration(row):
        all_nouns = row["all_nouns"]
        all_verbs = row["all_verbs"]

        if len(all_nouns) > len(all_verbs):
            pairs = [" ".join(x) for x in itertools.zip_longest(all_verbs, all_nouns, fillvalue=all_verbs[0])]
        else:
            pairs = [" ".join(x) for x in itertools.zip_longest(all_verbs, all_nouns, fillvalue=all_nouns[0])]

        return " and ".join(pairs)

    annot_df["nao_narration"] = annot_df.apply(verb_noun_to_narration, axis=1)
    # so the narration is not actually the narration of the current frame,
    # but is needed for compatibility with the other dataframes
    annot_df["narration"] = annot_df["nao_narration"]
    annot_df["det_sec"] = annot_df["Frame_no"] / annot_df["fps"]
    annot_df["orig_split"] = subset

    return annot_df
