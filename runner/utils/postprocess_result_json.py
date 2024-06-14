# %%

import argparse
import json
import os
from os.path import expandvars, join
import numpy as np


def get_iou(bb1, bb2):
    assert bb1["x1"] < bb1["x2"]
    assert bb1["y1"] < bb1["y2"]
    assert bb2["x1"] < bb2["x2"]
    assert bb2["y1"] < bb2["y2"]

    # determine the coordinates of the intersection rectangle
    x_left =   max(bb1["x1"], bb2["x1"])
    y_top =    max(bb1["y1"], bb2["y1"])
    x_right =  min(bb1["x2"], bb2["x2"])
    y_bottom = min(bb1["y2"], bb2["y2"])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
    bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, required=True)
    parser.add_argument("--annotation-dir", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        input_dict = json.load(f)

    if args.annotation_dir in {None, ""}:
        # attempt to read version from result file
        if len(input_dict["results"]) < 20000:  # v1
            args.annotation_dir = expandvars("${DATA}/Ego4d/v1/annotations")
        else:
            args.annotation_dir = expandvars("${DATA}/Ego4d/v2/annotations")

    path_train =   join(args.annotation_dir, "fho_sta_train.json")
    path_val   =   join(args.annotation_dir, "fho_sta_val.json")
    path_test  =   join(args.annotation_dir, "fho_sta_test_unannotated.json")

    if args.output_path in {None, ""}:
        args.output_path = args.json_path.rsplit(".", 1)[0] + "_corrected.json"

    print(f"JSON path is {args.json_path}")
    print(f"Annotation dir is {args.annotation_dir}")
    print(f"Output path is {args.output_path}")

    noun_verb_freqs = {}  # keys are nouns
    with open(path_train, "r") as f:
        ld_train = json.load(f)
    with open(path_val, "r") as f:
        ld_val = json.load(f)
    with open(path_test, "r") as f:
        ld_test = json.load(f)

    video_widths = {}
    video_heights = {}

    for video_id, video_meta in [*ld_train["info"]["video_metadata"].items(), *ld_val["info"]["video_metadata"].items(), *ld_test["info"]["video_metadata"].items()]:
        video_widths[video_id] = video_meta["frame_width"]
        video_heights[video_id] = video_meta["frame_height"]

    box_relative_areas = {}

    for annot in ld_train["annotations"]:
        objs = annot["objects"]
        video_id = annot["uid"].rsplit("_", 1)[0]
        for box in objs:
            noun = box["noun_category_id"]
            verb = box["verb_category_id"]
            if noun not in noun_verb_freqs:
                noun_verb_freqs[noun] = {}
            noun_verb_freqs[noun][verb] = noun_verb_freqs[noun].get(verb, 0) + 1
            if noun not in box_relative_areas:
                box_relative_areas[noun] = []

            relative_area = ((box["box"][2] - box["box"][0]) * (box["box"][3] - box["box"][1])) / (video_widths[video_id] * video_heights[video_id])
            box_relative_areas[noun].append(relative_area)
    
    for noun in box_relative_areas.keys():
        box_relative_areas[noun] = np.array(box_relative_areas[noun])

    output_dict = {**{k: input_dict[k] for k in ["version", "challenge"]}, "results": {}}
    num_unseen = 0

    q_mins = {}
    q_maxs = {}

    MAX_PREDS_PER_NOUN_BOX = 3
    num_rel_rejected_boxes = 0

    for frame_id, current_result in input_dict["results"].items():
        # sum up scores and create distribution
        video_id = frame_id.rsplit("_")[0]
        score_sum = sum([box["score"] for box in current_result])
        boxes = [(idx, box["score"] / score_sum, box["noun_category_id"], box["verb_category_id"], box["box"]) for idx, box in enumerate(current_result)]
        boxes = sorted(boxes, key=lambda ent: -ent[1])

        output_boxes = []
        frame_found_unseen = False

        frame_seen_verbs = set()
        frame_noun_verbs = {}
        
        for box_position, box_data in enumerate(boxes):
            box_idx, box_score, box_noun, box_verb, box_coords = box_data
            num_train_occ = noun_verb_freqs.get(box_noun, {}).get(box_verb, 0)
            if num_train_occ > 0:
                if box_noun not in frame_noun_verbs:
                    frame_noun_verbs[box_noun] = {}
                frame_noun_verbs[box_noun][box_verb] = frame_noun_verbs[box_noun].get(box_verb, 0) + 1

        frame_noun_verbs_sorted = {k: sorted([(vrb, l) for vrb, l in v.items()], key=lambda ent: -ent[1])
                                   for k, v in frame_noun_verbs.items()}

        frame_noun_boxes = {}
        frame_noun_verb_boxes = {}

        for box_position, box_data in enumerate(boxes):
            box_idx, box_score, box_noun, box_verb, box_coords = box_data
            num_train_occ = noun_verb_freqs.get(box_noun, {}).get(box_verb, 0)
            if num_train_occ == 0 and box_position <= 4 and not frame_found_unseen:
                frame_found_unseen = True
                num_unseen += 1
            if num_train_occ == 0:
                # replace by most frequent verb w.r.t this noun
                noun_verbs = sorted(noun_verb_freqs.get(box_noun, {}).items(), key=lambda ent: -ent[1])

                if len(noun_verbs) > 0:
                    for top_verb_ in noun_verbs:
                        top_verb = top_verb_[0]

                        if box_noun in frame_noun_verbs_sorted and len(frame_noun_verbs_sorted[box_noun]) > 0 and  box_position <= 4:
                            top_verb = frame_noun_verbs_sorted[box_noun][0][0]
                            # print(f"Found frame-wise top verb with #{frame_noun_verbs_sorted[box_noun][0][1]}")
                        
                        #if box_position <= 4:
                        #    print(f'Replaced {current_result[box_idx]["verb_category_id"]=} by {top_verb=} for {frame_id=}')
                        current_result[box_idx]["verb_category_id"] = top_verb
                        break

            current_result[box_idx]["time_to_contact"] = max(0.251, current_result[box_idx]["time_to_contact"])
            cbn, cbv = current_result[box_idx]["noun_category_id"], current_result[box_idx]["verb_category_id"]
            cb = frame_noun_verb_boxes.get((cbn, cbv))
            
            if not (box_coords[0] < box_coords[2] and box_coords[1] < box_coords[3]):
                continue 

            if cb is not None:
                skip = False
                for other_box in cb:
                    if not (other_box[0] < other_box[2] and other_box[1] < other_box[3]):
                        continue 

                    iou = get_iou({"x1": box_coords[0], "y1": box_coords[1], "x2": box_coords[2], "y2": box_coords[3]},
                                  {"x1": other_box[0], "y1": other_box[1], "x2": other_box[2], "y2": other_box[3]})
                    if iou > 0.0:
                        skip = True
                        break
                if skip:
                    continue
            else:
                frame_noun_verb_boxes[(cbn, cbv)] = []

            output_boxes.append(current_result[box_idx])
            frame_seen_verbs.add(current_result[box_idx]["verb_category_id"])
            if (cbn, cbv) not in frame_noun_verb_boxes:
                frame_noun_verb_boxes[(cbn, cbv)] = []

            frame_noun_verb_boxes[(cbn, cbv)].append(box_coords)

            if cbn not in frame_noun_boxes:
                frame_noun_boxes[cbn] = []
            frame_noun_boxes[cbn].append(box_coords)

        output_dict["results"][frame_id] = output_boxes
        
    with open(args.output_path, "w") as f:
        json.dump(output_dict, f)
    print(f"Output written to {args.output_path}")
