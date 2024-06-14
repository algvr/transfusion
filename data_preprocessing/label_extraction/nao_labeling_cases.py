import logging

import numpy as np
import pandas as pd
import torch

from detectron2.structures.boxes import Boxes, pairwise_intersection, pairwise_point_box_distance

HAND_LABEL = "person"


def match_frame_in_annotation(
    extracted_matches: pd.Series, curr_annotation: pd.DataFrame, soft_matches: dict, version: int
):
    """Take extracted labels for a given frame, preceding the curr_annotation and check if they match or not
    the annotation text, using soft_matches as well.

    Returns:
        List with idxs of the extracted labels that match the current annotation.
    """
    verb = curr_annotation["verb"].lower()
    actions_soft_matches = soft_matches["actions"]
    if extracted_matches.name in {
        "P22-R03-BaconAndEggs-243947-247593",
        "P29_03_237",
        "P30_107_240",
        "P16_01_63",
        "P16_01_11",
        17781,
    }:
        print("debu")
        pass
    # first try with verb matches for actions that use multiple objects or are special cases e.g. pour water
    if (
        verb in actions_soft_matches
        or verb in actions_soft_matches["obj_only"]
        or verb in actions_soft_matches["take_closest_to_hand"]
    ):
        matching_idxs = list(set(_verb_parse_case(verb, extracted_matches, curr_annotation, soft_matches)))
        return matching_idxs

    # e.g. if we have a vegetable type noun we might want to pick all its instances
    soft_matches_processor = _get_first_match
    matching_idxs = []
    for noun in curr_annotation["all_nouns"]:
        if noun in soft_matches["no_matches"]:
            return []

        noun_soft_matches = soft_matches.get(noun, {})
        default_label_matching_idx = soft_matches_processor(noun_soft_matches.get("default", []), extracted_matches)
        matching_idxs += default_label_matching_idx

        if verb in noun_soft_matches:
            # try with side matches e.g. for put pepper -> look for cutting board since that is likely "put" target
            side_label_matching_idx = soft_matches_processor(noun_soft_matches.get(verb, []), extracted_matches)
            matching_idxs += side_label_matching_idx

    if not matching_idxs:
        return []

    return list(set(matching_idxs))


def _verb_parse_case(
    verb,
    extracted_matches: pd.Series,
    curr_annotation: pd.DataFrame,
    soft_matches: dict,
):
    actions_soft_matches = soft_matches["actions"]
    verb_soft_matches = actions_soft_matches.get(verb, [])
    ann_nouns = curr_annotation["all_nouns"]

    match_processer = _get_soft_match_processer(verb, actions_soft_matches)

    matching_idxs = []
    if verb in actions_soft_matches["obj_only"] or "obj" in verb_soft_matches:
        verb_ext_matches = verb_soft_matches[1:] if "obj" in verb_soft_matches else []

        for noun in ann_nouns:
            # First try to get all possible noun matches
            noun_soft_matches = soft_matches.get(noun, dict())
            matching_idxs += match_processer(noun_soft_matches.get("default", []), extracted_matches)

            # verb-noun special cases have priority over verb special cases
            verb_ext_matches = noun_soft_matches.get(verb, []) + verb_ext_matches

        matching_idxs += match_processer(verb_ext_matches, extracted_matches)
        if verb == "cut":
            matching_idxs += match_processer(
                soft_matches.get("knife", {"default": ["knife"]})["default"], extracted_matches
            )

        return list(set(matching_idxs))

    else:
        return match_processer(verb_soft_matches, extracted_matches)


def _get_soft_match_processer(verb, soft_matches_actions):
    if verb == "put":
        return _get_put_obj_matches
    if verb in soft_matches_actions["take_closest_to_hand"]:
        return _get_closest_to_hands
    elif verb in soft_matches_actions["take_all_instances"]:
        return _get_all_matches
    else:
        return _get_first_match


def _get_closest_to_hands(desired_label_soft_matches, extracted_matches):
    if not desired_label_soft_matches:
        return []

    hand_positions = np.where(extracted_matches["Classes"] == HAND_LABEL)[0]
    if len(hand_positions) == 0:
        return []

    all_matching_idxs = _get_all_matches(desired_label_soft_matches, extracted_matches)
    if len(all_matching_idxs) == 0:
        return []

    hand_boxes, obj_boxes, pairwise_inters = intersect_obj_hands(
        extracted_matches["Bboxes"][hand_positions], extracted_matches["Bboxes"][all_matching_idxs]
    )
    if pairwise_inters.max() == 0:
        obj_centers = obj_boxes.get_centers()
        pairwise_dists = pairwise_point_box_distance(obj_centers, hand_boxes).numpy()
        # we want to pick the object which has the minimum maximum distance to the hands
        biggest_dists = np.max(pairwise_dists, axis=2)
        best_match = np.unravel_index(np.argmin(biggest_dists), biggest_dists.shape)[0]
    else:
        # pick the object with the largest overlap
        best_match = np.unravel_index(np.argmax(pairwise_inters), pairwise_inters.shape)[0]

    return [all_matching_idxs[best_match]]


def intersect_obj_hands(hand_boxes, obj_boxes):
    # the rows in the pairwise inters return represent the object entries, columns the hands
    hand_boxes = Boxes(hand_boxes)
    obj_boxes = Boxes(obj_boxes)
    pairwise_inters = pairwise_intersection(obj_boxes, hand_boxes).numpy()
    return hand_boxes, obj_boxes, pairwise_inters


def _get_put_obj_matches(desired_label_soft_matches, extracted_matches):
    if not desired_label_soft_matches:
        return []

    hand_positions = np.where(extracted_matches["Classes"] == HAND_LABEL)[0]
    if len(hand_positions) == 0:
        return []

    all_matching_idxs = _get_all_matches(desired_label_soft_matches, extracted_matches)
    if len(all_matching_idxs) == 0:
        return []

    _, _, pairwise_inters = intersect_obj_hands(
        extracted_matches["Bboxes"][hand_positions], extracted_matches["Bboxes"][all_matching_idxs]
    )

    if pairwise_inters.max() == 0:
        return []
    else:
        idx = np.unravel_index(np.argmax(pairwise_inters), pairwise_inters.shape)[0]
        return [all_matching_idxs[idx]]


def _get_all_matches(desired_label_soft_matches, extracted_matches):
    if not desired_label_soft_matches:
        return []

    idxs = np.array([])
    for desired_label in desired_label_soft_matches[:3]:
        idxs_in_detections = np.where(desired_label == extracted_matches["Classes"])[0]
        idxs = np.append(idxs, idxs_in_detections)

    if len(idxs) == 0:
        for desired_label in desired_label_soft_matches[3:]:
            idxs_in_detections = np.where(desired_label == extracted_matches["Classes"])[0]
            idxs = np.append(idxs, idxs_in_detections)

    return idxs.astype(int).tolist()


def _get_first_match(desired_label_soft_matches, extracted_matches):
    # getting the desired label and it's soft matches based on annotation
    # they are ordered on priority, i.e. the first one that matches an extracted labels is taken
    if not desired_label_soft_matches:
        return []

    for desired_label in desired_label_soft_matches:
        try:
            idx_in_detections = np.where(desired_label == extracted_matches["Classes"])[0][0]
            return [idx_in_detections]
        except IndexError:
            # means the desired label is not in extracted labels
            continue

    return []


def _get_soft_matches_list(soft_matches, label, key="default"):
    label_soft_matches = soft_matches.get(label, None)
    if not label_soft_matches:
        return []
    else:
        return label_soft_matches[key]
