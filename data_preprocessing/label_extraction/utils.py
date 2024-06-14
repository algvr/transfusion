import numpy as np
import torch

from detectron2.structures.boxes import Boxes, pairwise_iou


def _create_text_labels(classes, scores, class_names, is_crowd=None):
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


def labels_to_classes(preds, label_file, w_scores=True):
    classes_map = ["{}".format([xx for xx in x["name"].split("_") if xx != ""][0]) for x in label_file["categories"]]

    classes = [
        _create_text_labels(
            pred["instances"].pred_classes.cpu().numpy(),
            pred["instances"].scores if w_scores else None,
            classes_map,
        )
        for pred in preds
    ]
    return classes


def remove_extra_classes(pred, det_classes, extra_classes):
    """Remove extra classes such as `person`, `bracelet` which are not useful for our kitchen task from a pred
    instance"""
    instances = pred["instances"]
    idx_to_keep = []
    classes_to_keep = []
    for idx, det_class in enumerate(det_classes):
        if det_class.lower() not in extra_classes:
            idx_to_keep.append(idx)
            classes_to_keep.append(det_class)

    pred["instances"] = instances[idx_to_keep]
    return pred, classes_to_keep


def detections_to_pd_row(preds, label_file, metadata, extra_classes={"bracelet", "watch"}):
    """Each pred corresponds to instances detection a la detectron2 for a frame"""
    rows, row = [], []

    det_classes = labels_to_classes(preds, label_file, w_scores=False)
    frames = metadata["frames"].cpu().numpy()
    for i, example in enumerate(preds):
        kitchen_preds, kitchen_classes = remove_extra_classes(example, det_classes[i], extra_classes)
        if len(kitchen_classes) == 0:
            continue
        row.append(frames[i])
        row.append(kitchen_classes)
        row.append(np.round(kitchen_preds["instances"].scores.cpu().numpy(), 3).tolist())
        row.append(np.round(kitchen_preds["instances"].pred_boxes.tensor.cpu().numpy(), 3).tolist())
        rows.append(row)
        row = []

    return rows


def nms(annot, iou_lim):
    boxes = Boxes(annot["Bboxes"])
    scores = annot["Scores"]
    classes = annot["Classes"]

    boxes_cpy = boxes.clone()
    scores_cpy = scores.copy()
    classes_cpy = classes.copy()

    to_return_b = []
    to_return_s = []
    to_return_c = []

    while len(boxes_cpy) > 0:
        cur = boxes_cpy[0]

        to_return_b.append(boxes_cpy[0])
        to_return_s.append(scores_cpy[0])
        to_return_c.append(classes_cpy[0])

        boxes_cpy = boxes_cpy[1:]
        scores_cpy = scores_cpy[1:]
        classes_cpy = classes_cpy[1:]

        if len(boxes_cpy) == 0:
            break

        ious = pairwise_iou(cur, boxes_cpy)

        to_drop = (ious > iou_lim).squeeze()
        to_keep = torch.where(~to_drop)[0]

        boxes_cpy = boxes_cpy[to_keep]
        scores_cpy = scores_cpy[to_keep.numpy()]
        classes_cpy = classes_cpy[to_keep.numpy()]

    return np.array(to_return_c), np.array(to_return_s), Boxes.cat(to_return_b).tensor.numpy()
