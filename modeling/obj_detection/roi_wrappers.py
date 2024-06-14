from torchvision.models.detection import *
from torchvision.models.detection.backbone_utils import *
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform, resize_boxes
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from typing import Dict, List, Tuple, Optional

import copy
import numpy as np
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from modeling.ttc_pred import TTCPredictionHead
from runner.nao.postprocessing import get_iou


COND_VERB_ARGS = {"representation_size": 512, "w_detach": False, "w_bn": True}
IGNORE_VERB_IDX_BG = 999
IGNORE_VERB_IDX_BG_FLOAT = float(IGNORE_VERB_IDX_BG)


def get_roi_heads_wrapper(verbs, verb_classifier_args=None):
    if verbs:
        if not verb_classifier_args or verb_classifier_args["type"] is False:
            return DualClassRoiHeadsWrapper
        else:
            raise NotImplementedError
    else:
        return RoIHeadsWrapper


class RoIHeadsWrapper(nn.Module):
    """Wrapper arround Pytorch RoIHeads class. Loss computation will be moved in Lightining Trainer, not inside the model as there."""

    def __init__(
        self,
        roi_head_wrap,
        box_features_dropout_1,
        classif_droput,
        box_regressor,
        noun_classifier,
        verb_classifier,
        representation_size=1024,
        verb_classifier_args={},
        ttc_pred=False,
        ttc_hand_head_args=None,
        replace_heads=False,
        additional_postprocessing=False,
        additional_postprocessing_frequencies=None
    ) -> None:
        super().__init__()
        self.roi_head_wrap = copy.deepcopy(roi_head_wrap)
        self.dropout_1 = nn.Dropout2d(box_features_dropout_1) if box_features_dropout_1 else nn.Identity()
        self.classif_dropout = nn.Dropout(classif_droput) if classif_droput else nn.Identity()
        self.box_regressor = box_regressor
        self.noun_classifier = noun_classifier
        self.verb_classifier = verb_classifier
        self.ttc_pred = ttc_pred
        self.ttc_hand_head_args = ttc_hand_head_args
        self.verb_classifier_args = verb_classifier_args
        self.replace_heads = replace_heads
        self.additional_postprocessing = additional_postprocessing

        # create noun x verb matrix
        if additional_postprocessing_frequencies is not None:
            num_nouns = self.noun_classifier.weight.shape[0]
            num_verbs = self.verb_classifier.weight.shape[0]

            self.additional_postprocessing_frequencies = torch.zeros((num_nouns, num_verbs))
            for noun_id, verb_noun_freqs in additional_postprocessing_frequencies.items():
                for verb_id, verb_noun_freq in verb_noun_freqs.items():
                    self.additional_postprocessing_frequencies[noun_id, verb_id] = verb_noun_freq

        has_saved_checkpoint_params = hasattr(roi_head_wrap, "saved_checkpoint_params")

        if has_saved_checkpoint_params and all(
            [
                k in roi_head_wrap.saved_checkpoint_params
                for k in ["roi_heads.box_regressor.1.weight", "roi_heads.box_regressor.1.bias"]
            ]
        ) and representation_size == 1024 and self.replace_heads:
            self.box_regressor[1].weight = torch.nn.Parameter(
                roi_head_wrap.saved_checkpoint_params["roi_heads.box_regressor.1.weight"]
            )
            self.box_regressor[1].bias = torch.nn.Parameter(
                roi_head_wrap.saved_checkpoint_params["roi_heads.box_regressor.1.bias"]
            )
        else:
            nn.init.normal_(self.box_regressor[1].weight, std=0.01)
            nn.init.constant_(self.box_regressor[1].bias, 0)

        if has_saved_checkpoint_params and all(
            [
                k in roi_head_wrap.saved_checkpoint_params
                for k in ["roi_heads.noun_classifier.weight", "roi_heads.noun_classifier.bias"]
            ]
        ) and representation_size == 1024 and self.replace_heads:
            self.noun_classifier.weight = torch.nn.Parameter(
                roi_head_wrap.saved_checkpoint_params["roi_heads.noun_classifier.weight"]
            )
            self.noun_classifier.bias = torch.nn.Parameter(
                roi_head_wrap.saved_checkpoint_params["roi_heads.noun_classifier.bias"]
            )
        else:
            nn.init.normal_(self.noun_classifier.weight, std=0.01)
            nn.init.constant_(self.noun_classifier.bias, 0)

        if self.verb_classifier is not None:
            if has_saved_checkpoint_params and all(
                [
                    k in roi_head_wrap.saved_checkpoint_params
                    for k in ["roi_heads.verb_classifier.weight", "roi_heads.verb_classifier.bias"]
                ]
            ) and representation_size == 1024 and self.replace_heads:
                self.verb_classifier.weight = torch.nn.Parameter(
                    roi_head_wrap.saved_checkpoint_params["roi_heads.verb_classifier.weight"]
                )
                self.verb_classifier.bias = torch.nn.Parameter(
                    roi_head_wrap.saved_checkpoint_params["roi_heads.verb_classifier.bias"]
                )
            else:
                nn.init.normal_(self.verb_classifier.weight, std=0.01)
                nn.init.constant_(self.verb_classifier.bias, 0)

        self.representation_size = representation_size

        self.roi_head_wrap.box_predictor = None
        del self.roi_head_wrap.box_predictor

        if self.representation_size != self.roi_head_wrap.box_head.fc6.out_features:
            self.roi_head_wrap.box_head = TwoMLPHead(
                self.roi_head_wrap.box_head.fc6.in_features, self.representation_size
            )

    def has_mask(self):
        return False

    def has_keypoint(self):
        return False

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        return self.roi_head_wrap.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

    def subsample(self, labels):
        return self.roi_head_wrap.subsample(labels)

    def add_gt_proposals(self, proposals, gt_boxes):
        return self.roi_head_wrap.add_gt_proposals(proposals, gt_boxes)

    def check_targets(self, targets):
        return self.roi_head_wrap.check_targets(targets)

    def select_training_samples(self, proposals, targets):
        return self.roi_head_wrap.select_training_samples(proposals, targets)

    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        return self.roi_head_wrap.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                # if self.has_keypoint():
                #     assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            proposals, _, labels, regression_targets = self.roi_head_wrap.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None

        # here features come from FPN,  x 256 x 7 x 7
        # proposals is list of length bs, with 150 boxes per each sample -> 1200 in total
        box_features = self.roi_head_wrap.box_roi_pool(features, proposals, image_shapes)
        box_features = self.dropout_1(box_features)
        box_features = self.roi_head_wrap.box_head(box_features)
        # box_features = self.classif_dropout(box_features)
        # class_logits, box_regression = self.roi_head_wrap.box_predictor(box_features)

        if box_features.dim() == 4:
            assert list(box_features.shape[2:]) == [1, 1]
        box_features = box_features.flatten(start_dim=1)

        box_regression = self.box_regressor(box_features)

        box_features = self.classif_dropout(box_features)
        if self.noun_classifier:
            class_logits = self.noun_classifier(box_features)
        else:
            verb_logits = self.verb_classifier(box_features)
            class_logits = verb_logits

        if self.verb_classifier:
            verb_logits = self.verb_classifier(box_features)
        else:
            verb_logits = None

        if self.ttc_pred:
            if isinstance(self.ttc_pred_layer, TTCPredictionHead):
                #object_boxes = self.roi_head_wrap.box_coder.decode(box_regression, proposals)
                #ttc_prelim = self.ttc_pred_layer({"box_features": box_features,
                #                                  "object_boxes": object_boxes,
                #                                  "hand_boxes": features["hand_boxes"],
                #                                  "hand_poses": features["hand_poses"],
                #                                  "orig_shapes": [t["orig_shape"] for t in targets]})
                ttc_logits = -1.0 * torch.ones_like(box_features)[..., 0]
            else:
                ttc_prelim = self.ttc_pred_layer(box_features)
                ttc_logits = F.softplus(ttc_prelim).squeeze(-1)
        else:
            ttc_logits = None

        return {
            "class_logits": class_logits,
            "verb_logits": verb_logits,
            "ttcs": ttc_logits,
            "box_regression": box_regression,
            "labels": labels,
            "reg_targets": regression_targets,
            "proposals": proposals,
            "box_features": box_features
        }


class DualClassRoiHeadsWrapper(RoIHeadsWrapper):
    """Wrapper arround Pytorch RoIHeads class. Loss computation will be moved in Lightining Trainer,
    not inside the model as there."""

    def __init__(
        self,
        roi_head_wrap,
        box_features_dropout_1,
        classif_droput,
        box_regressor,
        noun_classifier,
        verb_classifier,
        representation_size=1024,
        verb_classifier_args={},
        ttc_pred=False,
        ttc_hand_head_args=None,
        replace_heads=False,
        additional_postprocessing=False,
        additional_postprocessing_frequencies=None
    ) -> None:
        super().__init__(
            roi_head_wrap,
            box_features_dropout_1,
            classif_droput,
            box_regressor,
            noun_classifier,
            verb_classifier,
            representation_size,
            verb_classifier_args=verb_classifier_args,
            ttc_pred=ttc_pred,
            ttc_hand_head_args=ttc_hand_head_args,
            replace_heads=replace_heads,
            additional_postprocessing=additional_postprocessing,
            additional_postprocessing_frequencies=additional_postprocessing_frequencies
        )

        self.init_v_biases = verb_classifier_args.get("init_v_biases", False)
        if self.init_v_biases:
            v_biases = torch.from_numpy(np.loadtxt(self.init_v_biases)).float()
            self.verb_classifier.bias = nn.Parameter(v_biases)

        self.roi_head_wrap.assign_targets_to_proposals = self.assign_targets_to_proposals
        self.roi_head_wrap.select_training_samples = self.select_training_samples

        if self.ttc_pred:
            if self.ttc_hand_head_args is not None and self.ttc_hand_head_args.get("use"):
                print("Using Transformer-based TTC head")
                self.ttc_pred_layer = TTCPredictionHead(
                    num_steps=self.ttc_hand_head_args["num_steps"],
                    num_heads=self.ttc_hand_head_args["num_heads"],
                    num_layers=self.ttc_hand_head_args["num_layers"],
                    feat_dim=self.ttc_hand_head_args["feat_dim"],
                    ff_dim=self.ttc_hand_head_args["ff_dim"],
                    dropout=self.ttc_hand_head_args["dropout"],
                    emb_steps_hand=self.ttc_hand_head_args["emb_steps_hand"],
                    emb_steps_object=self.ttc_hand_head_args["emb_steps_object"],
                    hand_feat_dim=63,
                    object_feat_dim=self.representation_size
                )
            else:
                print("Using linear TTC head")
                self.ttc_pred_layer = nn.Linear(self.verb_classifier.in_features, 1)

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [(t["labels"], t["verbs"], t["ttcs"]) for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # labels here contain 2 entries, the nouns and the verbs
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        # subsample based on the nouns
        sampled_inds = self.subsample(labels[0])
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[0][img_id] = labels[0][img_id][img_sampled_inds]
            labels[1][img_id] = labels[1][img_id][img_sampled_inds]
            labels[2][img_id] = labels[2][img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.roi_head_wrap.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        verbs = []
        ttcs = []

        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                verbs_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                ttcs_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.float32, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.roi_head_wrap.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[0][clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                verbs_in_image = gt_labels_in_image[1][clamped_matched_idxs_in_image].to(dtype=torch.int64)
                ttcs_in_image = gt_labels_in_image[2][clamped_matched_idxs_in_image].to(dtype=torch.float32)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.roi_head_wrap.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                verbs_in_image[bg_inds] = IGNORE_VERB_IDX_BG
                ttcs_in_image[bg_inds] = IGNORE_VERB_IDX_BG_FLOAT

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.roi_head_wrap.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                verbs_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                ttcs_in_image[ignore_inds] = -1.0  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            verbs.append(verbs_in_image)
            ttcs.append(ttcs_in_image)

        return matched_idxs, (labels, verbs, ttcs)

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        verb_logits,  # type: Tensor
        ttcs,
        box_regression,  # type: Tensor
        box_features_in,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        if self.additional_postprocessing_frequencies.device != device:
            self.additional_postprocessing_frequencies = self.additional_postprocessing_frequencies.to(device)

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.roi_head_wrap.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        verb_scores = torch.argmax(verb_logits[:, :-1], dim=-1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        verb_scores_list = verb_scores.split(boxes_per_image, 0)
        box_features_list = box_features_in.split(boxes_per_image, 0)
        if ttcs is not None:
            ttcs_scores_list = ttcs.split(boxes_per_image, 0)
        else:
            ttcs_scores_list = verb_scores_list

        all_idxs = []
        all_frame_ids = []
        all_boxes = []
        all_box_features = []
        all_scores = []
        all_nouns = []
        all_verbs = []
        all_ttcs = []
        for idx, boxes, box_features, scores, v_scores, ttcs_scores, image_shape in zip(
            range(len(pred_boxes_list)), pred_boxes_list, box_features_list, pred_scores_list, verb_scores_list, ttcs_scores_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]  # can place confidence into the BG class to reduce the weight of the argmax class
            labels = labels[:, 1:]

            init_scores = scores

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            idxs = torch.arange(box_features.shape[0])

            # remove low scoring boxes
            inds = torch.where(scores > self.roi_head_wrap.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            inds_in_2d = torch.where(init_scores > self.roi_head_wrap.score_thresh)
            verb_scores = v_scores[inds_in_2d[0]]
            ttc_scores = ttcs_scores[inds_in_2d[0]]
            box_features = box_features[inds_in_2d[0]]
            idxs = idxs[inds_in_2d[0]]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            verb_scores = verb_scores[keep]
            ttc_scores = ttc_scores[keep]
            box_features = box_features[keep]
            idxs = idxs[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.roi_head_wrap.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.roi_head_wrap.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            verb_scores = verb_scores[keep]
            ttc_scores = ttc_scores[keep]
            box_features = box_features[keep]
            idxs = idxs[keep]

            if self.additional_postprocessing and len(boxes) > 0:
                # "labels", "verb_scores", "ttc_scores" are already indices by this point
                # "ttc_values" are actual TTC values (or -1, if using a transformer head for the TTC prediction)
                # "boxes" are actual boxes
                box_noun_argmax_verbs = self.additional_postprocessing_frequencies[labels].argmax(dim=-1)
                box_noun_argmax_verb_freqs = self.additional_postprocessing_frequencies[labels, box_noun_argmax_verbs]
                box_num_train_occs = self.additional_postprocessing_frequencies[labels, verb_scores]

                replace_boxes = torch.logical_and(box_num_train_occs == 0, box_noun_argmax_verb_freqs > 0)
                verb_scores[replace_boxes] = box_noun_argmax_verbs[replace_boxes]

                # check for intersections
                boxes_torch = torch.tensor(boxes, device=device)

                nouns_torch_self = labels.repeat_interleave(len(boxes), dim=-1).view(len(boxes), len(boxes))
                nouns_torch_other = labels[None].repeat(len(boxes), 1, 1).view(len(boxes), len(boxes))
                verbs_torch_self = verb_scores.repeat_interleave(len(boxes), dim=-1).view(len(boxes), len(boxes))
                verbs_torch_other = verb_scores[None].repeat(len(boxes), 1, 1).view(len(boxes), len(boxes))

                boxes_torch_self = boxes_torch.repeat_interleave(len(boxes), dim=0).view(len(boxes), len(boxes), -1)
                boxes_torch_other = boxes_torch[None].repeat(len(boxes), 1, 1).view(len(boxes), len(boxes), -1)
                xs_left = torch.maximum(boxes_torch_self[..., 0], boxes_torch_other[..., 0])
                ys_top = torch.maximum(boxes_torch_self[..., 1], boxes_torch_other[..., 1])
                xs_right = torch.minimum(boxes_torch_self[..., 2], boxes_torch_other[..., 2])
                ys_bottom = torch.minimum(boxes_torch_self[..., 3], boxes_torch_other[..., 3])
                do_intersect = torch.logical_and(xs_left < xs_right, ys_top < ys_bottom)
                do_match_label = torch.logical_and(nouns_torch_self == nouns_torch_other, verbs_torch_self == verbs_torch_other)
                conflicts = torch.logical_and(~torch.eye(len(boxes), device=device, dtype=torch.bool), torch.logical_and(do_intersect, do_match_label))
                # "conflicts" will be symmetrical
                # we can keep the box if everything up to the diagonal is free
                keep = torch.tril(conflicts).sum(dim=-1) == 0
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                verb_scores = verb_scores[keep]
                ttc_scores = ttc_scores[keep]
                box_features = box_features[keep]
                idxs = idxs[keep]

            if targets is not None:
                all_frame_ids.append(targets[idx]["id"])
            else:
                all_frame_ids.append(None)
            all_idxs.append(idxs)
            all_boxes.append(boxes)
            all_box_features.append(box_features)
            all_scores.append(scores)
            all_nouns.append(labels)
            all_verbs.append(verb_scores)
            all_ttcs.append(ttc_scores)

        return all_idxs, all_frame_ids, all_boxes, all_box_features, all_scores, all_nouns, all_verbs, all_ttcs
