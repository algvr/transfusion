import json
import os

import torch
from torchmetrics import Accuracy
import wandb

from modeling.obj_detection.roi_wrappers import IGNORE_VERB_IDX_BG
from modeling.ttc_pred import TTCPredictionHead
from runner.abc_nao_trainer import ABCNAOTrainer
from runner.metrics_losses.ego_metrics import *
from runner.metrics_losses.losses import box_loss


class EgoNAOTrainer(ABCNAOTrainer):
    def __init__(self, run_config, class_infos, model) -> None:
        super().__init__(run_config, class_infos, model)

        self.val_ego_metric = get_ego_metrics(run_config["criterion"], top_k=5)
        self.train_ego_metric = get_ego_metrics(run_config["criterion"], top_k=5)
        self.ignore_verb_idx_bg = IGNORE_VERB_IDX_BG
        self.ignore_ttc_idx = float(IGNORE_VERB_IDX_BG)

        # we have 88 classes for Ego4d, from 0 to 87, 0 is bg
        # we want to remap the last one to the first for LM loss
        self.last_noun_cls_idx_good = class_infos["no_classes"]["noun"] - 1
        # need this to remap the verb bg class of 999 to the last position
        # verbs are 0 to 74, where 74 is bg
        self.last_verb_cls_idx_w_bg = class_infos["no_classes"]["verb"] - 1
        self.verb_bg = run_config.get("verb_bg", False)
        self.ttc_bg = run_config.get("ttc_bg", False)
        self.ttc_bg_val = run_config.get("ttc_bg_val", False)
       
        if self.lm_head_on:
            self.lm_noun_criterion = torch.nn.CrossEntropyLoss(
                reduction="mean"
            )
            self.lm_verb_criterion = torch.nn.CrossEntropyLoss(
                reduction="mean"
            )
    
    def set_preds_on(self, criterion_dict):
        super().set_preds_on(criterion_dict)
        self.bbox_pred_on = criterion_dict.get("bbox", 0) > 0
        self.obj_head_on = criterion_dict.get("obj_prop", 0) > 0
        self.lm_head_on = criterion_dict.get("lm", 0) > 0

    def get_log_keys(self, criterion_config):
        log_keys = super().get_log_keys(criterion_config)
        if self.bbox_pred_on:
            log_keys.append("bbox_loss")
            self.train_only_log_keys = ["bbox_loss", "objectness_loss", "loss_rpn_box_reg"]
        if self.lm_head_on:
            self.train_only_log_keys.append("lm_loss")

        return log_keys

    def load_state_dict(self, state_dict, strict=True):
        state_dict_keys = list(state_dict.keys())
        for k in state_dict_keys:
            if "pos_embedding" in k:
                ckpt_pos_embedding = state_dict[k]
                # have fewer patches if training on smaller input resolutions, was adapted after
                # the entries are identical for sin embeddings
                if ckpt_pos_embedding.shape[1] < self.state_dict()[k].shape[1]:
                    state_dict[k] = self.state_dict()[k]
            if any([f"inner_blocks.{idx}.0." in k for idx in [0, 1, 2, 3]]) or any([f"layer_blocks.{idx}.0." in k for idx in [0, 1, 2, 3]]):
                state_dict[k.replace(".0.weight", ".weight").replace(".0.bias", ".bias")] = state_dict[k]
                del state_dict[k]

            if f"head.conv.0.0." in k:
                state_dict[k.replace(".0.0", "")] = state_dict[k]
                del state_dict[k]
        
        position_ids_key = "model.narr_pooling_layer.encoder.0.auto_model.embeddings.position_ids"
        if position_ids_key not in state_dict:
            try:
                state_dict[position_ids_key] = self.model.narr_pooling_layer.encoder[0].auto_model.embeddings.position_ids
            except:
                pass

        return super().load_state_dict(state_dict, strict)

    def setup_class_metrics(self, no_classes, top_k=1):
        # return super().setup_class_metrics(no_classes, top_k=top_k)
        class_metrics = {}
        if self.lm_head_on:
            class_metrics["noun_lm_train_acc"] = Accuracy(
                num_classes=no_classes["noun"] - 1, average="macro", top_k=top_k
            )
            class_metrics["noun_lm_val_acc"] = Accuracy(
                num_classes=no_classes["noun"] - 1, average="macro", top_k=top_k
            )
            if self.classify_verb:
                class_metrics["verb_lm_train_acc"] = Accuracy(
                    num_classes=no_classes["verb"] - 1, average="macro", top_k=top_k
                )
                class_metrics["verb_lm_val_acc"] = Accuracy(
                    num_classes=no_classes["verb"] - 1, average="macro", top_k=top_k
                )

        return class_metrics

    def setup_localiz_metrics(self):
        pass

    def update_ego_metrics(self, src, ego_pred, ego_gt, criterion_dict):
        if src == "val":
            metric = self.val_ego_metric
        elif src == "train":
            metric = self.train_ego_metric
        else:
            raise NotImplementedError("Wrong source")

        no_ex = len(ego_pred)
        for i in range(no_ex):
            ego_gt[i]["nouns"] = ego_gt[i]["labels"]
            ego_gt[i]["boxes"] = ego_gt[i]["boxes"].int()
            ego_pred[i]["boxes"] = ego_pred[i]["boxes"].int()

            ks = [*ego_pred[i].keys()]
            for k in ks:
                if k in ["idxs", "frame_ids"]:
                    del ego_pred[i][k]
                    continue
                if k not in ["id", "orig_shape"]:
                    ego_pred[i][k] = ego_pred[i][k].detach().cpu().numpy()

            ks = [*ego_gt[i].keys()]
            for k in ks:
                if k in ["idxs", "frame_ids"]:
                    del ego_gt[i][k]
                    continue
                if k not in ["id", "orig_shape"]:
                    ego_gt[i][k] = ego_gt[i][k].detach().cpu().numpy()

            metric.add(preds=ego_pred[i], labels=ego_gt[i])
            ego_pred[i]["id"] = ego_gt[i]["id"]
            ego_pred[i]["orig_shape"] = ego_gt[i]["orig_shape"]

        return ego_pred

    def log_outs(self, src, outputs, on_step=False, on_epoch=True):
        print(f"{len(outputs)=}")
        if len(outputs) > 0:
            print() # f"{outputs[0]=}")
        if src == "val":
            metric = self.val_ego_metric
        elif src == "train":
            metric = self.train_ego_metric

        vals = metric.evaluate()
        names = metric.get_short_names()
        metric.reset()

        for name, val in zip(names, vals):
            self.log(f"{name}_{src}", val, on_epoch=on_epoch, on_step=on_step, rank_zero_only=True, sync_dist=True)
            print(f"{name}_{src}: {val:.5f}")

        for k in self.log_keys:
            self.log(
                f"{src}_{k}",
                torch.stack([out[k] for out in outputs]).mean(),
                on_epoch=on_epoch,
                on_step=on_step,
                rank_zero_only=True,
                sync_dist=True,
            )

        if src == "train":
            for k in self.train_only_log_keys:
                self.log(
                    f"train_{k}",
                    torch.stack([out[k] for out in outputs]).mean(),
                    on_epoch=on_epoch,
                    on_step=on_step,
                    rank_zero_only=True,
                    sync_dist=True,
                )
            if self.lm_head_on:
                self.log(
                    "noun_lm_train_acc",
                    self.classif_metrics["noun_lm_train_acc"],
                    on_epoch=on_epoch,
                    on_step=on_step,
                )
                if self.classify_verb:
                    self.log(
                        "verb_lm_train_acc",
                        self.classif_metrics["verb_lm_train_acc"],
                        on_epoch=on_epoch,
                        on_step=on_step,
                    )

        elif src == "val":
            # generate validation JSON
            val_dict = {
                "version": "1.0",
                "challenge": "ego4d_short_term_object_interaction_anticipation",
                "epoch": self.current_epoch,
                "results": {},
            }
            for output in outputs:
                for pred_coll in output["preds"]:
                    pred_coll_out = []
                    # 100 -> 5
                    for pred_idx in range(min(len(pred_coll["scores"]), 5)):
                        box, noun, verb, ttc, score = [
                            pred_coll[key][pred_idx] for key in ["boxes", "nouns", "verbs", "ttcs", "scores"]
                        ]

                        if self.run_config["resize_spec"][-1] != 0 and not isinstance(self.run_config["resize_spec"], list):
                            height_ratio = pred_coll["orig_shape"][0] / self.run_config["resize_spec"][0]
                            width_ratio = pred_coll["orig_shape"][1] / self.run_config["resize_spec"][1]
                        elif isinstance(self.run_config["resize_spec"][0], list):
                            # the RCNN transform picks the last entry which should be the largest
                            height_ratio = 1
                            width_ratio = 1
                        else:
                            height_ratio = 1
                            width_ratio = 1

                        # box format: [x1, y1, x2, y2]
                        for coord_idx in range(len(box)):
                            box[coord_idx] *= {0: width_ratio, 1: height_ratio}[coord_idx % 2]

                        if noun == self.last_noun_cls_idx_good:
                            noun = 0  # undo BG/class 0 switch; no need to check mapping

                        pred_coll_out.append(
                            {
                                "box": list(map(float, box.tolist())),
                                "noun_category_id": int(noun),
                                "verb_category_id": int(verb),
                                "time_to_contact": float(ttc),
                                "score": float(score),
                            }
                        )

                    val_dict["results"][pred_coll["id"]] = pred_coll_out

            run_name = wandb.run.name
            result_dir = os.path.join(os.path.expandvars("${CODE}"), "results", run_name)
            result_filename_base = f"{run_name}_val_{self.current_epoch}"
            result_path = os.path.join(result_dir, result_filename_base + ".json")
            os.makedirs(result_dir, exist_ok=True)
            with open(result_path, "w") as f:
                f.write(json.dumps(val_dict))

            wandb.run.log_artifact(os.path.abspath(result_path), name=f"{run_name}_val_results", type="json")

    def forward(self, x, targets=None):
        preds = self.model(self.grab_inputs(x), targets)
        return preds

    def validation_epoch_end(self, outputs) -> None:
        self.log_outs("val", outputs)

    def training_step(self, batch, batch_idx):
        src = "train"
        self.train()
        if batch_idx == 0:
            self.model.call_model_epoch_triggers(self.current_epoch)

            freeze_backbone_epoch = self.run_config.get("freeze_backbone_at_epoch", -1)

            if self.current_epoch >= freeze_backbone_epoch and freeze_backbone_epoch != -1:
                for name, param in self.model.named_parameters(recurse=True):
                    if "roi" not in name:
                        param.requires_grad = False
                
                print("FROZE MODEL EXCEPT ROI")

        gt_data = batch["targets"][0]["boxes"]

        full_bbox_loss = torch.tensor(0).type_as(gt_data)
        noun_loss = torch.tensor(0).type_as(gt_data)
        verb_loss = torch.tensor(0).type_as(gt_data)
        ttc_loss = torch.tensor(0).type_as(gt_data)
        loss_objectness = torch.tensor(0).type_as(gt_data)
        loss_rpn_box_reg = torch.tensor(0).type_as(gt_data)
        lm_loss = torch.tensor(0).type_as(gt_data)
        outputs = self.model(self.grab_inputs(batch), batch["targets"])
     
        noun_labels = (
            torch.clone(outputs["roi_outputs"]["labels"]) if not self.classify_verb else [torch.clone(label) for label in outputs["roi_outputs"]["labels"][0]]
        )

        box_lossy = box_loss(
            outputs["roi_outputs"]["class_logits"],
            outputs["roi_outputs"]["box_regression"],
            noun_labels,
            outputs["roi_outputs"]["reg_targets"],
        )
        full_bbox_loss = box_lossy

        if self.obj_head_on:
            loss_objectness, loss_rpn_box_reg = self.model.compute_rpn_loss(
                outputs["proposals"]["objectness"],
                outputs["proposals"]["pred_bbox_deltas"],
                outputs["proposals"]["labels"],
                # noun_labels,
                outputs["proposals"]["reg_targets"],
            )
            full_bbox_loss += loss_objectness + loss_rpn_box_reg

        if self.classify_noun:
            noun_logits = outputs["roi_outputs"]["class_logits"]
            targets = torch.cat(noun_labels, dim=0)
            noun_loss = self.noun_criterion(noun_logits + 1e-6, targets)

        if self.classify_verb:
            verb_logits = outputs["roi_outputs"]["verb_logits"]
            targets = torch.cat(outputs["roi_outputs"]["labels"][1], dim=0)
            # replace with the proper bg idx
            v_targets = torch.where(targets == self.ignore_verb_idx_bg, self.last_verb_cls_idx_w_bg, targets)
            if not self.verb_bg:
                v_idxs = torch.where(targets != self.ignore_verb_idx_bg)[0]
                verb_logits = verb_logits[v_idxs]
                v_targets = targets[v_idxs]

            verb_loss = self.verb_criterion(verb_logits + 1e-6, v_targets)

        # "with torch.no_grad()" moved inside function to allow training TTC head
        preds = self.model.dets_from_outs(outputs,
                                          orig_img_shapes=outputs["original_image_sizes"],
                                          hand_poses=batch.get("hand_poses"),
                                          hand_boxes=batch.get("hand_boxes"))
        if self.ttc_pred_on:
            if isinstance(self.model.rcnn_model.rcnn_to_wrap.roi_heads.ttc_pred_layer, TTCPredictionHead):
                # TODO: account for cases with multiple GT bboxes
                ttc_logits = torch.cat([p["ttcs"][ p["ttcs"] >= 0 ] for p in preds], dim=0)
                ttcs_per_img = outputs["roi_outputs"]["ttcs"].shape[0] // len(batch["targets"])
                if ttc_logits.shape[0] > 0:
                    ttc_logit_idxs = torch.cat([idx * ttcs_per_img + p["idxs"][ p["ttcs"] >= 0 ] for idx, p in enumerate(preds) if (p["ttcs"] >= 0).sum() > 0], dim=0)
                    outputs["roi_outputs"]["ttcs"][ttc_logit_idxs.detach().cpu().long().tolist()] = ttc_logits.to(dtype=outputs["roi_outputs"]["ttcs"].dtype, device=outputs["roi_outputs"]["ttcs"].device)

                ttc_targets = torch.cat([batch["targets"][idx]["ttcs"][:1].repeat_interleave(p["ttcs"].shape[0], dim=0) for idx, p in enumerate(preds)], dim=0)
                not_nan = ~torch.isnan(ttc_targets)
                ttc_logits = ttc_logits[not_nan]
                ttc_targets = ttc_targets[not_nan]

                #ttc_split = outputs["roi_outputs"]["ttcs"].split(ttcs_per_img)
                #for sample_idx in range(len(batch["targets"])):
                #    preds["ttcs"][sample_idx] = ttc_split[sample_idx].detach()
            else:
                ttc_logits = outputs["roi_outputs"]["ttcs"]
                ttc_targets = torch.cat(outputs["roi_outputs"]["labels"][2], dim=0)
                if not self.ttc_bg:
                    ttc_idxs = torch.where(targets != self.ignore_verb_idx_bg)[0]
                    ttc_logits = ttc_logits[ttc_idxs]
                    ttc_targets = ttc_targets[ttc_idxs]
                else:
                    ttc_targets = torch.where(
                        ttc_targets == self.ignore_verb_idx_bg, self.ttc_bg_val, ttc_targets.double()
                    ).float()

            if ttc_logits.shape[0] > 0:
                ttc_loss = self.ttc_criterion(ttc_logits, ttc_targets)

        if self.lm_head_on:
            lm_outs = outputs["lm"]
            lm_noun_logits = lm_outs["noun_logits"]
            lm_noun_targets = torch.stack([batch["targets"][i]["labels"][0] for i in range(len(batch["targets"]))])
            # we have moved the 1st class (idx 0) as the last one s.t. we have bg class 0 for EGO
            lm_noun_targets = torch.where(lm_noun_targets == self.last_noun_cls_idx_good, 0, lm_noun_targets)
            lm_loss = self.lm_noun_criterion(lm_noun_logits, lm_noun_targets)
            self.classif_metrics["noun_lm_train_acc"].update(lm_noun_logits, lm_noun_targets)
            if self.classify_verb:
                lm_verb_logits = lm_outs["verb_logits"]
                lm_verb_targets = torch.stack([batch["targets"][i]["verbs"][0] for i in range(len(batch["targets"]))])
                lm_loss += self.lm_verb_criterion(lm_verb_logits, lm_verb_targets)
                self.classif_metrics["verb_lm_train_acc"].update(lm_verb_logits, lm_verb_targets)
                lm_loss = lm_loss / 2

        losses = torch.stack([full_bbox_loss, noun_loss, verb_loss, ttc_loss, lm_loss])
        if self.criterion_dict["agg"] == "mean":
            loss = (losses * self.loss_w_init.type_as(losses)).sum()
        else:
            loss = losses.sum()

        self.update_ego_metrics(
            src,
            ego_gt=batch["targets"],
            ego_pred=preds,
            criterion_dict=self.run_config["criterion"],
        )

        return {
            "loss": loss,
            "bbox_loss": box_lossy.detach(),
            "objectness_loss": loss_objectness.detach(),
            "loss_rpn_box_reg": loss_rpn_box_reg.detach(),
            "noun_loss": noun_loss.detach(),
            "verb_loss": verb_loss.detach(),
            "ttc_loss": ttc_loss.detach(),
            "lm_loss": lm_loss.detach(),
        }

    def validation_step(self, batch, idx):
        gt_data = batch["targets"][0]["boxes"]
        src = "val"
        self.eval()
        # inputs format
        # images of shape N x 3 x W x H
        # target format: [{"boxes", "labels"}]

        loss_box_reg = torch.tensor(0).type_as(gt_data)
        noun_loss = torch.tensor(0).type_as(gt_data)
        verb_loss = torch.tensor(0).type_as(gt_data)
        ttc_loss = torch.tensor(0).type_as(gt_data)
        lm_loss = torch.tensor(0).type_as(gt_data)

        with torch.no_grad():
            preds = self.model.forward_w_dets(self.grab_inputs(batch), targets=batch["targets"])

        preds = self.update_ego_metrics(
            src,
            ego_gt=batch["targets"],
            ego_pred=preds,
            criterion_dict=self.run_config["criterion"],
        )

        losses = torch.stack([loss_box_reg, noun_loss, verb_loss, ttc_loss, lm_loss])
        if self.criterion_dict["agg"] == "mean":
            loss = (losses * self.loss_w.type_as(losses)).sum()
        else:
            loss = losses.sum()

        return {
            "loss": loss,
            "bbox_loss": loss_box_reg.detach(),
            "noun_loss": noun_loss.detach(),
            "verb_loss": verb_loss.detach(),
            "preds": preds,
            "ttc_loss": ttc_loss.detach(),
            "lm_loss": lm_loss.detach(),
        }

    def filter_model_params(self, model, optimizer_cfg):
        if optimizer_cfg["sep_encoders"]:
            divizor = optimizer_cfg["sep_encoders"]["div_rate"]
            ttc_lr = optimizer_cfg["sep_encoders"].get("ttc_lr")
            divizor_ttc = optimizer_cfg["sep_encoders"].get("ttc_rate")
            params = []
            # parameters with lowered learning rate
            if "RCNNWrapper" in str(type(model)):
                params.append(
                    {
                        "params": model.rcnn_to_wrap.backbone.body.parameters(),
                        "lr": self.learning_rate / divizor,
                    }
                )
            else:
                params.append({"params": model.narr_pooling_layer.parameters(), "lr": self.learning_rate / divizor})
                params.append(
                    {
                        "params": model.rcnn_model.rcnn_to_wrap.backbone.body.parameters(),
                        "lr": self.learning_rate / divizor,
                    }
                )

            custom_ttc_lr = False
            try:
                if divizor_ttc is not None:
                    if getattr(model.rcnn_model.rcnn_to_wrap.roi_heads, "ttc_pred_layer"):
                        params.append(
                            {
                                "params": model.rcnn_model.rcnn_to_wrap.roi_heads.ttc_pred_layer.parameters(),
                                "lr": self.learning_rate / divizor_ttc,
                            }
                        )
                        custom_ttc_lr = True
                elif ttc_lr is not None:
                    if getattr(model.rcnn_model.rcnn_to_wrap.roi_heads, "ttc_pred_layer"):
                        params.append(
                            {
                                "params": model.rcnn_model.rcnn_to_wrap.roi_heads.ttc_pred_layer.parameters(),
                                "lr": ttc_lr,
                            }
                        )
                        custom_ttc_lr = True
            except:
                pass

            # parameters with the default learning rate
            main_params = []
            for k, v in model.named_parameters():
                if "narr_pooling_layer" in k or "rcnn_to_wrap.backbone.body" in k or ("ttc_pred_layer" in k and custom_ttc_lr):
                    continue
                main_params.append(v)

            params.append({"params": main_params})
        else:
            params = model.parameters()

        return params


def get_ego_metrics(criterion_dict, top_k):
    return STAMeanAveragePrecision(top_k=top_k)
