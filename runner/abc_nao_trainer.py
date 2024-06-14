import numbers
import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, _LRScheduler
from torchmetrics import Accuracy, MeanAbsoluteError
from warmup_scheduler import GradualWarmupScheduler

from runner.metrics_losses.losses import get_optimizer
from runner.metrics_losses.radam_optim import RAdam


class MyMultiStepLR(MultiStepLR, _LRScheduler):
    ...


class ABCNAOTrainer(LightningModule):
    def __init__(self, run_config, class_info, model):
        super().__init__()
        self.model = model
        self.run_config = run_config
        self.class_info = class_info
        self.criterion_dict = run_config["criterion"]
        self.verb_bg = run_config.get("verb_bg", False)

        self.verb_bg = run_config.get("verb_bg", False)

        self.set_preds_on(run_config["criterion"])
        self.log_keys = self.get_log_keys(run_config["criterion"])

        self.bg_weight = run_config.get("bg_weight", 1)
        self.all_class_w = run_config.get("all_class_w", False)

        if self.all_class_w:
            n_weights = class_info["weights"]["noun"]
            v_weights = class_info["weights"]["verb"]
        else:
            n_weights = np.ones(len(self.class_info["weights"]["noun"]))
            v_weights = np.ones(len(self.class_info["weights"]["verb"]))

        if self.bg_weight != 1:
            n_weights[0] = self.bg_weight
            
            # verb bg class is the last one 
            if self.verb_bg:
               v_weights = np.append(v_weights, self.bg_weight)

        else:
            n_weights[0] = n_weights.mean()
            v_weights = np.append(v_weights, v_weights.mean())

        self.noun_criterion = torch.nn.CrossEntropyLoss(torch.tensor(n_weights, dtype=torch.float32),reduction="mean")
        self.verb_criterion = torch.nn.CrossEntropyLoss(torch.tensor(v_weights, dtype=torch.float32),reduction="mean")
        
        self.ttc_criterion = torch.nn.SmoothL1Loss(beta = run_config["criterion"].get("ttc_beta", 1))
        self.loss_w_init = torch.FloatTensor(
            [
                val
                for k, val in run_config["criterion"].items()
                if (
                    isinstance(val, numbers.Number)
                    and k not in {"no_samples", "obj_prop", "lm_decay", "ttc_beta", "obj_prop_rate"}
                )
            ]
        )
        self.loss_w = self.loss_w_init / self.loss_w_init.sum()
        self.obj_prop_rate = self.criterion_dict.get("obj_prop_rate", 1)
        self.curr_obj_prop_w = 1

        self.lm_decay = self.criterion_dict.get("lm_decay", 0)
        self.obj_prop_rate = self.criterion_dict.get("obj_prop_rate", 1)
        self.curr_obj_prop_w = torch.Tensor([1.0])

        self.learning_rate = self.run_config["optimizer"]["lr"]
        self.use_scheduler = self.run_config["scheduler"]["use"]

        self.classif_metrics = nn.ModuleDict(self.setup_class_metrics(self.class_info["no_classes"]))
        self.localiz_metrics = nn.ModuleDict(self.setup_localiz_metrics())
        self.ttc_metrics = nn.ModuleDict(self.setup_ttc_metrics())

        self.setup_grab_inputs()

    def get_log_keys(self, criterion_dict):
        log_keys = ["loss"]
        if criterion_dict["noun"] > 0:
            log_keys.append("noun_loss")
        if criterion_dict["verb"] > 0:
            log_keys.append("verb_loss")
        if criterion_dict["ttc"] > 0:
            log_keys.append("ttc_loss")
        return log_keys

    def set_preds_on(self, criterion_dict):
        self.classify_noun = criterion_dict.get("noun", 0) > 0
        self.classify_verb = criterion_dict.get("verb", 0) > 0
        self.ttc_pred_on = criterion_dict.get("ttc", 0) > 0
        # self.bbox_pred_on = criterion_dict.get("box", 0) > 0

    def setup_class_metrics(self, no_classes, top_k=5):
        class_metrics = {}
        if self.classify_noun:
            class_metrics["noun_train_acc"] = Accuracy(num_classes=no_classes["noun"], average="macro", top_k=top_k)
            class_metrics["noun_val_acc"] = Accuracy(num_classes=no_classes["noun"], average="macro", top_k=top_k)
        if self.classify_verb:
            class_metrics["verb_train_acc"] = Accuracy(num_classes=no_classes["verb"], average="macro", top_k=top_k)
            class_metrics["verb_val_acc"] = Accuracy(num_classes=no_classes["verb"], average="macro", top_k=top_k)

        return class_metrics

    def setup_localiz_metrics(self):
        raise NotImplementedError

    def setup_ttc_metrics(self):
        if self.ttc_pred_on:
            return {"ttc_mae_val": MeanAbsoluteError(), "ttc_mae_train": MeanAbsoluteError()}

    # this is overriden in the subclasses for vsnao
    def setup_grab_inputs(self):
        if self.run_config["narration_embeds"]["use"]:
            if self.run_config["narration_embeds"]["args"]["text_pooling"] == "self_attention":
                grab_inputs = lambda inputs: {
                    "image": inputs["image"],
                    "language_f": inputs["language_f"],
                    "pad_mask": inputs["pad_mask"],
                }
            else:
                grab_inputs = lambda inputs: {"image": inputs["image"], "language_f": inputs["language_f"]}
        else:
            grab_inputs = lambda inputs: {"image": inputs["image"]}

        a = grab_inputs

        if self.run_config.get("hand_args", {"use": False})["use"]:
            b = lambda inputs: {**a(inputs), "hand_poses": inputs["hand_poses"], "hand_boxes": inputs["hand_boxes"]}
        else:
            b = a

        if self.run_config["narration_embeds"].get("res50_f", False) or self.run_config["narration_embeds"].get("slowfast_f_v", False):
            c = lambda inputs:{**b(inputs), "visual_features": inputs["visual_features"]}
        else:
            c = b

        self.grab_inputs = c

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.model.call_model_epoch_triggers(self.current_epoch)

        outputs = self.batch_step(batch, "train", batch_idx)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.batch_step(batch, "val", batch_idx)
        return outputs

    def batch_step(self, batch, src, idx):
        raise NotImplementedError

    def training_epoch_end(self, outputs) -> None:
        if self.lm_decay:
            self.loss_w_init[-1] *= self.lm_decay
            self.loss_w = self.loss_w_init / self.loss_w_init.sum()

        self.curr_obj_prop_w *= self.obj_prop_rate

        self.model.epoch = self.current_epoch
        self.log_outs("train", outputs)

    def validation_epoch_end(self, outputs) -> None:
        self.log_outs("val", outputs)

    def log_outs(self, source, outputs):
        raise NotImplementedError

    def filter_model_params(self, model, optimizer_cfg):
        return model.parameters()

    def configure_optimizers(self):
        opt_class = get_optimizer(self.run_config["optimizer"]["name"])
        params = self.filter_model_params(self.model, self.run_config["optimizer"])
        if opt_class in {torch.optim.Adam, torch.optim.AdamW, RAdam}:
            optimizer = opt_class(
                params,
                lr=self.learning_rate,
                weight_decay=self.run_config["optimizer"]["weight_decay"],
            )
        else:
            optimizer = opt_class(
                params,
                lr=self.learning_rate,
                weight_decay=self.run_config["optimizer"]["weight_decay"],
                momentum=self.run_config["optimizer"]["momentum"],
            )

        if self.use_scheduler:
            scheduler = get_scheduler(self.run_config["scheduler"], optimizer)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer


def get_scheduler(scheduler_args, optimizer):
    if scheduler_args["name"] == "exponential":
        return ExponentialLR(optimizer, gamma=scheduler_args["gamma"], verbose=True)

    if scheduler_args["name"] == "warmup":
        if scheduler_args["after_warmup"] == "exponential":
            after_scheduler = ExponentialLR(optimizer, gamma=scheduler_args["gamma"], verbose=True)
        elif not scheduler_args["after_warmup"]:
            after_scheduler = None
        elif scheduler_args["after_warmup"] == "multistep":
            after_scheduler = MyMultiStepLR(
                optimizer=optimizer,
                gamma=scheduler_args["gamma"],
                verbose=True,
                milestones=scheduler_args["milestones"],
            )
        else:
            raise NotImplementedError(f"after warmup {scheduler_args['after_warmup']} not implemented")

        return GradualWarmupScheduler(
            optimizer,
            scheduler_args["multiplier"],
            total_epoch=scheduler_args["total_epoch"],
            after_scheduler=after_scheduler,
        )
    elif scheduler_args["name"] == "multistep":
        after_scheduler = MyMultiStepLR(
            optimizer=optimizer,
            gamma=scheduler_args["gamma"],
            verbose=True,
            milestones=scheduler_args["milestones"],
        )
        return after_scheduler
