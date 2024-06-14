from os.path import expandvars

from modeling.cross_fusion.cross_f_wrapper import CrossFusionWrapper
from modeling.cross_fusion.ego_fusion.cross_f_box_vis_language_wrapper import \
    VisLangFusionBoxWrapper
from modeling.cross_fusion.ego_fusion.cross_f_box_wrapper import (
    CrossFusionBoxWrapper, CrossFusionBoxWrapperShared)
from modeling.model_loading import (get_flow_adapter_w_weights,
                                         get_full_rcnn_weights,
                                         get_visual_backbone_w_weights)
from modeling.obj_detection.rcnn_factory import get_rcnn_model
from modeling.resnet.resnet_wrapper import get_resnet, get_resnet_wrapper
from runner.utils.data_transforms import get_denormalize, get_norm_mean_std


def get_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def get_base_models(model_cfg):
    if model_cfg["type"] == "resnet":
        base_model = get_resnet(resnet_type=model_cfg["flavor"], pretrained=expandvars(model_cfg["pretrained"]))
    return base_model


def get_model(experiment, model_cfg, run_cfg, num_classes, train_noun_verb_frequencies):
    if experiment != "egonao":
        if model_cfg["type"] == "resnet":
            to_return = get_resnet_wrapper
        else:
            raise ValueError(f"{model_cfg['type']=} not implemented")

        return to_return(run_cfg, model_cfg, num_classes)
    else:
        num_classes["noun"] = num_classes["noun"] if run_cfg["criterion"]["noun"] else 0
        num_classes["verb"] = num_classes["verb"] if run_cfg["criterion"]["verb"] else 0

        ttc_hand_head_args = model_cfg.get("ttc_hand_head")
        if ttc_hand_head_args is not None:
            ttc_hand_head_args["step"] = run_cfg["hand_args"]["step"]
            ttc_hand_head_args["num_steps"] = run_cfg["hand_args"]["num_steps"]

        additional_postprocessing = model_cfg.get("additional_postprocessing", False)
        return get_rcnn_model(
            model_cfg["type"],
            pretrained=expandvars(model_cfg["pretrained"]),
            trainable_backbone_layers=model_cfg["trainable_layers"],
            num_classes=num_classes,
            box_1_dropout=model_cfg["box_1_dropout"],
            box_2_dropout=model_cfg["box_2_dropout"],
            classif_dropout=run_cfg["class_dropout"],
            representation_size=model_cfg["representation_size"],
            dual_stream=False,
            resize_spec=run_cfg["resize_spec"],
            rcnn_kwargs=model_cfg["rcnn_kwargs"],
            load_fpn_rpn=model_cfg["load_fpn_rpn"],
            fpn_layers_to_return=model_cfg.get("fpn_return_layers", None),
            train_ep = model_cfg["train_ep"],
            batch_norm=model_cfg.get("batch_norm", {"use":False}),
            adapt_to_detectron=model_cfg.get("adapt_to_detectron", False),
            verb_classifier_args=model_cfg.get("verb_classifier", None),
            ttc_on=run_cfg["criterion"].get("ttc", False),
            ttc_hand_head_args=ttc_hand_head_args,
            replace_heads = run_cfg.get("replace_heads", False),
            fpn_out_channels=model_cfg.get("fpn_out_channels", 256),
            additional_postprocessing=additional_postprocessing,
            train_noun_verb_frequencies=train_noun_verb_frequencies
        )


def get_fusion_model(base_model, model_cfg, run_cfg, class_sizes):
    if run_cfg["narration_embeds"]["use"]:
        if run_cfg["narr_fusion"]["model"] == "cross_f":
            ##Bbox models
            if run_cfg["experiment"] == "egonao":
                if run_cfg["narration_embeds"].get("res50_f", False) or run_cfg["narration_embeds"].get("slowfast_f_v",False):
                    clzz = VisLangFusionBoxWrapper
                    if run_cfg["narration_embeds"].get("res50_f", False):
                        run_cfg["flow_args"]["vis_in_features"] =2048
                    else:
                        run_cfg["flow_args"]["vis_in_features"] = 2304
                    obj = clzz(base_model, 
                                run_cfg["narr_fusion"],
                                narr_embed_args=run_cfg["narration_embeds"]["args"], 
                                criterion = run_cfg["criterion"],
                                vis_args = run_cfg["flow_args"]
                                )
                   
                    return obj
                
                # Language only models
                if run_cfg["narr_fusion"]["share_encoders"]:
                    clzz = CrossFusionBoxWrapperShared
                else:
                    clzz = CrossFusionBoxWrapper
                return clzz(
                    base_model,
                    run_cfg["narr_fusion"],
                    narr_embed_args=run_cfg["narration_embeds"]["args"],
                    criterion = run_cfg["criterion"]
                    )
            else:
                back_to_image = run_cfg["heatmap_type"] != "gaussian_dist"
                return CrossFusionWrapper(
                base_model,
                back_to_image,
                run_cfg["narr_fusion"],
                narr_embed_args=run_cfg["narration_embeds"]["args"],
            )
        else:
            raise NotImplementedError(f'{run_cfg["narr_fusion"]["model"]=} is not implemented as fusion model.')
    else:
        return base_model


if __name__ == "__main__":
    run_config = {"criterion": {"noun": 0, "verb": 0, "mae": 1}, "w_sigmoid": False, "resize_spec": [200, 200]}
  