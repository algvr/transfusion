import torch


def get_flow_adapter_w_weights(base_model, weights_path, remove_to_wrap=False):
    """Load a flow adapter instance weights that have been pretrained by us. It can be the motion stream
    in a dual stream configuration or the feature extraction stage motion only classification"""
    # pl_module = pl_module_class.load_from_checkpoint(weights_path, strict=False)
    model_dict = torch.load(weights_path, map_location="cpu")
    # first we get rid of the model key prefix that comes from lightning checkpoint saver
    model_state_dict = {k[6:]: v for k, v in model_dict["state_dict"].items() if k.startswith("model")}
    # then we need to get rid of the to_wrap prefix that comes from mobilenet wrapper
    if remove_to_wrap:
        model_state_dict = {k[8:]: v for k, v in model_state_dict.items() if k.startswith("to_wrap")}

    # might have missing classifier keys because they are deleted since we use the wrapper noun, verb classifier
    base_model.load_state_dict(model_state_dict, strict=True)

    return base_model


def get_visual_backbone_w_weights(base_model, weights_path):
    """Used to load weights for an own pretrained rcnn backbone. They should come from the proper model hierarchy e.g.
    Ego_nao_trainer, rcnn_wrapper, torchvision.rcnn"""
    # pl_module = pl_module_class.load_from_checkpoint(weights_path, strict=False)
    model_dict = torch.load(weights_path, map_location="cpu")
    # first we get rid of the model key prefix that comes from lightning checkpoint saver
    model_state_dict = {k[6:]: v for k, v in model_dict["state_dict"].items() if k.startswith("model")}
    # then we get rid of the rcnn_to_wrap prefix that comes from the rcnn wrapper
    model_state_dict = {k[13:]: v for k, v in model_state_dict.items() if k.startswith("rcnn_to_wrap")}
    # then we get rid of the backbone prefix that comes from the torchvision.rcnn model
    model_state_dict = {k[9:]: v for k, v in model_state_dict.items() if k.startswith("backbone")}

    base_model.load_state_dict(model_state_dict, strict=False)

    return base_model


def get_full_rcnn_weights(base_model, weights_path):
    """Used to load the all own pretrained weights for a rcnn model, including fpn and prediction layers.
    Base model must be an FasterRCNN wrapper instance. If classifier weights do no match they will be skipped"""
    # pl_module = pl_module_class.load_from_checkpoint(weights_path, strict=False)
    model_dict = torch.load(weights_path, map_location="cpu")
    # first we get rid of the model key prefix that comes from lightning checkpoint saver
    model_state_dict = {k[6:]: v for k, v in model_dict["state_dict"].items() if k.startswith("model")}

    cls_keys_to_skip = {
        "rcnn_to_wrap.roi_heads.noun_classifier.weight",
        "rcnn_to_wrap.roi_heads.noun_classifier.bias",
        "rcnn_to_wrap.roi_heads.box_regressor.1.weight",
        "rcnn_to_wrap.roi_heads.box_regressor.1.bias",
    }

    if (
        base_model.rcnn_to_wrap.roi_heads.noun_classifier.weight.size()
        != model_state_dict["rcnn_to_wrap.roi_heads.noun_classifier.weight"].size()
    ):
        print("CLASSIFIER WEIGHTS DO NOT MATCH, SKIPPING THEM")
        for key in cls_keys_to_skip:
            model_state_dict.pop(key)

    repr_keys_to_skip = {
        "rcnn_to_wrap.roi_heads.roi_head_wrap.box_head.fc6.weight",
        "rcnn_to_wrap.roi_heads.roi_head_wrap.box_head.fc6.bias",
        "rcnn_to_wrap.roi_heads.roi_head_wrap.box_head.fc7.weight",
        "rcnn_to_wrap.roi_heads.roi_head_wrap.box_head.fc7.bias",
    }

    repr_size = base_model.rcnn_to_wrap.roi_heads.roi_head_wrap.box_head.fc6.weight.size()
    if repr_size != model_state_dict["rcnn_to_wrap.roi_heads.roi_head_wrap.box_head.fc6.weight"].size():
        print("REPRESENTATION WEIGHTS DO NOT MATCH, SKIPPING THEM")
        for key in repr_keys_to_skip:
            model_state_dict.pop(key)

    base_model.load_state_dict(model_state_dict, strict=False)

    return base_model
