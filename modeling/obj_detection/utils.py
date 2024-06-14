from torch import nn

def replace_frozen_bn(torchvision_rcnn_model, batch_norm):
    if torchvision_rcnn_model.model_type == "mobilenet":
        num_layers = 16
        for i in range(num_layers+1):
            key = str(i)
            layer = torchvision_rcnn_model.backbone.body[key]
            torchvision_rcnn_model.backbone.body[key] = replace_layer_bn(layer, batch_norm)
    if torchvision_rcnn_model.model_type == "resnet":
        torchvision_rcnn_model.backbone.body.bn1 = replace_single_frozen_bn(torchvision_rcnn_model.backbone.body.bn1, 
                                                                        batch_norm)

        for i in range(1, 5):
            res_layer = torchvision_rcnn_model.backbone.body[f"layer{i}"]                                                     
            no_layers = len(res_layer)
            for j in range(no_layers):
                new_layer = replace_layer_bn(res_layer[j], batch_norm)
                res_layer[j] = new_layer

    return torchvision_rcnn_model

def replace_layer_bn(layer, batch_norm):
    layer_clzz = str(type(layer))
    #mobnet replace logic
    if "ConvBNActivation" in  layer_clzz:
        frozen_bn = layer[1]
        new_bn = replace_single_frozen_bn(frozen_bn, batch_norm)
        layer[1] = new_bn
    elif "InvertedResidual" in layer_clzz:
        no_blocks = len(layer.block)
        for i in range(no_blocks):
            s_block = layer.block[i]
            layer.block[i] = replace_layer_bn(s_block, batch_norm) 
    # resnet replace logic
    elif "Bottleneck" in layer_clzz:
        layer.bn1 = replace_single_frozen_bn(layer.bn1, batch_norm)
        layer.bn2 = replace_single_frozen_bn(layer.bn2, batch_norm)
        layer.bn3 = replace_single_frozen_bn(layer.bn3, batch_norm)
        if layer.downsample:
            for i in range(len(layer.downsample)):
                if "FrozenBatchNorm" in str(type(layer.downsample[i])): 
                    layer.downsample[i] = replace_single_frozen_bn(layer.downsample[i], batch_norm)
    return layer

def replace_single_frozen_bn(frozen_bn_layer, batch_norm):
    num_features = len(frozen_bn_layer.weight)
    eps = frozen_bn_layer.eps
    new_bn = nn.BatchNorm2d(num_features=num_features,
                            eps=batch_norm.get("eps", eps),
                            momentum=batch_norm.get("momentum", 0.1),
                            affine=batch_norm.get("affine", True))
    new_bn.weight = nn.Parameter(frozen_bn_layer.weight)
    new_bn.bias = nn.Parameter(frozen_bn_layer.bias)
    new_bn.running_mean = frozen_bn_layer.running_mean
    new_bn.running_var = frozen_bn_layer.running_var
    return new_bn
