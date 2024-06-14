import numpy as np
import torchvision.models as models
from modeling.commons import NaoWrapperBase, freeze_all_but_bn
from torch import nn


def get_resnet(resnet_type, pretrained):
    fun = getattr(models.resnet, resnet_type)
    obj = fun(pretrained=pretrained)

    obj = adapt_resnet(obj)
    return obj


def adapt_resnet(obj):
    if not hasattr(obj, "blocks"):
        setattr(obj, "blocks", [obj.layer1, obj.layer2, obj.layer3, obj.layer4])

    if not hasattr(obj, "forward_features"):

        def forward_features(x, obj=obj):
            # cannot use "self"
            x = obj.forward_stem(x)
            x = obj.layer1(x)
            x = obj.layer2(x)
            x = obj.layer3(x)
            x = obj.layer4(x)
            return x

        setattr(obj, "forward_features", forward_features)

    if not hasattr(obj, "forward_stem"):

        def forward_stem(x, obj=obj):
            # cannot use "self"
            x = obj.conv1(x)
            x = obj.bn1(x)
            x = obj.relu(x)
            x = obj.maxpool(x)
            return x

        setattr(obj, "forward_stem", forward_stem)

    return obj


def adapt_block(block, fusion_fn):
    if "Basic" in str(type(block)):
        return ResNetBasicBDualWrapper(block, fusion_fn)
    elif "Bottleneck" in str(type(block)):
        return ResNetBneckBDualWrapper(block, fusion_fn)


class ResNetBasicBDualWrapper(nn.Module):
    def __init__(self, basic_block, fusion_fn):
        super().__init__()
        self.basic_block = basic_block
        self.fusion_fn = fusion_fn

    def forward(self, x, extra):
        identity = x

        x = self.fusion_fn(x, extra)

        out = self.basic_block.conv1(x)
        out = self.basic_block.bn1(out)
        out = self.basic_block.relu(out)

        out = self.basic_block.conv2(out)
        out = self.basic_block.bn2(out)

        if self.basic_block.downsample is not None:
            identity = self.basic_block.downsample(x)

        out += identity
        out = self.basic_block.relu(out)

        return out


class ResNetBneckBDualWrapper(nn.Module):
    def __init__(self, bneck_block, fusion_fn):
        super().__init__()
        self.bneck_block = bneck_block
        self.fusion_fn = fusion_fn

    def forward(self, x, extra):
        identity = x

        x = self.fusion_fn(x, extra)

        out = self.bneck_block.conv1(x)
        out = self.bneck_block.bn1(out)
        out = self.bneck_block.relu(out)

        out = self.bneck_block.conv2(out)
        out = self.bneck_block.bn2(out)
        out = self.bneck_block.relu(out)

        out = self.bneck_block.conv3(out)
        out = self.bneck_block.bn3(out)

        if self.bneck_block.downsample is not None:
            identity = self.bneck_block.downsample(x)

        out += identity
        out = self.bneck_block.relu(out)

        return out


class ResnetWrapper(NaoWrapperBase):
    def __init__(
        self,
        to_wrap,
        run_config,
        noun_classes,
        verb_classes,
        heatmap_blocks,
        train_ep,
        finetune=False,
        head_k_size=3,
        hmap_head_upscale=1,
        vis_input_key="image",
    ):
        self.downsample_factor = 2 ** (heatmap_blocks + 1)
        self.heatmap_blocks = heatmap_blocks
        self.no_layers = 4
        self.vis_input_key = vis_input_key
        self.train_ep = train_ep

        super().__init__(
            to_wrap,
            noun_classes,
            verb_classes,
            run_config["criterion"],
            run_config["w_sigmoid"],
            finetune,
            run_config,
        )

        if self.finetune:
            self.to_wrap.apply(freeze_all_but_bn)

        if self.is_classifying() or self.is_ttc_pred_on():
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.setup_classifiers(self.get_features_out_channels())
        else:
            self.cut_blocks_up_to(heatmap_blocks)

        del self.to_wrap.fc
        self.blocks = self.to_wrap.blocks

    def call_model_epoch_triggers(self, epoch):
        if self.train_ep != -1 and epoch >= self.train_ep:
            for param in self.parameters():
                param.requires_grad = True

            print("Unfroze Resnet parameters")

    def get_final_dsampled_size(self):
        return np.prod(np.ceil(np.array(self.run_config["resize_spec"]) // self.downsample_factor)).astype(int)

    def get_final_dsampled_shape(self):
        return (np.array(self.run_config["resize_spec"]) // self.downsample_factor).astype(int)

    def get_features_out_channels(self):
        last_block = self.to_wrap.blocks[self.heatmap_blocks - 1][-1]
        if isinstance(last_block, models.resnet.BasicBlock):
            return last_block.bn2.num_features
        elif isinstance(last_block, models.resnet.Bottleneck):
            return last_block.bn3.num_features
        else:
            raise ValueError

    def forward_stem(self, x):
        return self.to_wrap.forward_stem(x[self.vis_input_key])

    def forward_features(self, x):
        x = self.to_wrap.forward_stem(x)
        for i in range(self.heatmap_blocks):
            x = self.to_wrap.blocks[i](x)
        return x

    def classif_branch(self, features):
        for i in range(self.heatmap_blocks, self.no_layers):
            features = self.to_wrap.blocks[i](features)

        features = self.avg_pool(features)
        return features.flatten(1)

    def cut_blocks_up_to(self, keep_block_id):
        self.to_wrap.blocks = self.to_wrap.blocks[:keep_block_id]


def get_resnet_wrapper(run_config, model_config, class_sizes, flow_args=None):
    resnet_base = get_resnet(resnet_type=model_config["flavor"], pretrained=model_config["pretrained"])
    vis_input_key = "image"
    return ResnetWrapper(
        resnet_base,
        run_config,
        noun_classes=class_sizes["noun"],
        verb_classes=class_sizes["verb"],
        heatmap_blocks=model_config["heatmap_blocks"],
        train_ep=model_config["train_ep"],
        finetune=model_config["finetune"],
        head_k_size=model_config["head_k_size"],
        hmap_head_upscale=model_config["hmap_head_upscale"],
        vis_input_key=vis_input_key,
    )
