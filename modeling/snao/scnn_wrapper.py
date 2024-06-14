import numpy as np
from modeling.fast_scnn.models.fast_scnn import ScnnHeatmapPred, LinearBottleneck, _make_layer, get_fast_scnn
from modeling.commons import NaoWrapperBase, freeze_all_but_bn
from torch import nn


class SCNNWrapper(NaoWrapperBase):
    def __init__(
        self,
        to_wrap,
        run_config,
        noun_classes,
        verb_classes,
        finetune=False,
        head_k_size=3,
        hmap_head_upscale=1,
        out_class_channels=512,
        bottleneck_layer=LinearBottleneck,
    ):
        super().__init__(
            to_wrap, noun_classes, verb_classes, run_config["criterion"], run_config["w_sigmoid"], finetune, run_config
        )
        self.vis_input_key = "image"
        self.finetune = finetune
        self.in_head_channels = to_wrap.get_features_out_channels()
        self.out_class_channels = out_class_channels
        self.activ_f = self.to_wrap.activ_f
        if self.finetune:
            self.to_wrap.apply(freeze_all_but_bn)

        self.heatmap_head = ScnnHeatmapPred(
            self.in_head_channels, 1, hmap_head_upscale, head_k_size, dropout=self.run_config["hmap_dropout"]
        )

        if self.is_classifying():
            self.bottleneck_1 = _make_layer(
                bottleneck_layer, self.in_head_channels, int(self.in_head_channels * 1.25), blocks=3, t=4, stride=2
            )
            self.bottleneck_2 = _make_layer(
                bottleneck_layer,
                int(self.in_head_channels * 1.25),
                out_class_channels,
                blocks=1,
                t=4,
                stride=1,
            )
            self.setup_classifiers(out_class_channels)
            self.adpt_pooling = nn.AdaptiveAvgPool2d(1)

        del self.to_wrap.classifier

    def get_heatmap_channels(self):
        return self.in_head_channels

    def get_final_dsampled_size(self):
        return np.prod(np.ceil(np.array(self.run_config["resize_spec"]) / 8)).astype(int)

    def get_final_dsampled_shape(self):
        return (np.ceil(np.array(self.run_config["resize_spec"]) / 8)).astype(int)

    def get_features_out_channels(self):
        return self.in_head_channels

    def get_features_out_channels(self):
        return self.to_wrap.get_features_out_channels()

    def forward_features(self, x):
        global_features, high_res_features = self.to_wrap.forward_features(x)
        features = self.fuse_branches(high_res_features, global_features)
        return features

    def fuse_branches(self, higher_res, lower_res):
        x = self.to_wrap.feature_fusion(higher_res, lower_res)
        return x

    def classif_branch(self, features):
        features = self.bottleneck_1(features)
        features = self.activ_f(features)
        features = self.bottleneck_2(features)
        features = self.activ_f(features)
        features = self.adpt_pooling(features)
        return features.flatten(1)


def get_test_scnn_wraper():
    run_config = {
        "pretrained": True,
        "finetune": False,
        "criterion": {"noun": 0, "verb": 0, "mae": 1},
        "w_sigmoid": False,
        "class_dropout": 0.25,
        "heatmap_type": "gaussian",
        "resize_spec": [192, 384],
        "hmap_dropout": 0.25,
    }
    model_config = {
        "pretrained": False,
        "finetune": False,
        "downsample_c": [24, 32, 48],
        "feature_extractor_c": [48, 64, 96],
        "num_blocks": [3, 3, 3],
        "feature_fusion_c": [48, 96, 96],
        "activ_f": "relu",
    }
    class_sizes = {"noun": 1, "verb": 1}

    fast_scnn = get_fast_scnn()
    return SCNNWrapper(fast_scnn, run_config, 1, 1)
