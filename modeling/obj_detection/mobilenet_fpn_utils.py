# taken from torchvision.detection.models and utils to allow multiple fpn layers for mobilenet
import os
import sys
sys.path.append(os.path.dirname(__file__))
from model_urls import model_urls

from torch.hub import *
from torchvision.models.detection.backbone_utils import *
from torchvision.models.detection.faster_rcnn import *
from torchvision.models.detection.anchor_utils import *
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection._utils import overwrite_eps


def fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    returned_layers=None,
    **kwargs
):
    """
    Constructs a low resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone tunned for mobile use-cases.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See `fasterrcnn_resnet50_fpn` for more details.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
    """
    weights_name = "fasterrcnn_mobilenet_v3_large_320_fpn_coco"
    defaults = {
        "min_size": 320,
        "max_size": 640,
        "rpn_pre_nms_top_n_test": 150,
        "rpn_post_nms_top_n_test": 150,
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return large_mobnet_fpn_adapter(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
        **kwargs,
    )


def fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    returned_layers=None,
    **kwargs
):
    """
    Constructs a high resolution Faster R-CNN model with a MobileNetV3-Large FPN backbone.
    It works similarly to Faster R-CNN with ResNet-50 FPN backbone. See `fasterrcnn_resnet50_fpn` for more details.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 6, with 6 meaning all backbone layers are trainable.
    """
    weights_name = "fasterrcnn_mobilenet_v3_large_fpn_coco"
    defaults = {
        "rpn_score_thresh": 0.05,
    }

    kwargs = {**defaults, **kwargs}
    return large_mobnet_fpn_adapter(
        weights_name,
        pretrained=pretrained,
        progress=progress,
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
        **kwargs,
    )


def large_mobnet_fpn_adapter(
    weights_name,
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    returned_layers=None,
    **kwargs
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 3
    )

    if pretrained:
        pretrained_backbone = False
    backbone = mobilenet_backbone(
        "mobilenet_v3_large",
        pretrained_backbone,
        True,
        trainable_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
    )

    mul_factor = 3 if (returned_layers == None or returned_layers == [4, 5]) else (len(returned_layers) + 1)
    anchor_sizes = (
        (
            32,
            64,
            128,
            256,
            512,
        ),
    ) * mul_factor
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    model = FasterRCNN(
        backbone, num_classes, rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios), **kwargs
    )
    if pretrained:
        if model_urls.get(weights_name, None) is None:
            raise ValueError("No checkpoint is available for model {}".format(weights_name))
        state_dict = load_state_dict_from_url(model_urls[weights_name], progress=progress)
        if returned_layers != None or returned_layers != [4, 5]:
            to_pop = [k for k in state_dict.keys() if "fpn" in k]
            for k in to_pop:
                state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    return model


def fasterrcnn_resnet50_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    returned_layers=None,
    fpn_out_channels=256,
    **kwargs
):
    """
    Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        >>> # For training
        >>> images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
        >>> labels = torch.randint(1, 91, (4, 11))
        >>> images = list(image for image in images)
        >>> targets = []
        >>> for i in range(len(images)):
        >>>     d = {}
        >>>     d['boxes'] = boxes[i]
        >>>     d['labels'] = labels[i]
        >>>     targets.append(d)
        >>> output = model(images, targets)
        >>> # For inference
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    if returned_layers is not None and returned_layers != [1,2,3,4]:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        dif = 4-len(returned_layers)
        anchor_sizes = anchor_sizes[dif:]
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
    else:
        rpn_anchor_generator = None
    backbone = resnet_fpn_backbone("resnet50", 
                                   pretrained_backbone, 
                                   trainable_layers=trainable_backbone_layers, 
                                   returned_layers=returned_layers,
                                   # out_channels=fpn_out_channels
                                   )
    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=rpn_anchor_generator,
                       **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model


if __name__ == "__main__":
    import torchvision

    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn = fasterrcnn_mobilenet_v3_large_320_fpn
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn = fasterrcnn_mobilenet_v3_large_fpn

    backbone = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        pretrained=True, returned_layers=[3, 4, 5]
    )

    print(backbone)
