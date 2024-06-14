from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import Transform
import numbers
import numpy as np
from pytorchvideo.transforms.transforms import Div255, RandomResizedCrop
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo, RandomHorizontalFlipVideo


IMNET_MEAN = np.array([0.485, 0.456, 0.406])
IMNET_STD = np.array([0.229, 0.224, 0.225])

OWN_MEAN = np.array([0.4223, 0.3524, 0.2924])
OWN_STD = np.array([0.2435, 0.2265, 0.2260])

X3D_MEAN = np.array([0.45, 0.45, 0.45])
X3D_STD = np.array([0.225, 0.225, 0.225])

EGO4D_MEAN = np.array([0.4054, 0.3713, 0.3364])
EGO4D_STD = np.array([0.2400, 0.2237, 0.2219])

EGO4D_BASELINE_MEAN = np.array([103.53, 116.28, 123.675]) / 255.
EGO4D_BASELINE_STD = np.array([1., 1., 1.]) / 255.  # go back to 255-scale

EGO4D_EK_MEAN = np.array([0.4059, 0.3627, 0.3229])
EGO4D_EK_STD = np.array([0.2386, 0.2267, 0.2256])

FLOW_MEAN = 0.1494
FLOW_STD = 4.9383

FLOW_MEAN_JPG = [0.4056, 0.3716, 0.3367]
FLOW_STD_JPG = [0.2400, 0.2237, 0.2218]

FLOW_MEAN_JPG_1 = 0.38
FLOW_STD_JPG_1 = 0.225

FLOW_CLIP_MEAN = 0.1549
FLOW_CLIP_STD = 4.3792

FLOW_STATS = {
    "fp": {"mean": FLOW_MEAN, "std": FLOW_STD},
    "jpg": {"mean": FLOW_MEAN_JPG, "std": FLOW_STD_JPG},
    "jpg_1": {"mean": FLOW_MEAN_JPG_1, "std": FLOW_STD_JPG_1},
}

NO_OP_AUG_DICT = {
    "crop_spec": [1.0, 1.0],
    "aspect_ratio": [2, 2],
    "brightness": 0,
    "hue": 0,
    "saturation": 0,
    "contrast": 0,
}


NORM_DICT = {
    "imagenet": (IMNET_MEAN, IMNET_STD),
    "ego4d_baseline": (EGO4D_BASELINE_MEAN, EGO4D_BASELINE_STD),
    "own": {
        "all": (OWN_MEAN, OWN_STD),
        "ego4d": (EGO4D_MEAN, EGO4D_STD),
    },
    "flow": (FLOW_MEAN, FLOW_STD),
    "x3d": (X3D_MEAN, X3D_STD),
}


def get_norm_mean_std(normalization_type, dataset):
    if normalization_type == "own":
        mean, std = NORM_DICT["own"][dataset]
    else:
        mean, std = NORM_DICT[normalization_type]
    return mean, std


class ChannelPermutationTransform(Transform):
    """
    Permute the channels in the input image.
    """

    def __init__(self, original_order="RGB", new_order="RGB"):
        super().__init__()
        self.original_order = original_order.upper().strip()
        self.new_order = new_order.upper().strip()
        self.do_transform = self.original_order != self.new_order
        self.permut_order = [original_order.index(char) for char in new_order]
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if not self.do_transform:
            return img
        if img.shape[-1] != 3:
            return img
        return img[..., self.permut_order]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def inverse(self) -> Transform:
        return ChannelPermutationTransform(self.new_order, self.original_order)


class ChannelPermutationAugmentation(Augmentation):
    """
    Permute the channels in the input image.
    """

    def __init__(self, original_order="RGB", new_order="RGB"):
        super().__init__()
        self.original_order = original_order.upper().strip()
        self.new_order = new_order.upper().strip()
        self.do_transform = self.original_order != self.new_order
        self.permut_order = [original_order.index(char) for char in new_order]
        self._init(locals())

    def get_transform(self, image):
        return ChannelPermutationTransform(self.original_order, self.new_order)


def set_transforms_to_dsets(train_dset, eval_dsets, run_cfg, aug_dict, experiment="snao"):
    resize_fn = get_resize_transforms(experiment)
    localiz_transforms = get_localiz_transforms(run_cfg.get("hmap_scaling", 1), experiment)
    input_transforms = get_input_transforms(experiment)

    mean, std = get_norm_mean_std(run_cfg["normalization"], run_cfg["dataset"])

    train_dset.set_input_transforms(input_transforms(mean, std, aug_dict))
    train_dset.set_localiz_transforms(localiz_transforms)
    train_dset.set_resize_fn(resize_fn(aug_dict))

    flow_transforms = get_flow_transforms(
        run_cfg["flow_args"]["norm"],
        run_cfg["flow_args"]["clip"],
        run_cfg["flow_args"]["as_jpg"],
        run_cfg["flow_args"]["concat_magnitude"],
    )

    train_dset.set_flow_transform(flow_transforms)

    eval_aug_dict = NO_OP_AUG_DICT.copy()
    eval_aug_dict["resize_spec"] = aug_dict["resize_spec"]
    eval_aug_dict["channel_order"] = aug_dict["channel_order"]

    for dset in eval_dsets:
        dset.set_input_transforms(input_transforms(mean, std, eval_aug_dict))
        dset.set_localiz_transforms(localiz_transforms)
        dset.set_resize_fn(resize_fn(eval_aug_dict))
        dset.set_flow_transform(flow_transforms)


def get_resize_transforms(experiment):
    if experiment == "egonao":
        return get_snao_resize_transforms
    else:
        raise ValueError(f"{experiment=} was not recognized")


def get_snao_resize_transforms(augmentations_dict=None):
    """Return a detectron list of transforms instance used to obtain"""
    listy = []

    if not augmentations_dict:
        listy = [T.NoOpTransform()]

    else:
        if "crop_spec" in augmentations_dict and augmentations_dict["crop_spec"] != [1, 1]:
            listy.append(
                T.RandomCrop("relative_range", augmentations_dict["crop_spec"]),
            )

        if augmentations_dict["resize_spec"][-1] != 0 and not isinstance(augmentations_dict["resize_spec"][-1], list):
            listy.append(T.Resize(augmentations_dict["resize_spec"]))

        if augmentations_dict.get("flip", False):
            listy.append(T.RandomFlip(0.5))


    augs = T.AugmentationList(listy)

    # cannot use a transform directly, or a Pickle-related error will be raised
    if augmentations_dict and augmentations_dict.get("channel_order", "RGB") != "RGB":
        augs.augs.append(ChannelPermutationAugmentation(original_order="RGB",
                                                        new_order=augmentations_dict["channel_order"]))
    
    return augs


def get_denormalize(mean, std):
    denormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return denormalize


def get_inv_hmap(hmap_scaling):
    if isinstance(hmap_scaling, numbers.Number):
        if hmap_scaling != 1:
            return transforms.Lambda(lambda x: x / hmap_scaling)
        else:
            return lambda x: x
    else:
        return lambda x: x


def get_input_transforms(experiment):
    if experiment == "egonao":
        return get_snao_input_transforms
    else:
        raise ValueError(f"{experiment=} was not recognized")


def get_vsnao_input_transforms(mean, std, augmentations_dict):
    transform = transforms.Compose(
        [
            Div255(),
            # 182 gives 242 width
            NormalizeVideo(mean, std),
            # can also add color augsssss
        ]
    )

    return transform


cst_1_255 = 1 / 255.0


def div255(x):
    return x * cst_1_255


cst_1_122_5 = 1 / 122.5


def from_0_255_1_1(x):
    return (x - 122.5) * cst_1_122_5


def get_flow_transforms(norm=True, clip=True, as_jpg=True, concat_magnitude=False):
    if norm == "-1_1":
        transform = from_0_255_1_1

    elif as_jpg:
        if norm:
            if concat_magnitude:
                transform = transforms.Compose([div255, transforms.Normalize(FLOW_MEAN_JPG, FLOW_STD_JPG)])
            else:
                transform = transforms.Compose([div255, transforms.Normalize(FLOW_MEAN_JPG_1, FLOW_STD_JPG_1)])

        else:
            transform = div255

    else:
        if norm:
            if clip:
                transform = transforms.Normalize(FLOW_CLIP_MEAN, FLOW_CLIP_STD)
            else:
                transform = transforms.Normalize(FLOW_MEAN, FLOW_STD)
        else:
            transform = lambda x: x

    return transform


def get_snao_input_transforms(mean, std, augmentations_dict):
    my_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ColorJitter(
                augmentations_dict["brightness"],
                augmentations_dict["contrast"],
                augmentations_dict["saturation"],
                augmentations_dict["hue"],
            ),
            transforms.Normalize(mean, std),
        ]
    )
    return my_transforms


def get_localiz_transforms(hmap_scaling, experiment):
    if experiment in {"egonao", "ego_nao"}:
        listy = [transforms.ToTensor()]
    else:
        listy = []

    if isinstance(hmap_scaling, numbers.Number):
        if hmap_scaling != 1:
            listy.append(transforms.Lambda(lambda x: x * hmap_scaling))
    elif hmap_scaling == "sum":
        listy.append(transforms.Lambda(lambda x: x / x.sum()))
    else:
        raise ValueError(f"{hmap_scaling=} not recognised")

    my_transforms = transforms.Compose(listy)

    return my_transforms
