from argparse import ArgumentParser
import os
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from runner.nao.snao_data import get_snao_datasets, get_train_test_split
from runner.utils.data_transforms import get_snao_resize_transforms


class OnlineMeanStd:
    def __init__(self):
        pass

    def __call__(self, dataset, batch_size, method="strong", flow=False, flow_as_jpg=True):
        """
        Calculate mean and std of a dataset in lazy mode (online)
        On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.

        :param dataset: Dataset object corresponding to your dataset
        :param batch_size: higher size, more accurate approximation
        :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
        :return: A tuple of (mean, std) with size of (3,)
        """
        if not flow or flow_as_jpg:
            fst_moment = torch.empty(3)
            snd_moment = torch.empty(3)
            key = "image"

        else:
            fst_moment = torch.empty(3)
            snd_moment = torch.empty(3)
            key = "flow_data"

        if method == "weak":
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=0)
            mean = 0.0
            std = 0.0
            nb_samples = 0.0
            for data in loader:
                data = data[key]
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples

            mean /= nb_samples
            std /= nb_samples

            return mean, std

        elif method == "strong":
            loader = DataLoader(
                dataset=dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=0, prefetch_factor=16
            )
            cnt = 0

            for data in tqdm(loader, total=len(dataset)):
                data = data[key]
                b, c, h, w = data.shape
                nb_pixels = b * h * w
                sum_ = torch.sum(data, dim=[0, 2, 3])
                sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
                fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                cnt += nb_pixels

            return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_flow", default=True, action="store_true")
    args = parser.parse_args()
    use_flow = args.use_flow

    config = {
        "split": {
            "type": "stratified",
            "subset": 0,
            "version": 42,
            "strat_col": None,
            "egtea_test": False,
        },
        "debug": False,
        "dataset": {
            "args": {
                "label_merging": os.path.expandvars("$CODE/data_preprocessing/configs/label_merging.json"),
                "offset_s": 0.4,
                "label_cutoff": {"verb": 0, "noun": 0, "drop": False},
                "nao_version": 1,
                "coarse": False,
                "take_double": False,
            },
            "name": "ego4d",
            "subsample": None,
        },
        "run": {
            "seed": 42,
            "heatmap_type": "gaussian",
            "narration_embeds": {"use": False},
            "flow_args": {
                "use": use_flow,
                "num_frames": 1,
                "stride": 1,
                "block": 1,
                "clip": 0,
                "as_jpg": True,
                "concat_magnitude": True,
            },
        },
    }

    if config["run"]["flow_args"]["as_jpg"]:
        config["run"]["flow_args"]["clip"] = 0

    raw_dataset = get_snao_datasets(config)
    train, val_dataset, test_dataset, train_test_split = get_train_test_split(raw_dataset, config)
    resize_fn = get_snao_resize_transforms(augmentations_dict={"resize_spec": [240, 360]})
    train.set_resize_fn(resize_fn)
    train.set_input_transforms(torchvision.transforms.ToTensor())
    train.set_localiz_transforms(torchvision.transforms.ToTensor())
    mean, var = OnlineMeanStd()(train, 128, method="strong", flow=use_flow)

    print(mean, var)
