import numpy as np
import torch
from torchvision import transforms

from data_preprocessing.datasets.snao_datasets import get_test_snao

from modeling.narration_embeds.datasets.all_embeddings_dsets import AllNarrEmbedWrapper, AllNarrSbertWrapper
from modeling.narration_embeds.datasets.previous_embeddings_dsets import (
    PrevNarrEmbedDataset,
    PrevNarrSbertDataset,
    PrevNarrWordDataset,
)
from modeling.narration_embeds.datasets.current_embeddings_dsets import (
    CurrNarrEmbedDataset,
    CurrNarrSbertDataset,
    CurrNarrWordDataset,
)
from modeling.narration_embeds.datasets.narration_embeddings import DEF_EMBED_ARGS
from modeling.narration_embeds.narr_pooling_layers import LEARNABLE_LM


def get_narr_wrapper(embed_args):
    if embed_args["strategy"] == "all":
        if embed_args["pooling"] == "sbert":
            return AllNarrSbertWrapper
        elif embed_args["finetune"]:
            return None
        else:
            return AllNarrEmbedWrapper
    elif embed_args["strategy"] == "current":
        if embed_args["pooling"] == "sbert":
            return CurrNarrSbertDataset
        elif embed_args["text_pooling"] in LEARNABLE_LM:
            return CurrNarrWordDataset
        else:
            return CurrNarrEmbedDataset
    elif "prev" in embed_args["strategy"]:
        if embed_args["pooling"] == "sbert":
            return PrevNarrSbertDataset
        elif embed_args["text_pooling"] in LEARNABLE_LM:
            return PrevNarrWordDataset
        else:
            return PrevNarrEmbedDataset
    else:
        raise ValueError(f"{embed_args['strategy']} not implemented")


def pad_language_f(batch_samples):
    """masks and pads batch samples to have equal length to be used in sequence models"""
    lens = np.array([len(sample["language_f"]) for sample in batch_samples])
    f_size = batch_samples[0]["language_f"].size(1)

    max_len = lens.max()
    diffs = max_len - lens

    for i, sample in enumerate(batch_samples):
        mask = np.zeros(max_len, dtype=np.bool)
        if diffs[i]:
            sample["language_f"] = torch.cat(
                (sample["language_f"], -1 * torch.ones((diffs[i], f_size), dtype=sample["language_f"].dtype)), dim=0
            )
            mask[lens[i] :] = 1
        sample["pad_mask"] = torch.Tensor(mask)

    return batch_samples


def collate_embed_fn(batch_samples):
    batch_samples = pad_language_f(batch_samples)
    batch_samples = {k: torch.stack([dic[k] for dic in batch_samples]) for k in batch_samples[0]}
    return batch_samples


def collate_boxes_fn_languagef(batch):
    leny = range(len(batch))
    language_f = [batch[i]["language_f"] for i in leny]
    reg_batch = collate_boxes_fn(batch)
    reg_batch["language_f"] = language_f
    if "hand_boxes" in batch[0]:
        reg_batch["hand_boxes"] = torch.stack([batch[i]["hand_boxes"] for i in leny])
    if "hand_poses" in batch[0]:
        reg_batch["hand_poses"] = torch.stack([batch[i]["hand_poses"] for i in leny])
    return reg_batch


def collate_idx_fn(batch_samples, pad_idx=-1):
    lens = np.array([len(sample["language_f"]) for sample in batch_samples])

    max_len = lens.max()
    diffs = max_len - lens

    for i, sample in enumerate(batch_samples):
        mask = np.zeros(max_len, dtype=np.bool)
        if diffs[i]:
            sample["language_f"] = torch.cat(
                (sample["language_f"], pad_idx * torch.ones(diffs[i], dtype=sample["language_f"].dtype)), dim=0
            )
            mask[lens[i] :] = 1
        sample["pad_mask"] = torch.Tensor(mask)

    batch_samples = {k: torch.stack([dic[k] for dic in batch_samples]) for k in batch_samples[0]}
    return batch_samples


def collate_boxes_fn(batch):
    leny = range(len(batch))

    images = [batch[i]["image"] for i in leny]
    visual_features = None
    flow_data = None
    hand_pos_data = None

    if "flow_data" in batch[0]:
        flow_data = torch.stack([batch[i]["flow_data"] for i in leny])
    if "hand_pos" in batch[0]:
        hand_pos_data = torch.stack([batch[i]["hand_pos"] for i in leny])
    if "visual_features" in batch[0]:
        visual_features = torch.stack([batch[i]["visual_features"] for i in leny])

    targets = []
    for i in leny:
        entry = {}
        entry["boxes"] = batch[i]["bboxes"]
        entry["labels"] = batch[i]["labels"]
        entry["verbs"] = batch[i]["verb"]
        entry["ttcs"] = batch[i]["ttc"]
        entry["id"] = batch[i]["id"]
        entry["orig_shape"] = batch[i]["orig_shape"]
        targets.append(entry)

    return {"image": images, 
            "targets": targets,
            "flow_data": flow_data, 
            "hand_pos": hand_pos_data, 
            "visual_features":visual_features
            }

def get_collate_fn(experiment, narr_embed_config, raw_dataset):
    if experiment != "egonao":
        if narr_embed_config["use"]:
            if narr_embed_config["args"]["text_pooling"] == "self_attention":
                return collate_embed_fn
            else:
                return None
        else:
            return None

    else:
        if narr_embed_config["use"]:
            if narr_embed_config["args"]["text_pooling"] == "self_attention":
                return collate_embed_fn
            else:
                return collate_boxes_fn_languagef
        else:
            return collate_boxes_fn

if __name__ == "__main__":
    _, snao_dataset = get_test_snao()
    snao_dataset.set_hmap_transforms(transforms.ToTensor())
    dset = get_narr_wrapper(DEF_EMBED_ARGS)(snao_dataset, DEF_EMBED_ARGS, snao_dataset.annotations)
    example = dset[0]
    print(example)
