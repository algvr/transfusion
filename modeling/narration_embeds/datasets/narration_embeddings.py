import copy
import logging
import numpy as np
import os


DEF_EMBED_ARGS = {
    "strategy": "prev_5",
    "pooling": "max",
    "text_pooling": "self_attention",
    "type": "glove",
    "size": 300,
}
SBERT_ENCODE_BS = 128


def get_glove_embeds(embed_args):
    embed_size = embed_args["size"]
    "For all narrations of an annotation df compute the"
    embed_path = os.path.expandvars(f"$DATA/glove.6B.{embed_size}d.txt")
    embed_dict = {}

    # first read txt
    with open(embed_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            space_pos = line.index(" ")
            word, vec = line[:space_pos], line[space_pos:]
            vec = np.fromstring(vec, sep=" ")
            if embed_args["normalize"]:
                embed_dict[word] = vec / np.sqrt(vec.dot(vec))
            else:
                embed_dict[word] = vec

    embed_dict["courgette"] = embed_dict["zucchini"]
    embed_dict["airer"] = embed_dict["hanger"]
    embed_dict["let-go"] = embed_dict["drop"]
    embed_dict["turn-down"] = embed_dict["reduce"]
    embed_dict["fishcakes"] = embed_dict["nugget"]
    embed_dict["inspect/read"] = embed_dict["read"]
    embed_dict["divide/pull"] = embed_dict["pull"]
    embed_dict["clean/wipe"] = embed_dict["clean"]
    embed_dict["indument"] = embed_dict["cloth"]

    return embed_dict


def apply_narration_embeds_pooling(narration, narration_embeds, embed_dict, pooling_type):
    if narration not in narration_embeds:
        narr_f = []

        for w in narration.replace(",", " ").split(" "):
            if w == "":
                continue
            w_embed = embed_dict.get(w, None)
            if w_embed is None:
                logging.warn(f"{w=} does not have embed")
            else:
                narr_f.append(w_embed)

        narr_f = np.array(narr_f)
        if pooling_type == "max":
            narr_f = narr_f.max(axis=0)
        elif pooling_type == "mean":
            narr_f = narr_f.mean(axis=0)
        else:
            raise ValueError(f"{pooling_type=} not implemented")

        narration_embeds[narration] = narr_f

    return narration_embeds


def get_embed_dict(embed_type):
    logging.warning(f"Loading {embed_type=} values.")
    if embed_type == "glove":
        return get_glove_embeds
    else:
        raise ValueError(f"{embed_type=} not implemented")


def get_narration_pooling(embed_args):
    if embed_args["pooling"]:
        return apply_narration_embeds_pooling
    else:
        raise ValueError(f'{embed_args["pooling"]=} not implemented')


def get_text_pooling(text_pooling_method, args):
    """Gives method used to pool together narration embeddings from"""
    if text_pooling_method in {"max", "mean"}:
        fun = getattr(np, text_pooling_method)

        def apply_pooling(x):
            return fun(np.array(x), axis=0)

        return apply_pooling
    else:
        return np.array


class NarrEmbedBase:
    def __init__(
        self,
        to_wrap_dset,
        embed_args,
        all_annots,
        uniq_narrations,
        sort_by="video_id",
        stop_by="video_id",
        device="cpu",
    ) -> None:
        self.to_wrap_dset = to_wrap_dset
        self.embed_args = embed_args
        self.nao_annots = copy.deepcopy(to_wrap_dset.get_nao_annots())
        self.sort_by = sort_by
        self.stop_by = stop_by
        self.all_annots = (
            copy.deepcopy(all_annots).reset_index().set_index("episode_id").sort_values([sort_by, "start_frame"])
        )
        self.device = device
        self.uniq_narrations = uniq_narrations
        self.text_pooling = get_text_pooling(embed_args.get("text_pooling", None), embed_args)
        self.set_narr_to_idx_mapping(0)
        self.empty_prev_nar_token = "no_narr"
        self.narr_to_idx[self.empty_prev_nar_token] = len(self.narr_to_idx)
        self.readers = self.to_wrap_dset.readers

    def set_narr_to_idx_mapping(self, idx_offset):
        self.narr_to_idx = {k: i + idx_offset for i, k in enumerate(sorted(self.uniq_narrations))}

    def set_narration_embeds(self):
        self.embed_dict = get_embed_dict(self.embed_args["type"])(self.embed_args)
        self.narration_embeds = {}
        narration_pooling_fn = get_narration_pooling(self.embed_args)
        for narration in self.uniq_narrations:
            self.narration_embeds = narration_pooling_fn(
                narration, self.narration_embeds, self.embed_dict, self.embed_args["pooling"]
            )
        self.narration_embeds[self.empty_prev_nar_token] = np.random.standard_normal(self.embed_args["size"])

    def __len__(self):
        return len(self.to_wrap_dset)

    def get_nao_annots(self):
        return self.nao_annots

    def get_raw_item(self, idx):
        return self.to_wrap_dset.get_raw_item(idx)

    def get_ignore_labels(self, col):
        return self.to_wrap_dset.get_ignore_labels(col)

    def get_label_cutoff(self):
        return self.to_wrap_dset.get_label_cutoff()

    def drop_cutoff_classes(self):
        return self.to_wrap_dset.drop_cutoff_classes()

    def get_verb_mapping(self):
        return self.to_wrap_dset.get_verb_mapping()

    def get_noun_mapping(self):
        return self.to_wrap_dset.get_noun_mapping()

    def set_input_transforms(self, transforms):
        self.to_wrap_dset.set_input_transforms(transforms)

    def set_hmap_transforms(self, hmap_transforms):
        self.to_wrap_dset.set_hmap_transforms(hmap_transforms)

    def set_resize_fn(self, resize_fn):
        self.to_wrap_dset.set_resize_fn(resize_fn)

    def get_narration_embeds(self):
        nar_embeds = np.array([self.narration_embeds[k] for k in sorted(self.narration_embeds.keys())])
        nar_embeds = np.append(nar_embeds, np.zeros((1, nar_embeds.shape[-1])), axis=0)
        return nar_embeds

    def get_no_nouns(self):
        return self.to_wrap_dset.get_no_nouns()

    def get_no_verbs(self):
        return self.to_wrap_dset.get_no_verbs()

    def get_classes(self, clzz):
        return self.to_wrap_dset.get_classes(clzz)

    def set_resize_fn(self, resize_fn):
        self.to_wrap_dset.set_resize_fn(resize_fn)

    def set_input_transforms(self, transforms):
        self.to_wrap_dset.set_input_transforms(transforms)

    def set_localiz_transforms(self, transforms):
        self.to_wrap_dset.set_localiz_transforms(transforms)

    def set_flow_transform(self, transform):
        self.to_wrap_dset.set_flow_transform(transform)

    def get_samples(self, no_samples, seed, collate_fn=None):
        return self.to_wrap_dset.get_samples(no_samples, seed, collate_fn)
