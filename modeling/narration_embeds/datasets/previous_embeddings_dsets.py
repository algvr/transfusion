import json
import torch
from sentence_transformers import SentenceTransformer as ST
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from modeling.narration_embeds.datasets.narration_embeddings import NarrEmbedBase, SBERT_ENCODE_BS


class PrevNarrEmbedDataset(Dataset, NarrEmbedBase):
    """Returns precomputed float embeds for each unique narration in the dataset."""

    def __init__(
        self, to_wrap_dset, embed_args, all_annots, sort_by="video_id", stop_by="video_id", device="cpu"
    ) -> None:
        Dataset.__init__(self)
        NarrEmbedBase.__init__(
            self, to_wrap_dset, embed_args, all_annots, sort_by=sort_by, stop_by=stop_by, device=device
        )
        self.no_prev = int(embed_args["strategy"].split("_")[-1])
        self.set_narration_embeds()

    def get_nars_for_idx(self, idx):
        annot = self.to_wrap_dset.get_annot_by_idx(idx)
        # we check this s.t. we won't grab a narration from a different activity
        stop_by_id = annot[self.stop_by]
        episode_id = annot["nao_clip_id"]
        pos = self.all_annots.index.get_loc(episode_id)

        narrations = []
        for idx in range(self.no_prev):
            item = self.all_annots.iloc[pos - idx - 1]
            if item[self.stop_by] != stop_by_id:
                break
            narrations.append(item["narration"])

        if not narrations:
            narrations.append(self.empty_prev_nar_token)

        return narrations

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        narrations = self.get_nars_for_idx(idx)
        embeds = []
        for narr in narrations:
            embeds.append(self.narration_embeds[narr])

        return {**example, "language_f": torch.Tensor(self.text_pooling(embeds)).type(torch.float32)}


class PrevNarrSbertDataset(Dataset, NarrEmbedBase):
    def __init__(
        self, to_wrap_dset, embed_args, all_annots, sort_by="video_id", stop_by="video_id", device=None
    ) -> None:
        Dataset.__init__(self)
        NarrEmbedBase.__init__(
            self, to_wrap_dset, embed_args, all_annots, sort_by=sort_by, stop_by=stop_by, device=device
        )
        self.no_prev = int(embed_args["strategy"].split("_")[-1])
        self.sbert_v = "all-MiniLM-L12-v2"
        self.setup_narrs_table()

    def setup_narrs_table(self):
        self.lookup = {}
        encoder = ST(self.sbert_v, device="cpu" if self.device is None else f"cuda:{self.device}")

        batch_narrs = []
        batch_episodes = []
        for _, annot in self.to_wrap_dset.get_nao_annots()[[self.stop_by, "nao_clip_id"]].iterrows():
            stop_by_id = annot[self.stop_by]
            episode_id = annot["nao_clip_id"]
            pos = self.all_annots.index.get_loc(episode_id)

            narrs = []
            for idx in range(self.no_prev):
                item = self.all_annots.iloc[pos - idx - 1]
                if item[self.stop_by] != stop_by_id:
                    break
                narrs.insert(0, item["narration"])

            narrs = ", ".join(narrs)
            batch_narrs.append(narrs)
            batch_episodes.append(episode_id)

        with torch.no_grad():
            embeds = encoder.encode(batch_narrs, batch_size=SBERT_ENCODE_BS)
        for i, episode_id in enumerate(batch_episodes):
            self.lookup[episode_id] = torch.Tensor(embeds[i]).type(torch.float32)

        del encoder

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        annot = self.to_wrap_dset.get_annot_by_idx(idx)
        episode_id = annot["nao_clip_id"]
        return {**example, "language_f": self.lookup[episode_id]}


class PrevNarrWordDataset(Dataset, NarrEmbedBase):
    """Used for Encoder that have tokenizers included and hence operate on strings"""

    def __init__(
        self, to_wrap_dset, embed_args, all_annots, uniq_narrations, sort_by="video_id", stop_by="video_id", device=None
    ) -> None:
        Dataset.__init__(self)
        self.empty_prompt = embed_args.get("empty_prompt", None)
        self.end_prompt = embed_args.get("end_prompt", None)
        self.start_prompt = embed_args.get("start_prompt", None)
        self.final_concat = embed_args.get("final_concat", None)
        NarrEmbedBase.__init__(
            self, to_wrap_dset, embed_args, all_annots, uniq_narrations, sort_by=sort_by, stop_by=stop_by, device=device
        )
        self.no_prev = int(embed_args["strategy"].split("_")[-1])
        self.setup_narrs_table()

    def setup_narrs_table(self):
        self.lookup = {}

        for _, annot in tqdm(self.to_wrap_dset.get_nao_annots().iterrows(), total=len(self.to_wrap_dset.get_nao_annots())):
            stop_by_id = annot[self.stop_by]
            episode_id = annot["nao_clip_id"]

            pos = self.all_annots.index.get_loc(episode_id)
            narrs = []

            if self.to_wrap_dset.source not in {"ego4d", "ego4djpg", "ego4djpgv2"}:
                for idx in range(self.no_prev):
                    item = self.all_annots.iloc[pos - idx - 1]
                    if item[self.stop_by] != stop_by_id:
                        break
                    narrs.insert(0, item["narration"])

            else:
                # needed because ego4d is split at smaller clips level, not full movie like egtea/ek
                episode_action_id = annot["episode_action_id"]
                idx = 0
                item = self.all_annots.iloc[pos - idx - 1]

                # while I am in the same video
                while len(narrs) < self.no_prev and item[self.stop_by] == stop_by_id:

                    # add one label per previous segment
                    while item["episode_action_id"] == episode_action_id:
                        idx += 1
                        item = self.all_annots.iloc[pos - idx - 1]

                    if item[self.stop_by] == stop_by_id:
                        narrs.insert(0, item["narration"])
                        episode_action_id = item["episode_action_id"]

            narrs = ", ".join(narrs)
            if self.final_concat:
                narrs = self.final_concat.join(narrs.rsplit(",", 1))

            if self.start_prompt:
                narrs = self.start_prompt + narrs

            if self.end_prompt:
                narrs += self.end_prompt

            if len(narrs) == 0 and self.empty_prompt:
                self.lookup[episode_id] = self.empty_prompt
            else:
                self.lookup[episode_id] = narrs

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        annot = self.to_wrap_dset.get_annot_by_idx(idx)
        episode_id = annot["nao_clip_id"]
        if "language_f" not in example:
            example["language_f"] = self.lookup[episode_id]
        
        return example
