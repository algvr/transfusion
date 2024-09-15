from sentence_transformers import SentenceTransformer as ST
import torch
from torch.utils.data.dataset import Dataset

from modeling.narration_embeds.datasets.narration_embeddings import NarrEmbedBase, SBERT_ENCODE_BS


class CurrNarrEmbedDataset(Dataset, NarrEmbedBase):
    """Returns precomputed float embeds for each unique narration in the dataset."""

    def __init__(
        self, to_wrap_dset, embed_args, all_annots, sort_by="video_id", stop_by="video_id", device="cpu"
    ) -> None:
        Dataset.__init__(self)
        NarrEmbedBase.__init__(
            self, to_wrap_dset, embed_args, all_annots, sort_by=sort_by, stop_by=stop_by, device=device
        )
        self.set_narration_embeds()

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        narration = self.all_annots.loc[example["nao_clip_id"]]["narration"]
        embeds = [self.narration_embeds[narration]]
        return {**example, "language_f": torch.Tensor(self.text_pooling(embeds)).type(torch.float32)}


class CurrNarrSbertDataset(Dataset, NarrEmbedBase):
    def __init__(
        self, to_wrap_dset, embed_args, all_annots, sort_by="video_id", stop_by="video_id", device=None
    ) -> None:
        Dataset.__init__(self)
        NarrEmbedBase.__init__(
            self, to_wrap_dset, embed_args, all_annots, sort_by=sort_by, stop_by=stop_by, device=device
        )
        self.sbert_v = "all-MiniLM-L12-v2"
        self.setup_narrs_table()

    def setup_narrs_table(self):
        self.lookup = {}
        encoder = ST(self.sbert_v, device="cpu" if self.device is None else f"cuda:{self.device}")

        batch_narrs = []
        batch_episodes = []
        for _, annot in self.to_wrap_dset.get_nao_annots()[[self.stop_by, "nao_clip_id"]].iterrows():
            narrs = [annot["narration"]]
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


class CurrNarrWordDataset(Dataset, NarrEmbedBase):
    """Used for Encoder that have tokenizers included and hence operate on strings"""

    def __init__(
        self, to_wrap_dset, embed_args, all_annots, uniq_narrations, sort_by="video_id", stop_by="video_id", device=None
    ) -> None:
        Dataset.__init__(self)
        self.empty_prompt = embed_args.get("empty_prompt", None)
        self.end_prompt = embed_args.get("end_prompt", None)
        self.start_prompt = embed_args.get("start_prompt", None)
        NarrEmbedBase.__init__(
            self, to_wrap_dset, embed_args, all_annots, uniq_narrations, sort_by=sort_by, stop_by=stop_by, device=device
        )
        self.setup_narrs_table()

    def setup_narrs_table(self):
        self.lookup = {}

        for _, annot in self.to_wrap_dset.get_nao_annots().iterrows():
            episode_id = annot["nao_clip_id"]

            narrs = annot["narration"]
            
            if self.start_prompt:
                narrs = self.start_prompt + narrs

            if self.end_prompt:
                narrs += self.end_prompt

            if (len(narrs) == 0 or (len(narrs) == 1 and len(narrs[0]) == 0)) and self.empty_prompt:
                self.lookup[episode_id] = self.empty_prompt
            else:
                self.lookup[episode_id] = narrs

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        annot = self.to_wrap_dset.get_annot_by_idx(idx)
        episode_id = annot["nao_clip_id"]
        return {**example, "language_f": self.lookup[episode_id]}
