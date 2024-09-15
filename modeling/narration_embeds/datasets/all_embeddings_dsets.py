import numpy as np
from sentence_transformers import SentenceTransformer as ST
import torch
from torch.utils.data.dataset import Dataset

from modeling.narration_embeds.datasets.narration_embeddings import NarrEmbedBase, SBERT_ENCODE_BS


class AllNarrEmbedWrapper(Dataset, NarrEmbedBase):
    def __init__(self, to_wrap_dset, embed_args, all_annots, device="cpu") -> None:
        Dataset.__init__(self)
        NarrEmbedBase.__init__(self, to_wrap_dset, embed_args, all_annots, device="cpu")

        self.set_video_embeds()

    def set_video_embeds(self):
        self.set_narration_embeds()
        video_ids = self.nao_annots["video_id"].unique()
        self.video_embeds = {}
        for video_id in video_ids:
            all_video_nars = self.all_annots[self.all_annots["video_id"] == video_id]["narration"]
            video_embed = []
            for narr in all_video_nars:
                video_embed.append(self.narration_embeds[narr])
            self.video_embeds[video_id] = self.text_pooling(np.array(video_embed), axis=0)

    def __getitem__(self, idx):
        example = self.to_wrap_dset[idx]
        annot = self.to_wrap_dset.get_annot_by_idx(idx)
        video_id = annot["video_id"]
        return {**example, "language_f": torch.Tensor(self.video_embeds[video_id]).type(torch.float32)}


class AllNarrSbertWrapper(AllNarrEmbedWrapper):
    def __init__(self, to_wrap_dset, embed_args, all_annots, device="cpu") -> None:
        AllNarrEmbedWrapper.__init__(self, to_wrap_dset, embed_args, all_annots, device)

    def set_video_embeds(self):
        sbert_v = "all-MiniLM-L12-v2"
        encoder = ST(sbert_v, device="cpu" if self.device == "cpu" else f"cuda:{self.device}")
        seq_length = encoder[0].max_seq_length
        video_ids = self.nao_annots["video_id"].unique()
        self.video_embeds = {}

        for video_id in video_ids:
            video_nars = self.nao_annots[self.nao_annots["video_id"] == video_id]["nao_narration"]
            video_nars = ",".join(video_nars)

            no_slices = len(video_nars) // seq_length
            nars_batch = []
            for start in range(no_slices):
                narrs = video_nars[start * seq_length : (start + 1) * seq_length]
                nars_batch.append(narrs)

            if no_slices * seq_length != len(video_nars):
                narrs = video_nars[no_slices * seq_length :]
                nars_batch.append(narrs)

            with torch.no_grad():
                video_embed = encoder.encode(narrs, batch_size=SBERT_ENCODE_BS)

            self.video_embeds[video_id] = self.text_pooling(np.array(video_embed), axis=0)
