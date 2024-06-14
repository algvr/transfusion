from types import MethodType
from typing import Dict
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from modeling.commons import freeze_all_but_bn
from modeling.narration_embeds.datasets.slowfast_features_dsets import SlowFastPooling
from modeling.narration_embeds.t5_adapter_no_loss import *

from sentence_transformers import SentenceTransformer as ST
from sentence_transformers.models.Pooling import Pooling


LEARNABLE_LM = {"sbert_finetune", "gpt2", "t5-wikihow", "slowfast"}


def get_narr_pooling_layer(typey):
    if typey == "slowfast":
        return SlowFastPooling
    elif typey == "sbert_finetune":
        return SBertLayer
    elif typey == "gpt2":
        return GPT2Layer
    elif typey == "t5-wikihow":
        return T5WikiLayer
    else:
        return IdentityLayer


class NarrEmbeddingWrapper(nn.Module):
    def __init__(self, base_model, no_embeds, embed_dim, narration_embeds, padding_idx):
        super().__init__()
        self.embed_layer = nn.Embedding(
            no_embeds, embed_dim, _weight=torch.Tensor(narration_embeds), padding_idx=padding_idx
        )
        self.embed_layer.weight.requires_grad = False
        self.base_model = base_model
        self.epoch = base_model.epoch

    def forward(self, x):
        language_f = self.embed_layer(x["language_f"])
        x["language_f"] = language_f
        return self.base_model(x)

    def call_model_epoch_triggers(self, epoch):
        if epoch >= self.base_model.narr_embed_args["train_ep"] and self.base_model.narr_embed_args["train_ep"] != -1:
            self.unfreeze_embeddings()
            print("Unfroze embeddings gradients")

    def unfreeze_embeddings(self):
        self.embed_layer.weight.requires_grad = True

    def get_heatmap_channels(self):
        return self.base_model.get_heatmap_channels()

    def is_classfying(self):
        return self.base_model.is_classfying()

    def is_heatmap_pred_on(self):
        return self.base_model.is_heatmap_pred_on()

    def get_classify_noun(self):
        return self.base_model.get_classify_noun()

    def get_classify_verb(self):
        return self.base_model.get_classify_verb()


class SBertLayer(nn.Module):
    def __init__(self, narr_args, out_mode="embedding"):
        super().__init__()
        self.narr_args = narr_args
        self.dim = narr_args["size"]
        self.lang_f_dropout = nn.Dropout(narr_args["lang_dropout"])
        self.out_key = "sentence_embedding" if out_mode == "embedding" else "token_embeddings"
        self.model_v = narr_args["model_v"]
        self.encoder = ST(self.model_v)
        self.finetune_layers = narr_args.get("finetune_layers", 1)

        self.encoder.apply(freeze_all_but_bn)
        self.unfreeze_layers = [
            self.encoder._modules["0"]._modules["auto_model"]._modules["pooler"],
        ]
        for i in range(self.finetune_layers):
            self.unfreeze_layers.append(self.encoder._modules["0"]._modules["auto_model"]._modules["encoder"]._modules["layer"][-i])

        self.use_out_mlp = narr_args["out_mlp"] and narr_args["out_mlp"] != self.dim
        if self.use_out_mlp: 
            self.out_mlp = nn.Linear(self.dim, narr_args["out_mlp"])
        self.use_out_tanh = narr_args["out_tanh"]
        self.out_dropout = nn.Dropout(narr_args["out_dropout"])

        self.type_embeddings = narr_args.get("type_embeddings", [])
        self.type_embedding_params = nn.ParameterDict(
            {
                k: nn.Parameter(
                    torch.randn((self.dim,), device=self.encoder.device) / narr_args["type_embedding_init_div"]
                )
                for k in self.type_embeddings
            }
        )

    def unfreeze_embeddings(self):
        for layer in self.unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True
        print("Unfroze narration pooling layer")

    def forward(self, lang_f, pad_mask=False, w_attentions=False):
        ### example input: ###
        # into constructor:
        # self.type_embeddings = ["test1", "test2"]
        # self.type_embedding_params = nn.ParameterDict({k: nn.Parameter(torch.randn((self.dim,),
        #                                                                            device=self.encoder.device))
        #                                                for k in self.type_embeddings})
        # lang_f = ['sit<test1> bed,<test1> hold<test2> shirt<test2>', 'cut shirt, hold object']
        # (note that test1 is applied to "bed", not to ",")

        # store mapping from start indices of words to corresponding type embeddings where applicable

        # the type-cleared text to forward:
        lang_f_processed = []

        # for each sample in batch, mapping of word start indices to nn.Parameters:
        lang_f_mappings = []

        for lang_f_str in lang_f:
            lang_f_str_processed = ""
            lang_f_str_param_mapping = {}
            for word in lang_f_str.split(" "):
                if len(lang_f_str_processed) > 0:
                    lang_f_str_processed += " "
                word_has_types = "<" in word and ">" in word
                word_types = word[(word.index("<") + 1) : word.index(">")] if word_has_types else None
                word_processed = word[: word.index("<")] if word_has_types else word
                if word_has_types:
                    lang_f_str_param_mapping[len(lang_f_str_processed)] = list(
                        self.type_embedding_params[word_type.strip()] for word_type in word_types.split(",")
                    )
                lang_f_str_processed += word_processed

            lang_f_processed.append(lang_f_str_processed)
            lang_f_mappings.append(lang_f_str_param_mapping)

        # running example: lang_f_processed = ['sit bed, hold shirt', 'cut shirt, hold object']

        tokenized_lang_f = self.encoder.tokenizer(
            lang_f_processed,
            return_tensors="pt",
            padding=True,
            truncation="longest_first",
            max_length=self.encoder.max_seq_length,
        )
        for k, v in tokenized_lang_f.items():
            tokenized_lang_f[k] = v.to(self.encoder.device)

        # apply transformer
        output = self.encoder[0](tokenized_lang_f)
        if w_attentions:
            attentions = output["attentions"]
            del output["attentions"]
        else:
            attentions = None

        # add type embeddings to corresponding outputs using word start -> embedding dictionary
        for lang_f_idx, lang_f_mapping in enumerate(lang_f_mappings):
            for word_idx, offset in enumerate(tokenized_lang_f.encodings[lang_f_idx].offsets):
                # get corresponding embedding for start index of this word, if any
                params = lang_f_mapping.get(offset[0], None)
                if params is not None:
                    # get word index in input (not start index)
                    word_id = tokenized_lang_f.encodings[lang_f_idx].words[word_idx]
                    if word_id is None:  # BOS
                        continue

                    # get tokens corresponding to word index (this word); add corresponding embedding to each token
                    token_span = tokenized_lang_f.word_to_tokens(lang_f_idx, word_id)
                    for token_idx in range(token_span.start, token_span.end):
                        for param in params:
                            output["token_embeddings"][lang_f_idx][token_idx] += param

        # apply pooling and normalization
        output = self.encoder[1](output)
        output = self.encoder[2](output)
        embeddings = output[self.out_key]

        if self.use_out_mlp:
            embeddings = self.out_mlp(embeddings)

        if self.use_out_tanh:
            embeddings = torch.tanh(embeddings)

        if pad_mask:
            return self.out_dropout(embeddings), attentions, output["attention_mask"]
        else:
            return self.out_dropout(embeddings), attentions, None


class STPoolingAdapter(Pooling):
    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
    ):
        super().__init__(
            word_embedding_dimension,
            pooling_mode,
            pooling_mode_cls_token,
            pooling_mode_max_tokens,
            pooling_mode_mean_tokens,
            pooling_mode_mean_sqrt_len_tokens,
        )

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["last_hidden_state"]
        attention_mask = features["attention_mask"]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        features["sentence_embedding"] = output_vector
        return features


class GPT2Layer(nn.Module):
    def __init__(self, narr_args, out_mode="embedding"):
        super().__init__()
        self.narr_args = narr_args
        self.dim = narr_args["size"]
        self.lang_f_dropout = nn.Dropout(narr_args["lang_dropout"])
        self.out_key = "sentence_embedding" if out_mode == "embedding" else "last_hidden_state"
        self.model_v = narr_args["model_v"]
        self.mean_pool_layer = STPoolingAdapter(self.dim, pooling_mode="mean")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_v)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = AutoModelForCausalLM.from_pretrained(self.model_v)

        self.encoder.lm_head = nn.Identity()

        self.encoder.apply(freeze_all_but_bn)

        self.encoder._modules["transformer"]._modules["h"][5].mlp.dropout = nn.Identity()
        self.unfreeze_layers = [
            # self.encoder._modules["0"]._modules["auto_model"]._modules["encoder"]._modules["layer"][10],
            self.encoder._modules["transformer"]._modules["h"][5].mlp,
            # self.encoder._modules["transformer"]._modules["h"]._modules["pooler"],
        ]

        self.use_out_mlp = narr_args["out_mlp"]
        if self.use_out_mlp and self.use_out_mlp != self.dim:
            self.out_mlp = nn.Linear(self.dim, self.use_out_mlp)
        self.use_out_tanh = narr_args["out_tanh"]
        self.out_dropout = nn.Dropout(narr_args["out_dropout"])

    def unfreeze_embeddings(self):
        for layer in self.unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True
        print("Unfroze narration pooling layer")

    def forward(self, lang_f, pad_mask=False, w_attentions=False):
        tokenized_lang_f = self.tokenizer(lang_f, return_tensors="pt", padding=True)
        for k, v in tokenized_lang_f.items():
            tokenized_lang_f[k] = v.to(self.encoder.device)

        # apply transformer
        # output = self.encoder[0](tokenized_lang_f)
        output = self.encoder.transformer(**tokenized_lang_f, return_dict=True, output_attentions=w_attentions)
        if w_attentions:
            attentions = output["attentions"]
            del output["attentions"]
        else:
            attentions = None

        output["attention_mask"] = tokenized_lang_f.pop("attention_mask")
        # apply pooling
        output = self.mean_pool_layer(output)
        embeddings = output[self.out_key]

        # apply normalization on features
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.use_out_mlp:
            embeddings = self.out_mlp(embeddings)

        if self.use_out_tanh:
            embeddings = torch.tanh(embeddings)

        if pad_mask:
            return self.out_dropout(embeddings), attentions, output["attention_mask"]
        else:
            return self.out_dropout(embeddings), attentions, None


t5_urls = {
    "t5-small": "Chikashi/t5-small-finetuned-cnndm-wikihow",
    "t5-large": "Chikashi/t5-large-finetuned-cnndm-wikihow",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-small": "google/flan-t5-small",
}


class T5WikiLayer(nn.Module):
    def __init__(self, narr_args, out_mode="embedding"):
        super().__init__()
        self.narr_args = narr_args
        self.dim = narr_args["size"]
        self.lang_f_dropout = nn.Dropout(narr_args["lang_dropout"])
        self.out_key = "sentence_embedding" if out_mode == "embedding" else "last_hidden_state"
        self.model_v = t5_urls[narr_args["model_v"]]
        self.mean_pool_layer = STPoolingAdapter(self.dim, pooling_mode="mean")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_v)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = AutoModelForSeq2SeqLM.from_pretrained(self.model_v)
        self.encoder.forward = MethodType(forward_t5_no_loss, self.encoder)

        del self.encoder.lm_head

        self.encoder.apply(freeze_all_but_bn)
        self.encoder.encoder.dropout = nn.Identity()
        self.unfreeze_layers = [
            # self.encoder._modules["0"]._modules["auto_model"]._modules["encoder"]._modules["layer"][10],
            self.encoder.encoder.block[-1]
        ]

        self.use_out_mlp = narr_args["out_mlp"] != self.dim
        if self.use_out_mlp:
            self.out_mlp = nn.Linear(self.dim, narr_args["out_mlp"])
        self.use_out_tanh = narr_args["out_tanh"]
        self.out_dropout = nn.Dropout(narr_args["out_dropout"])

    def unfreeze_embeddings(self):
        for layer in self.unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True
        print("Unfroze narration pooling layer")

    def forward(self, lang_f, pad_mask=False, w_attentions=False):
        tokenized_lang_f = self.tokenizer(lang_f, return_tensors="pt", padding=True)
        for k, v in tokenized_lang_f.items():
            tokenized_lang_f[k] = v.to(self.encoder.device)

        # apply transformer
        # output = self.encoder[0](tokenized_lang_f)
        output = self.encoder.encoder(**tokenized_lang_f, return_dict=True, output_attentions=w_attentions)
        if w_attentions:
            attentions = output["attentions"]
            del output["attentions"]
        else:
            attentions = None

        output["attention_mask"] = tokenized_lang_f.pop("attention_mask")
        # apply pooling
        output = self.mean_pool_layer(output)
        embeddings = output[self.out_key]

        # apply normalization on features
        embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.use_out_mlp:
            embeddings = self.out_mlp(embeddings)

        if self.use_out_tanh:
            embeddings = torch.tanh(embeddings)

        if pad_mask:
            return self.out_dropout(embeddings), attentions, output["attention_mask"]
        else:
            return self.out_dropout(embeddings), attentions, None


class IdentityLayer(torch.nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, tensor, *args, **kwargs):
        return tensor, None, None
