#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import os
import random
import re
import string
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
import tiktoken


class TokenClfDataset(Dataset):
    
    def __init__(
        self,
        texts,
        max_len=512,
        tokenizer=None,
        model_name="bert-base-multilingual-cased",
    ):
        self.len = len(texts)
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.model_name = model_name
        

        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"


    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer.tokenize(text)

        tokenized_text = (
            [self.cls_token] + tokenized_text + [self.sep_token]
        )

        if len(tokenized_text) > self.max_len:
            tokenized_text = tokenized_text[: self.max_len]
        else:
            tokenized_text = tokenized_text + [
                self.pad_token for _ in range(self.max_len - len(tokenized_text))
            ]

        attn_mask = [1 if tok != self.pad_token else 0 for tok in tokenized_text]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
        }

    def __len__(self):
        return self.len


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_begin_of_new_word(token, model_name, force_tokens, token_map):

    if token.lstrip("##") in force_tokens or token.lstrip("##") in set(
        token_map.values()
    ):
        return True
    return not token.startswith("##")



def replace_added_token(token, token_map):
    for ori_token, new_token in token_map.items():
        token = token.replace(new_token, ori_token)
    return token


def get_pure_token(token, model_name):
    return token.lstrip("##")



class TokenCompressor:
    
    def __init__(
        self,
        model_name: str = "CHTest2001/tokencompressor",
        device_map: str = "cuda",
        model_config: dict = {},
        max_batch_size: int = 50,
        max_force_token: int = 100,
    ):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_seq_len = 512
        self.max_force_token = max_force_token
        
        seed_everything(42)
        
        self._load_model(model_name, device_map, model_config)
        
        self.oai_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        self.special_tokens = set(
            [
                v
                for k, v in self.tokenizer.special_tokens_map.items()
                if k != "additional_special_tokens"
            ]
        )
        
        self.added_tokens = [f"[NEW{i}]" for i in range(max_force_token)]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.added_tokens}
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _load_model(self, model_name: str, device_map: str = "cuda", model_config: dict = {}):
        trust_remote_code = model_config.get("trust_remote_code", True)
        if "trust_remote_code" not in model_config:
            model_config["trust_remote_code"] = trust_remote_code
            
        config = AutoConfig.from_pretrained(model_name, **model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_config)
        
        if model_config.get("pad_to_left", True):
            tokenizer.padding_side = "left"
            tokenizer.pad_token_id = (
                config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            )
            
        self.device = (
            device_map
            if any(key in device_map for key in ["cuda", "cpu", "mps"])
            else "cuda"
        )
        
        if "cuda" in device_map or "cpu" in device_map:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                torch_dtype=model_config.pop(
                    "torch_dtype", "auto" if device_map == "cuda" else torch.float32
                ),
                device_map=device_map,
                config=config,
                ignore_mismatched_sizes=True,
                **model_config,
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                device_map=device_map,
                torch_dtype=model_config.pop("torch_dtype", "auto"),
                pad_token_id=tokenizer.pad_token_id,
                **model_config,
            )
            
        self.tokenizer = tokenizer
        self.model = model

    def get_token_length(self, text: str, use_oai_tokenizer: bool = False):
        if use_oai_tokenizer:
            return len(self.oai_tokenizer.encode(text))
        else:
            return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def compress_prompt(
        self,
        context: List[str],
        rate: float = 0.5,
        target_token: int = -1,
        use_context_level_filter: bool = False,
        use_token_level_filter: bool = True,
        target_context: int = -1,
        context_level_rate: float = 1.0,
        context_level_target_token: int = -1,
        force_context_ids: List[int] = [],
        return_word_label: bool = False,
        word_sep: str = "\t\t|\t\t",
        label_sep: str = " ",
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
        chunk_end_tokens: List[str] = [".", "\n"],
    ):
        assert len(force_tokens) <= self.max_force_token
        
        token_map = {}
        for i, t in enumerate(force_tokens):
            if len(self.tokenizer.tokenize(t)) != 1:
                token_map[t] = self.added_tokens[i]
                
        chunk_end_tokens = copy.deepcopy(chunk_end_tokens)
        for c in chunk_end_tokens:
            if c in token_map:
                chunk_end_tokens.append(token_map[c])
        chunk_end_tokens = set(chunk_end_tokens)

        if isinstance(context, str):
            context = [context]
        context = copy.deepcopy(context)

        if len(context) == 1 and use_context_level_filter:
            use_context_level_filter = False

        n_original_token = 0
        context_chunked = []
        for i in range(len(context)):
            n_original_token += self.get_token_length(context[i], use_oai_tokenizer=True)
            for ori_token, new_token in token_map.items():
                context[i] = context[i].replace(ori_token, new_token)
            context_chunked.append(
                self._chunk_context(context[i], chunk_end_tokens=chunk_end_tokens)
            )

        if use_context_level_filter:
            if (
                target_context <= 0
                and context_level_rate >= 1.0
                and context_level_target_token <= 0
            ):
                if target_token < 0 and rate < 1.0:
                    context_level_rate = (
                        (rate + 1.0) / 2 if use_token_level_filter else rate
                    )
                if target_token >= 0:
                    context_level_target_token = (
                        target_token * 2 if use_token_level_filter else target_token
                    )

            if target_context >= 0:
                context_level_rate = min(target_context / len(context), 1.0)
            if context_level_target_token >= 0:
                context_level_rate = min(
                    context_level_target_token / n_original_token, 1.0
                )

            context_probs, context_words = self._get_context_prob(
                context_chunked,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
            )

            threshold = np.percentile(
                context_probs, int(100 * (1 - context_level_rate))
            )

            reserved_context = []
            context_label = [False] * len(context_probs)
            for i, p in enumerate(context_probs):
                if p >= threshold or (
                    force_context_ids is not None and i in force_context_ids
                ):
                    reserved_context.append(context_chunked[i])
                    context_label[i] = True
                    
            n_reserved_token = 0
            for chunks in reserved_context:
                for c in chunks:
                    n_reserved_token += self.get_token_length(c, use_oai_tokenizer=True)
            if target_token >= 0:
                rate = min(target_token / n_reserved_token, 1.0)

            if use_token_level_filter:
                compressed_context, word_list, word_label_list = self._compress(
                    reserved_context,
                    reduce_rate=max(0, 1 - rate),
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                )
            else:
                compressed_context, word_list, word_label_list = self._compress(
                    reserved_context,
                    reduce_rate=0,
                    token_to_word=token_to_word,
                    force_tokens=force_tokens,
                    token_map=token_map,
                    force_reserve_digit=force_reserve_digit,
                    drop_consecutive=drop_consecutive,
                )

            n_compressed_token = 0
            for c in compressed_context:
                n_compressed_token += self.get_token_length(c, use_oai_tokenizer=True)
            saving = (n_original_token - n_compressed_token) * 0.06 / 1000
            ratio = (
                1 if n_compressed_token == 0 else n_original_token / n_compressed_token
            )
            
            res = {
                "compressed_prompt": "\n\n".join(compressed_context),
                "compressed_prompt_list": compressed_context,
                "origin_tokens": n_original_token,
                "compressed_tokens": n_compressed_token,
                "ratio": f"{ratio:.1f}x",
                "rate": f"{1 / ratio * 100:.1f}%",
                "saving": f", Saving ${saving:.1f} in GPT-4.",
            }
            
            if return_word_label:
                words = []
                labels = []
                j = 0
                for i in range(len(context)):
                    if context_label[i]:
                        words.extend(word_list[j])
                        labels.extend(word_label_list[j])
                        j += 1
                    else:
                        words.extend(context_words[i])
                        labels.extend([0] * len(context_words[i]))
                word_label_lines = word_sep.join(
                    [f"{word}{label_sep}{label}" for word, label in zip(words, labels)]
                )
                res["fn_labeled_original_prompt"] = word_label_lines
            return res

        if target_token > 0:
            rate = min(target_token / n_original_token, 1.0)

        if use_token_level_filter:
            compressed_context, word_list, word_label_list = self._compress(
                context_chunked,
                reduce_rate=max(0, 1 - rate),
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )
        else:
            compressed_context, word_list, word_label_list = self._compress(
                context_chunked,
                reduce_rate=0,
                token_to_word=token_to_word,
                force_tokens=force_tokens,
                token_map=token_map,
                force_reserve_digit=force_reserve_digit,
                drop_consecutive=drop_consecutive,
            )

        n_compressed_token = 0
        for c in compressed_context:
            n_compressed_token += self.get_token_length(c, use_oai_tokenizer=True)
        saving = (n_original_token - n_compressed_token) * 0.06 / 1000
        ratio = 1 if n_compressed_token == 0 else n_original_token / n_compressed_token
        
        res = {
            "compressed_prompt": "\n\n".join(compressed_context),
            "compressed_prompt_list": compressed_context,
            "origin_tokens": n_original_token,
            "compressed_tokens": n_compressed_token,
            "ratio": f"{ratio:.1f}x",
            "rate": f"{1 / ratio * 100:.1f}%",
            "saving": f", Saving ${saving:.1f} in GPT-4.",
        }
        
        if return_word_label:
            words = []
            labels = []
            for w_list, l_list in zip(word_list, word_label_list):
                words.extend(w_list)
                labels.extend(l_list)

            word_label_lines = word_sep.join(
                [f"{word}{label_sep}{label}" for word, label in zip(words, labels)]
            )
            res["fn_labeled_original_prompt"] = word_label_lines
        return res

    def _get_context_prob(
        self,
        context_list: list,
        token_to_word="mean",
        force_tokens: List[str] = [],
        token_map: dict = {},
        force_reserve_digit: bool = False,
    ):
        chunk_list = []
        for chunks in context_list:
            for c in chunks:
                chunk_list.append(c)

        dataset = TokenClfDataset(
            chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len
        )
        dataloader = DataLoader(
            dataset, batch_size=self.max_batch_size, shuffle=False, drop_last=False
        )

        chunk_probs = []
        chunk_words = []
        
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["ids"].to(self.device, dtype=torch.long)
                mask = batch["mask"].to(self.device, dtype=torch.long) == 1

                outputs = self.model(input_ids=ids, attention_mask=mask)
                loss, logits = outputs.loss, outputs.logits
                probs = F.softmax(logits, dim=-1)

                for j in range(ids.shape[0]):
                    _probs = probs[j, :, 1]
                    _ids = ids[j]
                    _mask = mask[j]

                    active_probs = torch.masked_select(_probs, _mask)
                    active_ids = torch.masked_select(_ids, _mask)

                    tokens = self.tokenizer.convert_ids_to_tokens(
                        active_ids.squeeze().tolist()
                    )
                    token_probs = [prob for prob in active_probs.cpu().numpy()]

                    (
                        words,
                        valid_token_probs,
                        valid_token_probs_no_force,
                    ) = self._merge_token_to_word(
                        tokens,
                        token_probs,
                        force_tokens=force_tokens,
                        token_map=token_map,
                        force_reserve_digit=force_reserve_digit,
                    )
                    word_probs_no_force = self._token_prob_to_word_prob(
                        valid_token_probs_no_force, convert_mode=token_to_word
                    )

                    if "xlm-roberta-large" in self.model_name:
                        for i in range(len(words)):
                            words[i] = words[i].lstrip("▁")
                    chunk_words.append(words)
                    chunk_probs.append(word_probs_no_force)

        prev_idx = 0
        context_probs = []
        context_words = []
        for chunk_list in context_list:
            n_chunk = len(chunk_list)
            context_probs.append([])
            context_words.append([])
            for i in range(n_chunk):
                context_probs[-1].extend(chunk_probs[prev_idx + i])
                context_words[-1].extend(chunk_words[prev_idx + i])
            prev_idx = prev_idx + n_chunk
        context_probs = [sum(probs) / len(probs) for probs in context_probs]
        return context_probs, context_words

    def _chunk_context(self, origin_text, chunk_end_tokens):
        max_len = self.max_seq_len - 2
        origin_list = []
        origin_tokens = self.tokenizer.tokenize(origin_text)
        n = len(origin_tokens)
        st = 0
        
        while st < n:
            if st + max_len > n - 1:
                chunk = self.tokenizer.convert_tokens_to_string(origin_tokens[st:n])
                origin_list.append(chunk)
                break
            else:
                ed = st + max_len
                for j in range(0, ed - st):
                    if origin_tokens[ed - j] in chunk_end_tokens:
                        ed = ed - j
                        break
                chunk = self.tokenizer.convert_tokens_to_string(
                    origin_tokens[st : ed + 1]
                )
                origin_list.append(chunk)
                st = ed + 1
        return origin_list

    def _merge_token_to_word(
        self, tokens, token_probs, force_tokens, token_map, force_reserve_digit
    ):
        words = []
        word_probs = []
        word_probs_no_force = []

        for token, prob in zip(tokens, token_probs):
            if token in self.special_tokens:
                continue
            elif is_begin_of_new_word(token, self.model_name, force_tokens, token_map):
                pure_token = get_pure_token(token, self.model_name)
                prob_no_force = prob
                if pure_token in force_tokens or pure_token in set(token_map.values()):
                    prob = 1.0
                token = replace_added_token(token, token_map)
                words.append(token)
                word_probs.append(
                    [
                        1.0
                        if force_reserve_digit and bool(re.search(r"\d", token))
                        else prob
                    ]
                )
                word_probs_no_force.append([prob_no_force])
            else:
                pure_token = get_pure_token(token, self.model_name)
                words[-1] += pure_token
                word_probs[-1].append(
                    1.0
                    if force_reserve_digit and bool(re.search(r"\d", token))
                    else prob
                )
                word_probs_no_force[-1].append(prob_no_force)

        return words, word_probs, word_probs_no_force

    def _token_prob_to_word_prob(self, token_probs, convert_mode="mean"):
        if convert_mode == "mean":
            word_probs = [sum(p) / len(p) for p in token_probs]
        elif convert_mode == "first":
            word_probs = [p[0] for p in token_probs]
        else:
            raise NotImplementedError(f"Convert mode {convert_mode} not supported")

        return word_probs

    def _compress(
        self,
        context_list: list,
        reduce_rate: float = 0.5,
        token_to_word: str = "mean",
        force_tokens: List[str] = [],
        token_map: dict = {},
        force_reserve_digit: bool = False,
        drop_consecutive: bool = False,
    ):
        def split_string_to_words(input_string):
            pattern = r'\b\w+\b|[<>=/!@#$%^&*()?":{}|\\`~;_+-]'
            result = re.findall(pattern, input_string)
            return result

        if reduce_rate <= 0:
            words, word_labels = [], []
            for i in range(len(context_list)):
                chunk_list = context_list[i]
                chunk_words = []
                chunk_word_labels = []
                for j in range(len(chunk_list)):
                    # 恢复原始token
                    for ori_token, new_token in token_map.items():
                        chunk_list[j] = chunk_list[j].replace(new_token, ori_token)
                    ws = split_string_to_words(chunk_list[j])
                    chunk_words.extend(ws)
                    chunk_word_labels.extend([1 for _ in range(len(ws))])
                context_list[i] = "".join(chunk_list)
                words.append(chunk_words)
                word_labels.append(chunk_word_labels)
            return context_list, words, word_labels

        chunk_list = []
        for chunks in context_list:
            for c in chunks:
                chunk_list.append(c)

        dataset = TokenClfDataset(
            chunk_list, tokenizer=self.tokenizer, max_len=self.max_seq_len
        )
        dataloader = DataLoader(
            dataset, batch_size=self.max_batch_size, shuffle=False, drop_last=False
        )

        compressed_chunk_list = []
        word_list = []
        word_label_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                ids = batch["ids"].to(self.device, dtype=torch.long)
                mask = batch["mask"].to(self.device, dtype=torch.long) == 1

                outputs = self.model(input_ids=ids, attention_mask=mask)
                loss, logits = outputs.loss, outputs.logits
                probs = F.softmax(logits, dim=-1)

                for j in range(ids.shape[0]):
                    chunk_probs = probs[j, :, 1]
                    chunk_ids = ids[j]
                    chunk_mask = mask[j]

                    active_probs = torch.masked_select(chunk_probs, chunk_mask)
                    active_ids = torch.masked_select(chunk_ids, chunk_mask)

                    tokens = self.tokenizer.convert_ids_to_tokens(
                        active_ids.squeeze().tolist()
                    )
                    token_probs = [prob for prob in active_probs.cpu().numpy()]

                    words, valid_token_probs, _ = self._merge_token_to_word(
                        tokens=tokens,
                        token_probs=token_probs,
                        force_tokens=force_tokens,
                        token_map=token_map,
                        force_reserve_digit=force_reserve_digit,
                    )
                    word_probs = self._token_prob_to_word_prob(
                        valid_token_probs, convert_mode=token_to_word
                    )

                    if drop_consecutive:
                        threshold = np.percentile(word_probs, int(100 * reduce_rate))
                        is_token_between = False
                        prev = None
                        for i, (word, word_prob) in enumerate(zip(words, word_probs)):
                            if word in force_tokens:
                                if is_token_between:
                                    is_token_between = False
                                elif not is_token_between and word == prev:
                                    word_probs[i] = 0.0
                                prev = word
                            else:
                                is_token_between |= word_prob > threshold

                    new_token_probs = []
                    for word, word_prob in zip(words, word_probs):
                        num_token = len(self.oai_tokenizer.encode(word))
                        new_token_probs.extend([word_prob for _ in range(num_token)])
                    threshold = np.percentile(
                        new_token_probs, int(100 * reduce_rate + 1)
                    )

                    keep_words = []
                    word_labels = []
                    assert len(words) == len(word_probs)
                    for word, word_prob in zip(words, word_probs):
                        if word_prob > threshold or (
                            threshold == 1.0 and word_prob == threshold
                        ):
                            if (
                                drop_consecutive
                                and word in force_tokens
                                and len(keep_words) > 0
                                and keep_words[-1] == word
                            ):
                                word_labels.append(0)
                            else:
                                keep_words.append(word)
                                word_labels.append(1)
                        else:
                            word_labels.append(0)
                            
                    keep_str = self.tokenizer.convert_tokens_to_string(keep_words)
                    if "xlm-roberta-large" in self.model_name:
                        for i in range(len(words)):
                            words[i] = words[i].lstrip("▁")

                    compressed_chunk_list.append(keep_str)
                    word_list.append(words[:])
                    word_label_list.append(word_labels[:])

        compressed_context_list = []
        original_word_list = []
        original_word_label_list = []
        prev_idx = 0
        for chunk_list in context_list:
            n_chunk = len(chunk_list)
            compressed_context_list.append(
                "".join(compressed_chunk_list[prev_idx : prev_idx + n_chunk])
            )
            original_word_list.append([])
            original_word_label_list.append([])
            for i in range(n_chunk):
                original_word_list[-1].extend(word_list[prev_idx + i])
                original_word_label_list[-1].extend(word_label_list[prev_idx + i])
            prev_idx = prev_idx + n_chunk

        return compressed_context_list, original_word_list, original_word_label_list