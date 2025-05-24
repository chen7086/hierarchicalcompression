#!/usr/bin/env python3
import os
import sys
import random
import time
import json
import re
import torch
from argparse import ArgumentParser
from collections import defaultdict

import tiktoken
from tqdm import tqdm
from sentence_splitter import split_text_into_sentences
from munch import Munch

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'token'))

# --- sentence_level imports  ---
from model.model import load_model_and_tokenizer
from util.preprocessing import compress_sample, SamplePreprocessor
from util.util import SentenceEmbeddingType

# --- token_level imports ---
from token_compressor import TokenCompressor

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)



# --- Sentence‐level model & tokenizer ---
_MODEL1, _TOKENIZER1 = load_model_and_tokenizer(
    config_path="configs/cpc-1.0-qwen3.json",
    tokenizer_name_or_path="CHTest2001/sentencecompressor",
    lora_name_or_path="CHTest2001/sentencecompressor",
)

# tiktoken counter
_TIK_COUNTER = tiktoken.encoding_for_model("gpt-4")

# Sentence preprocessor
_PREPROCESSOR = SamplePreprocessor(
    tokenizer=_TOKENIZER1,
    max_context_len=6144,
    use_question_as_suffix=False,
    sentence_embedding_type=SentenceEmbeddingType.AVG,
)

# --- Token‐level compressor ---
_TOKEN_COMPRESSOR = TokenCompressor(
    model_name="CHTest2001/tokencompressor",
    device_map="auto",
)


def hierarchical_compress(question: str, content: str, target_tokens: int, hc_ratio: int = 2) -> str:
    """
    Two‐stage compression with peak GPU memory reporting for the highest usage across both stages.
    """
    # reset GPU peak memory counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1) Sentence‐level compression
    start = time.time()
    orig_tokens = len(_TIK_COUNTER.encode(content))
    sent_budget = target_tokens * hc_ratio

    if orig_tokens <= sent_budget:
        mid_text = content
    else:
        enc = _PREPROCESSOR(context=content, question=question, question_for_suffix=question)
        sample = {
            "context": content,
            "input": [question],
            "task": "",
            "encodings": enc
        }
        mid_sents = compress_sample(
            _MODEL1, _TOKENIZER1, _TIK_COUNTER,
            sample, sent_budget, boost_match_regex=None
        )
        mid_text = " ".join(mid_sents)

    mid_tokens = len(_TIK_COUNTER.encode(mid_text))

    # record peak after stage 1
    if torch.cuda.is_available():
        peak_stage1 = torch.cuda.max_memory_allocated()
    else:
        peak_stage1 = 0

    # if already below target, report and return
    if mid_tokens <= target_tokens:
        peak = peak_stage1
        peak_gb = peak / (1024 ** 3)
        print(f"\n↳ stage1: {orig_tokens}→{mid_tokens} tokens")
        print(f"↳ peak VRAM usage: {peak_gb:.2f} GB")
        return mid_text

    # 2) Token‐level compression
    rate = target_tokens / mid_tokens
    out = _TOKEN_COMPRESSOR.compress_prompt(
        [mid_text],
        rate=rate,
        force_tokens=['\n', '?', '.', '!']
    )
    elapsed = time.time() - start
    comp = out["compressed_prompt"]
    comp_tokens = len(_TIK_COUNTER.encode(comp))

    # record peak after stage 2
    if torch.cuda.is_available():
        peak_stage2 = torch.cuda.max_memory_allocated()
    else:
        peak_stage2 = 0

    # compute overall peak
    peak = max(peak_stage1, peak_stage2)
    peak_gb = peak / (1024 ** 3)

    # print stats
    print(f"\n↳ stage1: {orig_tokens}→{mid_tokens} tokens")
    print(f"↳ stage2: {mid_tokens}→{comp_tokens} tokens")
    print(f"↳ compress in {elapsed:.2f}s")
    print(f"↳ peak VRAM usage: {peak_gb:.2f} GB")

    return comp
