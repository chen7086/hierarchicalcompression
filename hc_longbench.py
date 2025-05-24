#!/usr/bin/env python3
import os
import sys
import random
import time
import json
import re
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

def main(args):
    # --- Initialize sentence_level ---
    print(f"\n[sentence_level] loading model & tokenizer for hierarchical compression")
    model1, tokenizer1 = load_model_and_tokenizer(
        config_path=args.config_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        lora_name_or_path=args.lora_bidirectional_name_or_path,
    )
    tok_counter1 = tiktoken.encoding_for_model(args.tiktoken_model)

    # sentence_level: dataset list
    all_datasets = [
        "narrativeqa","qasper","multifieldqa_en","hotpotqa",
        "2wikimqa","musique","gov_report","qmsum",
        "multi_news","lcc","repobench-p","passage_count",
        "passage_retrieval_en","trec","triviaqa","samsum",
    ]
    datasets = all_datasets if args.datasets == "all" else args.datasets.split(',')

    # optional ID filtering
    id_set = None
    if args.id_list_path:
        with open(args.id_list_path, 'r', encoding='utf-8') as f:
            id_set = {l.strip() for l in f if l.strip()}
        print(f"[sentence_level] loaded {len(id_set)} IDs from {args.id_list_path}")

    # sentence-level preprocessor per dataset
    dataset2processor = {
        ds: SamplePreprocessor(
            tokenizer=tokenizer1,
            max_context_len=args.max_context_len,
            use_question_as_suffix=False,
            sentence_embedding_type=SentenceEmbeddingType.AVG,
        )
        for ds in datasets
    }

    dataset2question = {
        'multi_news':         'You are given several news passages. Write a one-page summary of all news.',
        'gov_report':         'Write a one-page summary of the report.',
        'lcc':                'What is the next line of code?',
        'repobench-p':        'What is the next line of code?',
        'passage_count':      'Does this sentence contain meaningful information?',
    }

    # optional boost regex for sentence_level
    dataset2boost_sents_re = {
        # 'trec':        re.compile(r'^(Question\:|Answer\:|Type\:)'),
        'triviaqa':    re.compile(r'^(Passage\:|Answer\:)'),
        'passage_retrieval_en': re.compile(r'^Paragraph \d+\:', re.MULTILINE),
        'passage_count':        re.compile(r'^Paragraph \d+\:', re.MULTILINE),
    } if args.use_boost_regex else {}

    # load & sample each LongBench split
    samples = []
    for ds in tqdm(datasets, desc='Loading Datasets'):
        if 'zh' in ds or ds == 'lsht': continue
        examples = list(__import__('datasets').load_dataset('THUDM/LongBench', ds, split='test'))
        if id_set:
            examples = [e for e in examples if e.get('_id') in id_set]
        if args.sample_num > 0:
            random.seed(args.seed)
            random.shuffle(examples)
            examples = examples[:args.sample_num]

        for e in examples:
            ctx = e['context']
            if isinstance(ctx, list):
                ctx = " ".join(ctx)
            input_null = not e.get('input') or not e['input'][0] or not e['input'][0].strip()
            if input_null and ds not in dataset2question:
                continue

            question = e['input'][0] if not input_null else dataset2question[ds]
            enc = dataset2processor[ds](
                context=ctx,
                question=question,
                question_for_suffix=question
            )

            e['encodings'] = enc
            samples.append({'task': ds, **e})

    # sentence_level compression
    print("[sentence_level] compressing samples...")
    for s in tqdm(samples, desc='[sentence_level] Compress'):
        ctx = s['context']
        orig_tokens = len(tok_counter1.encode(ctx))
        if orig_tokens <= args.compression_target_tokens:
            s['compressed_context'] = ctx
        else:
            sents = compress_sample(
                model1, tokenizer1, tok_counter1,
                s, args.compression_target_tokens,
                dataset2boost_sents_re.get(s['task'], None)
            )
            s['compressed_context'] = ' '.join(sents)
        s.pop('encodings', None)

    # --- Initialize token_level ---
    print(f"\n[token_level] initializing token compressor")
    compressor2 = TokenCompressor(
        model_name=args.model_name,
        device_map=args.device_map
    )
    tok_counter2 = tok_counter1  # reuse same tiktoken model

    # optional boost regex for token_level
    boost_patterns2 = {
        "triviaqa":             re.compile(r"(Passage:|Answer:)"),
        "passage_retrieval_en": re.compile(r"^Paragraph \d+:", re.MULTILINE),
        "passage_count":        re.compile(r"^Paragraph \d+:", re.MULTILINE),
    }

    # token_level compression
    print("[token_level] refining compression...")
    final_samples = []
    for s in tqdm(samples, desc='[token_level] Compressing'):
        ds = s['task']
        ctx1 = s['compressed_context']
        orig_tokens = len(tok_counter2.encode(ctx1))

        # determine force tokens
        default_ft = ['\n', '?', '.', '!']
        if args.use_boost_regex and ds in boost_patterns2:
            hits = boost_patterns2[ds].findall(ctx1)
            seen = set(); boosted = []
            for tok in hits:
                if tok not in seen:
                    seen.add(tok); boosted.append(tok)
            force_tokens = default_ft + boosted
        else:
            force_tokens = default_ft

        # choose target
        if args.target_tokens > 0:
            if orig_tokens <= args.target_tokens:
                comp2 = ctx1; comp_tokens = orig_tokens; elapsed = 0.0
            else:
                rate = args.target_tokens / orig_tokens
                start = time.time()
                out = compressor2.compress_prompt([ctx1], rate=rate, force_tokens=force_tokens)
                elapsed = time.time() - start
                comp2 = out['compressed_prompt']
                comp_tokens = len(tok_counter2.encode(comp2))
        else:
            rate = args.target_ratio
            start = time.time()
            out = compressor2.compress_prompt([ctx1], rate=rate, force_tokens=force_tokens)
            elapsed = time.time() - start
            comp2 = out['compressed_prompt']
            comp_tokens = len(tok_counter2.encode(comp2))

        # update record and collect
        s.update({
            'stage1_tokens': orig_tokens,
            'stage2_tokens': comp_tokens,
            'final_ratio': round(comp_tokens / orig_tokens, 4),
            'stage2_time_s': round(elapsed, 2),
            'compressed_context': comp2
        })
        final_samples.append(s)

    # write out
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as fout:
        json.dump(final_samples, fout, ensure_ascii=False, indent=2)

    print(f"\n✅ All done! Final results saved to {args.save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--final_tokens", type=int, required=True)
    parser.add_argument("--hc_ratio", type=int, required=True, default=2)

    # sentence_level
    parser.add_argument('--config_path', default="configs/cpc-1.0-qwen3.json")
    parser.add_argument('--tokenizer_name_or_path', default="CHTest2001/sentencecompressor")
    parser.add_argument('--lora_bidirectional_name_or_path', default="CHTest2001/sentencecompressor")
    parser.add_argument('--max_context_len', type=int, default=6144)

    # token_level
    parser.add_argument("--model_name", type=str, default="CHTest2001/tokencompressor")
    parser.add_argument("--device_map", type=str, default="auto")

    # all
    parser.add_argument("--tiktoken_model", type=str, default="gpt-4")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--sample_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_boost_regex", action="store_true", default=True)
    parser.add_argument("--id_list_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)

    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    args.compression_target_tokens = args.final_tokens * args.hc_ratio
    args.target_tokens = args.final_tokens

    main(args)