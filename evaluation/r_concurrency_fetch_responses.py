#!/usr/bin/env python3
import os
import json
import argparse
import random
import math
from collections import defaultdict
import concurrent.futures
from openai import OpenAI
import tiktoken

dataset2prompt = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a single phrase "
        "if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\n"
        "Now, answer the question based on the story as concisely as you can, "
        "using a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be answered based on the information "
        "in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or "
        "\"unanswerable\". Do not provide any explanation.\n\n"
        "Article: {context}\n\n"
        "Answer the question based on the above article as concisely as you can, using a single phrase or sentence "
        "if possible. If the question cannot be answered based on the information in the article, write "
        "\"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". "
        "Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n"
        "{context}\n\n"
        "Now, answer the following question based on the above text, only give me the answer and do not output "
        "any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\n"
        "Report:\n{context}\n\n"
        "Now, write a one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\n"
        "Transcript:\n{context}\n\n"
        "Now, answer the query based on the above meeting transcript in one or more sentences.\n\n"
        "Query: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. \n\n"
        "News:\n{context}\n\n"
        "Now, write a one-page summary of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some examples of questions.\n\n"
        "{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the answer and do not output any other words. "
        "The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following are some examples.\n\n"
        "{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
        "Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. "
        "In other words, how many non-repeating paragraphs are there in total?\n\n"
        "{context}\n\n"
        "Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, "
        "such as 1, 2, 3, and so on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n"
        "{context}\n\n"
        "The following is an abstract.\n\n{input}\n\n"
        "Please enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\n"
        "The answer is: "
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line of code:\n"
    ),
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_json',      required=True)
    p.add_argument('--inference_model', required=True)
    p.add_argument('--openai_key',      required=True)
    p.add_argument('--answers_json',    required=True)
    p.add_argument('--sample_size', type=float, default=1.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--original',    action='store_true')
    p.add_argument(
        '--zeroshot',
        action='store_true',
        help='Enable zero-shot mode (takes priority over original)'
    )
    p.add_argument('--max_workers', type=int, default=0,
                   help='Number of concurrent threads; 0 means no limit, uses all samples')
    return p.parse_args()

def call_model(args, rec):
    """Encapsulated one inference iteration from the original for-loop, for convenient concurrent calls."""
    ds = rec['dataset']
    q = rec['input']
    
    if args.zeroshot:
        ctx = ""
    else:
        ctx = rec['context'] if args.original else rec['compressed_context']

    prompt = dataset2prompt[ds].format(context=ctx, input=q)


    if ds == "gov_report":
        max_toks = 1000
    elif ds == "multi_news":
        max_toks = 500
    else:
        max_toks = 256

    try:
        client = OpenAI(api_key=args.openai_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=args.inference_model,
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content": prompt}
            ],
            temperature=0.0, max_tokens=max_toks
        )
        full = resp.choices[0].message.content.strip()
        if ds in ("gov_report", "multi_news"):
            ans = full
        elif ds in ("lcc", "repobench-p"):
            ans = full.split('\n')[3]
        else:
            ans = full.split('\n')[0]
    except Exception as e:
        print(f"[Error][{ds}] Inference failed: {e}")
        ans = ""

    return {
        "_id":            rec.get('_id'),
        "dataset":        ds,
        "input":          rec['input'],
        "answers":        rec.get('answers', []),
        "all_classes":    rec.get('all_classes') or [],
        "llm_answers":    ans,
        "llm_rs_answers": None
    }

def main():
    args = parse_args()
    random.seed(args.seed)
    encoding = tiktoken.encoding_for_model("gpt-4")

    with open(args.input_json, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    if args.sample_size < 1.0:
        grouped = defaultdict(list)
        for rec in samples:
            grouped[rec['dataset']].append(rec)
        samples_to_eval = []
        for ds, recs in grouped.items():
            k = min(math.ceil(args.sample_size * len(recs)), len(recs))
            samples_to_eval.extend(random.sample(recs, k))
        print(f"[Info] Sample rate {args.sample_size:.2f}, total {len(samples_to_eval)} records")
    else:
        samples_to_eval = samples
        print(f"[Info] Using all {len(samples_to_eval)} records")

    total = len(samples_to_eval)
    max_workers = args.max_workers or total

    # Execute concurrently
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_rec = { pool.submit(call_model, args, rec): rec for rec in samples_to_eval }
        processed = 0
        for fut in concurrent.futures.as_completed(future_to_rec):
            rec = future_to_rec[fut]
            out = fut.result()
            results.append(out)
            processed += 1
            print(f"{args.input_json} processed {processed}/{total} ({rec['dataset']}), id: {rec.get('_id')}")

    # Write back

    save_dir = os.path.dirname(args.answers_json)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    with open(args.answers_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"LLM outputs saved to: {args.answers_json}")

if __name__ == "__main__":
    main()