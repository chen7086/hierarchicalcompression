# util/util.py
import evaluate
import torch
from typing import List

from sentence_splitter import split_text_into_sentences
from munch import Munch

import numpy as np

metric = evaluate.load("accuracy")

def preprocess_logits_for_metrics(outputs, labels):
    """
    Preprocess the raw model outputs for metric computation.
    Returns predicted class indices, contrastive loss tensor, and last hidden states.
    """
    if isinstance(outputs, tuple):
        # Depending on the model and config, logits may contain extra tensors
        logits, loss_contrastive, last_hidden_state = outputs
    else:
        # If outputs is not a tuple, assume it's a Namespace or similar
        logits = outputs.logits
        loss_contrastive = torch.tensor(0.0)
        last_hidden_state = None
    return logits.argmax(dim=-1), loss_contrastive, last_hidden_state

def compute_metrics(eval_preds):
    outputs, labels = eval_preds
    preds, loss_contrastive, last_hidden_state = outputs

    preds = preds[:, :-1]
    labels = labels[:, 1:]
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    total = labels.numel()

    result = metric.compute(predictions=preds, references=labels.numpy())
    result['loss_contrastive'] = loss_contrastive.mean().item()
    result['total'] = total

    return result


@torch.no_grad()
def get_ppl_one_step(model, tokenizer, prefix, suffix):
    """
    Compute per-step log probabilities for a given prefix-suffix pair.
    Returns the full logprobs tensor and a list of (token, logprob) for the suffix.
    """
    inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=True)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    all_tokens = prefix_tokens + suffix_tokens
    input_ = {
        'input_ids': torch.LongTensor(all_tokens)[None].cuda(),
        'attention_mask': torch.FloatTensor([1] * len(all_tokens))[None].cuda(),
    }
    outputs = model(**input_)
    logits = torch.log_softmax(outputs.logits[0][:-1], dim=-1)
    logits_chunk = logits[len(prefix_tokens) - 1:]

    scores = []
    assert len(suffix_tokens) == logits_chunk.shape[0]
    for i, t in enumerate(suffix_tokens):
        scores.append((inv_vocab[t], logits_chunk[i, t].item()))

    return {
        'logprobs': logits_chunk,
        'suffix_logprobs': scores,
    }


def kl_divergence(logp, logq):
    """
    Compute KL divergence between two log-probability distributions.
    """
    return torch.mean(torch.sum(torch.exp(logp) * (logp - logq), dim=1), dim=0)


def split_text_into_sentences_keep_slashn(text, language):
    """
    Split text into sentences while preserving explicit newline separators.
    """
    parts = text.split('\n')
    new_parts = []
    N = len(parts)
    prev = None

    for idx, part in enumerate(parts):
        if part.strip() == '':
            continue
        if prev is not None:
            new_parts.append([parts[prev]] + ['\n'] * (idx - prev))
        prev = idx
    if prev is not None:
        new_parts.append([parts[prev]] + ['\n'] * (N - 1 - prev))

    # Flatten and split sentences
    sents_splitted = []
    for block in new_parts:
        snippet, *lineterms = block
        if snippet == '\n':
            sents_splitted.append('\n')
            continue
        sents = split_text_into_sentences(snippet, language=language)
        sents[-1] = sents[-1] + ''.join(lineterms)
        sents_splitted.extend(sents)

    return sents_splitted



class SentenceEmbeddingType:
    AVG = 1


class SpecTokenType:
    END_OF_SENT = "<end_of_sent>"
    END_OF_QUESTION = "<end_of_question>"


def tokenize_and_clip_segments(
    tokenizer,
    segments: List[str],
    segments_labels: List[int],
    max_seq_len: int,
    sentence_embedding_type: int = SentenceEmbeddingType.AVG,
    end_of_sentence_token: str = None
):
    """
    Tokenize a list of text segments and clip to max_seq_len tokens.
    Returns a Munch with text_input_ids, text_segment_ids, segments, segments_labels, sentence_input_ids.
    """
    encodings = {
        'text_input_ids': [],
        'text_segment_ids': [],
        'segments': [],
        'segments_labels': [],
        'sentence_input_ids': [],
    }

    for i, (seg, seg_label) in enumerate(zip(segments, segments_labels)):
        inputs = tokenizer.encode(seg, add_special_tokens=False)
        if len(encodings['text_input_ids']) + len(inputs) > max_seq_len:
            break
        encodings['text_input_ids'].extend(inputs)
        encodings['text_segment_ids'].extend([i] * len(inputs))
        encodings['segments'].append(seg)
        encodings['segments_labels'].append(seg_label)
        encodings['sentence_input_ids'].append(inputs)

    return Munch(encodings)
