# import os
# import json
# import csv
# import logging
# import torch
# from tqdm import tqdm
# from datasets import load_dataset
# from src.model import load_model_and_tokenizer
# import evaluate
# from src.utils import clean_text

# logger = logging.getLogger(__name__)

# def compute_extractiveness(source: str, summary: str) -> float:
#     """
#     Fraction of summary tokens that appear in the source document.
#     """
#     src_tokens = set(source.lower().split())
#     summ_tokens = summary.lower().split()
#     if not summ_tokens:
#         return 0.0
#     overlap = sum(1 for tok in summ_tokens if tok in src_tokens)
#     return overlap / len(summ_tokens)


# def compute_density(source: str, summary: str) -> float:
#     """
#     Compression ratio: summary length divided by source length.
#     """
#     src_len = len(source.split())
#     summ_len = len(summary.split())
#     return summ_len / src_len if src_len > 0 else 0.0


# def evaluate_model(config, ckpt_dir: str, split: str = 'test') -> dict:
#     """
#     Run inference on the specified split and compute evaluation metrics.
#     Saves per-example records (JSON) and aggregate metrics (CSV) in the output directory.

#     Args:
#         config: Configuration object with data, generation, and output fields
#         ckpt_dir: Path to the model checkpoint directory (with tokenizer)
#         split: Dataset split name ('train', 'validation', 'test')

#     Returns:
#         Dictionary of aggregate metrics.
#     """
#     # Device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Load model and tokenizer
#     model, tokenizer = load_model_and_tokenizer(
#         ckpt_dir,
#         device,
#         ddp=False,
#         local_rank=0
#     )

#     # Load dataset
#     logger.info(f"Loading {split} split for evaluation")
#     dataset = load_dataset(
#         config.data.dataset_name,
#         split=split
#     )

#     # Prepare metrics
#     rouge = evaluate.load('rouge')
#     bertscore = evaluate.load('bertscore')

#     records = []
#     gen_texts = []
#     ref_texts = []

#     # Inference loop
#     for idx, example in enumerate(tqdm(dataset, desc="Generating summaries")):
#         src = clean_text(example['document'], remove_stopwords=False)
#         ref = clean_text(example['summary'], remove_stopwords=False)

#         tokens = tokenizer(
#             src,
#             max_length=config.data.max_input_length,
#             truncation=True,
#             return_tensors='pt'
#         ).to(device)

#         # Generate summary with correct max_output_length
#         output_ids = model.generate(
#             tokens['input_ids'],
#             attention_mask=tokens.get('attention_mask', None),
#             max_length=config.data.max_output_length,
#             num_beams=config.generation.num_beams,
#             length_penalty=config.generation.length_penalty,
#             early_stopping=True
#         )

#         gen = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#         ext = compute_extractiveness(src, gen)
#         dens = compute_density(src, gen)

#         records.append({
#             'index': idx,
#             'document': src,
#             'reference_summary': ref,
#             'generated_summary': gen,
#             'extractiveness': ext,
#             'density': dens
#         })
#         gen_texts.append(gen)
#         ref_texts.append(ref)

#     # Compute aggregate metrics
#     logger.info("Computing ROUGE scores...")
#     rouge_res = rouge.compute(predictions=gen_texts, references=ref_texts)
#     logger.info("Computing BERTScore...")
#     bert_res = bertscore.compute(predictions=gen_texts, references=ref_texts, lang='en')

#     metrics = {
#         'rouge1': rouge_res['rouge1'],
#         'rouge2': rouge_res['rouge2'],
#         'rougeL': rouge_res['rougeL'],
#         'bertscore_precision': sum(bert_res['precision']) / len(bert_res['precision']),
#         'bertscore_recall': sum(bert_res['recall']) / len(bert_res['recall']),
#         'bertscore_f1': sum(bert_res['f1']) / len(bert_res['f1']),
#         'avg_extractiveness': sum(r['extractiveness'] for r in records) / len(records),
#         'avg_density': sum(r['density'] for r in records) / len(records)
#     }

#     # Save results
#     out_dir = config.output.output_dir
#     os.makedirs(out_dir, exist_ok=True)

#     # Per-example JSON
#     with open(os.path.join(out_dir, 'eval_records.json'), 'w', encoding='utf-8') as jf:
#         json.dump(records, jf, indent=2)
#     logger.info(f"Saved per-example records to {out_dir}/eval_records.json")

#     # Aggregate CSV
#     metrics_file = os.path.join(out_dir, 'eval_metrics.csv')
#     with open(metrics_file, 'w', newline='', encoding='utf-8') as cf:
#         writer = csv.writer(cf)
#         writer.writerow(list(metrics.keys()))
#         writer.writerow(list(metrics.values()))
#     logger.info(f"Saved aggregate metrics to {out_dir}/eval_metrics.csv")

#     return metrics

# def compute_extractiveness(src: str, summ: str) -> float:
#     st, ss = set(src.lower().split()), summ.lower().split()
#     return sum(1 for w in ss if w in st) / len(ss) if ss else 0.0

# def compute_density(src: str, summ: str) -> float:
#     return len(summ.split()) / len(src.split()) if src.split() else 0.0

# src/evaluate.py

import os
import json
import csv
import logging
import random
import torch
from tqdm import tqdm
from datasets import load_dataset
import evaluate
import numpy as np

from src.model import load_model_and_tokenizer
from src.utils import clean_text, generate_summary

logger = logging.getLogger(__name__)


def compute_extractiveness(source: str, summary: str) -> float:
    """
    Fraction of summary tokens that appear in the source document.
    """
    src_tokens = set(source.lower().split())
    summ_tokens = summary.lower().split()
    if not summ_tokens:
        return 0.0
    overlap = sum(1 for tok in summ_tokens if tok in src_tokens)
    return overlap / len(summ_tokens)


def compute_density(source: str, summary: str) -> float:
    """
    Compression ratio: summary length divided by source length.
    """
    src_len = len(source.split())
    summ_len = len(summary.split())
    return summ_len / src_len if src_len > 0 else 0.0

# def evaluate_model(config, ckpt_dir: str, split: str = 'test', num_samples: int = None) -> dict:
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model, tokenizer = load_model_and_tokenizer(
#         ckpt_dir, device, ddp=False, local_rank=0
#     )

#     ds = load_dataset(config.data.dataset_name, split=split)

#     # If requested, take a small random subset
#     if num_samples is not None and num_samples < len(ds):
#         random.seed(config.training.seed)
#         indices = random.sample(range(len(ds)), num_samples)
#         ds = ds.select(indices)
#         logger.info(f"Sampled {len(ds)} examples for evaluation")


#     rouge = evaluate.load('rouge')
#     bertscore = evaluate.load('bertscore')

#     records, preds, refs = [], [], []
#     for ex in tqdm(ds, desc="Eval generation"):
#         src = clean_text(ex['document'], remove_stopwords=config.data.remove_stopwords)
#         ref = clean_text(ex['summary'],  remove_stopwords=config.data.remove_stopwords)
#         gen = generate_summary(model, tokenizer, ex['document'], config, device)

#         records.append({
#             'document': src,
#             'reference_summary': ref,
#             'generated_summary': gen,
#             'extractiveness': compute_extractiveness(src, gen),
#             'density': compute_density(src, gen)
#         })
#         preds.append(gen)
#         refs.append(ref)

#     logger.info("Computing ROUGE...")
#     rouge_res = rouge.compute(predictions=preds, references=refs)

#     logger.info("Computing BERTScore...")
#     bert_res  = bertscore.compute(predictions=preds, references=refs, lang='en')

#     metrics = {
#         'rouge1': rouge_res['rouge1'],
#         'rouge2': rouge_res['rouge2'],
#         'rougeL': rouge_res['rougeL'],
#         'bertscore_precision': sum(bert_res['precision'])/len(bert_res['precision']),
#         'bertscore_recall':    sum(bert_res['recall'])/len(bert_res['recall']),
#         'bertscore_f1':        sum(bert_res['f1'])/len(bert_res['f1']),
#         'avg_extractiveness':  sum(r['extractiveness'] for r in records)/len(records),
#         'avg_density':         sum(r['density']       for r in records)/len(records)
#     }

#     # out = config.output.output_dir
#     # os.makedirs(out, exist_ok=True)

#     # **Save under the checkpoint directory** **
#     out_dir = ckpt_dir
#     os.makedirs(out_dir, exist_ok=True)


#     # Per‐example JSON
#     json_path = os.path.join(out_dir, 'eval_records.json')
#     with open(json_path, 'w', encoding='utf-8') as jf:
#         json.dump(records, jf, indent=2)
#     logger.info(f"Saved per‐example records to {json_path}")

#     # Aggregate CSV
#     csv_path = os.path.join(out_dir, 'eval_metrics.csv')
#     with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
#         writer = csv.writer(cf)
#         writer.writerow(metrics.keys())
#         writer.writerow(metrics.values())
#     logger.info(f"Saved aggregate metrics to {csv_path}")

#     return metrics


def evaluate_model(
    config,
    ckpt_dir: str,
    split: str = 'test',
    num_samples: int = None
) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer(
        ckpt_dir, device, ddp=False, local_rank=0
    )

    # Load dataset
    ds = load_dataset(config.data.dataset_name, split=split)

    # Subsample if requested
    if num_samples is not None and num_samples < len(ds):
        random.seed(config.training.seed)
        indices = random.sample(range(len(ds)), num_samples)
        ds = ds.select(indices)
        logger.info(f"Sampled {len(ds)} examples for evaluation")

    # Load metrics
    rouge = evaluate.load('rouge')
    bertscore = evaluate.load('bertscore')

    records, preds, refs = [], [], []
    for ex in tqdm(ds, desc="Generating summaries"):
        src = clean_text(ex['document'], remove_stopwords=config.data.remove_stopwords)
        ref = clean_text(ex['summary'],  remove_stopwords=config.data.remove_stopwords)
        gen = generate_summary(model, tokenizer, ex['document'], config, device)

        records.append({
            'document': src,
            'reference_summary': ref,
            'generated_summary': gen,
            'extractiveness': compute_extractiveness(src, gen),
            'density': compute_density(src, gen)
        })
        preds.append(gen)
        refs.append(ref)

    # Compute aggregate ROUGE & BERTScore
    logger.info("Computing ROUGE...")
    rouge_res = rouge.compute(predictions=preds, references=refs)
    logger.info("Computing BERTScore...")
    bert_res  = bertscore.compute(predictions=preds, references=refs, lang='en')

    # Base metrics
    metrics = {
        'rouge1': rouge_res['rouge1'],
        'rouge2': rouge_res['rouge2'],
        'rougeL': rouge_res['rougeL'],
        'bertscore_precision': np.mean(bert_res['precision']),
        'bertscore_recall':    np.mean(bert_res['recall']),
        'bertscore_f1':        np.mean(bert_res['f1']),
        'avg_extractiveness':  np.mean([r['extractiveness'] for r in records]),
        'avg_density':         np.mean([r['density']       for r in records])
    }

    # BERTScore std
    metrics['bertscore_f1_std'] = float(np.std(bert_res['f1']))

    # Optional ROUGE-1 bootstrap CI
    n = len(preds)
    iters = getattr(config.evaluation, 'bootstrap_iters', 0) if hasattr(config, 'evaluation') else 0
    logger.info(f"Iters: {iters}")
    if iters and n > 0:
        logger.info(f"Bootstrap {iters} iters for ROUGE-1 CI...")
        r1_samples = []
        for _ in range(iters):
            idxs = [random.randrange(n) for _ in range(n)]
            p = [preds[i] for i in idxs]
            r = [refs[i] for i in idxs]
            score = rouge.compute(predictions=p, references=r)['rouge1']
            r1_samples.append(score)
        r1_arr = np.array(r1_samples)
        metrics['rouge1_std'] = float(r1_arr.std())
        lo, hi = np.percentile(r1_arr, [2.5, 97.5])
        metrics['rouge1_ci_lower'] = float(lo)
        metrics['rouge1_ci_upper'] = float(hi)

    # Save per‐example records & aggregate CSV under ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, 'eval_records.json'), 'w', encoding='utf-8') as jf:
        json.dump(records, jf, indent=2)
    with open(os.path.join(ckpt_dir, 'eval_metrics.csv'), 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

    logger.info(f"Saved evaluation artifacts to {ckpt_dir}")
    return metrics

