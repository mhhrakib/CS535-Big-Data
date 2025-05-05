import logging
import ftfy
import unicodedata
import re
import contractions
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Optional, Set
from transformers import PegasusTokenizerFast, BartTokenizerFast, LEDTokenizerFast, AutoTokenizer
import string
from types import SimpleNamespace

import yaml


import torch

# Ensure NLTK resources are available
# one-time downloads
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

# Configure module-level logger
debug_logger = logging.getLogger(__name__)

DOC_SEPARATOR = "<doc_sep>"


def clean_text(
    text: str,
    remove_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None
) -> str:
    # 1) Fix encoding
    text = ftfy.fix_text(text)
    # 2) Expand contractions
    text = contractions.fix(text)
    # 3) Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    # 4) Strip HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # 5) Mark doc boundaries
    text = text.replace("|||||", f" {DOC_SEPARATOR} ")
    # 6) Normalize all dash‐types to hyphen
    text = text.replace("—", "-").replace("–", "-")
    # 7) Strip control chars
    text = re.sub(r"[\r\n\t]+", " ", text)
    # 7b) Collapse runs of underscores (e.g. "___" -> "_")
    text = re.sub(r"(_)\1+", r"\1", text)
    # 8) Collapse repeated punctuation
    text = re.sub(r"([^\w<>\s])\1+", r"\1", text)
    # 9) Drop non‐ASCII (after we normalized dashes and preserved separator)
    text = text.encode("ascii", errors="ignore").decode()
    # 10) Whitelist letters, digits, whitespace, basic punctuation, angle brackets, underscore
    text = re.sub(r"[^A-Za-z0-9_\s\.\,\!\?\;\:\'\"\-\<\>]", " ", text)
    # 11) Collapse whitespace
    text = " ".join(text.split())

    # 12) Optional stopword removal
    tokens = text.split()
    if remove_stopwords:
        if stopwords is None:
            stopwords = set(nltk_stopwords.words("english"))
        tokens = [t for t in tokens if t.lower() not in stopwords]

    # 13) Final: keep only tokens with alnum or the separator exactly
    clean_tokens = [
        t for t in tokens
        if t == DOC_SEPARATOR or re.search(r"[A-Za-z0-9_]", t)
    ]

    return " ".join(clean_tokens)


def dedup_sentences(text: str) -> str:
    """
    Simple sentence‐level deduplication: only keep the first occurrence
    of each sentence (case‐insensitive).
    """
    from nltk.tokenize import sent_tokenize
    seen = set()
    out: List[str] = []
    for s in sent_tokenize(text):
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return ' '.join(out)



def dedup_sentences_fuzzy(text: str, threshold: float = 0.55) -> str:
    """
    Sentence‐level deduplication with fuzzy matching (Jaccard similarity) (punctuation removed):
      - Split text into sentences.
      - Keep a list of unique sentences; compare each new sentence
        to existing ones via Jaccard(token_set) ≥ threshold.
      - If similar, keep the longer of the two.
      - Otherwise, add as new.
    """

    sentences = sent_tokenize(text)
    kept: List[str] = []
    kept_token_sets: List[set] = []

    for s in sentences:
        # 1) Lowercase & strip punctuation for similarity
        s_lower = s.lower()
        s_nopunct = s_lower.translate(str.maketrans("", "", string.punctuation))
        tokens = set(s_nopunct.split())
        if not tokens:
            continue

        replaced = False
        for i, old_tokens in enumerate(kept_token_sets):
            # Jaccard similarity on pure word‐sets
            inter = tokens & old_tokens
            union = tokens | old_tokens
            sim   = len(inter) / len(union)
            if sim >= threshold:
                # keep the longer (original‐cased) sentence
                if len(s) > len(kept[i]):
                    kept[i] = s
                    kept_token_sets[i] = tokens
                replaced = True
                break

        if not replaced:
            kept.append(s)
            kept_token_sets.append(tokens)

    return " ".join(kept)




def get_tokenizer(model_name: str, use_fast: bool = True):
    """
    Load and return the appropriate Hugging Face tokenizer for the given model.

    Args:
        model_name: identifier, e.g. 'google/pegasus-large'
        use_fast: whether to use the Rust-backed fast tokenizer
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading tokenizer for model: %s", model_name)

    lower = model_name.lower()
    if "pegasus" in lower:
        tokenizer = PegasusTokenizerFast.from_pretrained(model_name, use_fast=use_fast)
    elif "bart" in lower:
        tokenizer = BartTokenizerFast.from_pretrained(model_name, use_fast=use_fast)
    elif "led" in lower:
        tokenizer = LEDTokenizerFast.from_pretrained(model_name, use_fast=use_fast)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    logger.debug(
        "Loaded tokenizer class %s for model %s",
        tokenizer.__class__.__name__, model_name
    )
    return tokenizer


def generate_summary(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    config,
    device: torch.device
) -> str:
    # 1) Clean text (including <doc_sep>)
    cleaned = clean_text(text, remove_stopwords=config.data.remove_stopwords)

    # debug_logger.info(f"cleaned text: {cleaned[:10]}")

    # 2) Tokenize (pad to max_input_length for simplicity)
    inputs = tokenizer(
        cleaned,
        max_length=config.data.max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    # 3) Prepare generation kwargs
    max_len = getattr(config.generation, "max_length", config.data.max_output_length)
    gen_kwargs = {
        "max_length":            max_len,
        "min_length":            config.generation.min_length,
        "num_beams":             config.generation.num_beams,
        "length_penalty":        config.generation.length_penalty,
        "no_repeat_ngram_size":  config.generation.no_repeat_ngram_size,
        "early_stopping":        config.generation.early_stopping,
        "decoder_start_token_id": tokenizer.bos_token_id
    }

    # 4) If this is an LED model, build & include global_attention_mask
    if "led" in config.model.name.lower():
        # sep token id
        sep_id = tokenizer.convert_tokens_to_ids(DOC_SEPARATOR)
        input_ids = inputs["input_ids"]
        # mask where input_ids == sep_id, plus BOS
        gmask = torch.zeros_like(input_ids)
        gmask[input_ids == sep_id] = 1
        gmask[:, 0] = 1
        gen_kwargs["global_attention_mask"] = gmask

    # 5) Do the generation
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs
        )

    # 6) Decode and return
    return tokenizer.decode(out[0], skip_special_tokens=True)


def dict_to_namespace(d):
    """
    Recursively convert dict to SimpleNamespace
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def load_config(path: str):
    """Load YAML config file into a SimpleNamespace object."""
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return dict_to_namespace(cfg_dict)
