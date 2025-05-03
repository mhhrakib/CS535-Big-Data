import logging
import ftfy
import unicodedata
import re
import contractions
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from typing import Optional, Set
from transformers import PegasusTokenizerFast, BartTokenizerFast, LEDTokenizerFast, AutoTokenizer

import torch

# Ensure NLTK resources are available
nltk.download('stopwords', quiet=True)

# Configure module-level logger
debug_logger = logging.getLogger(__name__)


def clean_text(
    text: str,
    remove_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None
) -> str:
    """
    Perform extensive text cleaning:
    1) Fix mojibake and encoding issues
    2) Expand contractions (e.g., "don't" -> "do not")
    3) Normalize unicode (NFKC)
    4) Remove HTML tags
    5) Remove control characters (newlines, tabs)
    6) Collapse repeated punctuation runs (e.g., "====", "###")
    7) Remove non-ASCII characters
    8) Remove special characters, keeping only alphanumeric and basic punctuation
    9) (Optional) Remove stopwords using NLTK list
    10) Collapse whitespace
    """
    debug_logger.debug("Starting clean_text. Original length: %d", len(text))

    # 1) Fix mojibake
    text = ftfy.fix_text(text)

    # 2) Expand contractions
    text = contractions.fix(text)

    # 3) Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # 4) Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 5) Remove control characters
    text = re.sub(r"[\r\n\t]+", " ", text)

    # 6) Collapse runs of punctuation (e.g., ====, ###, ```) -> single space
    text = re.sub(r"([=#`\\])\1{2,}", " ", text)

    # 7) Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 8) Remove special characters except basic punctuation
    #    Keep letters, numbers, whitespace, and . , ! ? : ; ( ) ' -
    text = re.sub(r"[^A-Za-z0-9\s\.,!\?:;()'\-]", " ", text)

    # 9) Optional stopword removal
    if remove_stopwords:
        if stopwords is None:
            stopwords = set(nltk_stopwords.words('english'))
        debug_logger.debug("Removing stopwords: total before = %d", len(text.split()))
        tokens = [tok for tok in text.split() if tok.lower() not in stopwords]
        text = " ".join(tokens)
        debug_logger.debug("After stopword removal: total = %d", len(tokens))

    # 10) Collapse multiple spaces
    text = " ".join(text.split())
    debug_logger.debug("Finished clean_text. Final length: %d", len(text))
    return text


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
    cleaned = clean_text(text, remove_stopwords=config.data.remove_stopwords)
    inputs = tokenizer(
        cleaned,
        max_length=config.data.max_input_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    gen_kwargs = {
        "max_length":            config.data.max_output_length,
        "min_length":            config.generation.min_length,
        "num_beams":             config.generation.num_beams,
        "length_penalty":        config.generation.length_penalty,
        "no_repeat_ngram_size":  config.generation.no_repeat_ngram_size,
        "early_stopping":        config.generation.early_stopping,
        "decoder_start_token_id": tokenizer.bos_token_id
    }

    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)