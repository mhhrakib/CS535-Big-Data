import ftfy
import unicodedata
import re
import contractions
from transformers import PegasusTokenizerFast, BartTokenizerFast, LEDTokenizerFast, AutoTokenizer


def clean_text(text: str) -> str:
    """
    Expand contractions, fix common encoding glitches, normalize unicode,
    remove control characters, collapse long punctuation runs,
    and strip non-ASCII characters.
    """
    # 1) Fix mojibake (encoding glitches)
    text = ftfy.fix_text(text)

    # 2) Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)

    # 3) Normalize unicode to NFKC form
    text = unicodedata.normalize("NFKC", text)

    # 4) Remove control characters, newlines, and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # 5) Collapse runs of unwanted punctuation (e.g., ===, ###, ```) into single space
    text = re.sub(r"([=#`\\\\])\1{2,}", " ", text)

    # 6) Optionally drop non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 7) Collapse multiple spaces into one
    text = " ".join(text.split())

    return text


def get_tokenizer(model_name: str, use_fast: bool = True):
    """
    Return a pretrained Hugging Face tokenizer for the given model name.
    Supports PEGASUS, BART, LED, or any other model via AutoTokenizer.
    """
    if "pegasus" in model_name.lower():
        return PegasusTokenizerFast.from_pretrained(model_name, use_fast=use_fast)
    if "bart" in model_name.lower():
        return BartTokenizerFast.from_pretrained(model_name, use_fast=use_fast)
    if "led" in model_name.lower():
        return LEDTokenizerFast.from_pretrained(model_name, use_fast=use_fast)
    # Fallback to AutoTokenizer for other models
    return AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
