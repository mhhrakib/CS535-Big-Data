# src/collator.py

import logging
import torch
from transformers import DataCollatorForSeq2Seq

from src.utils import DOC_SEPARATOR

logger = logging.getLogger(__name__)

class LEDDataCollator:
    """
    Wraps the standard Seq2Seq collator to:
      1) dynamically pad inputs + labels
      2) set labels’ pad token to -100
      3) inject global_attention_mask for LED models
      4) sanity‐check that <doc_sep> is in the tokenizer vocab
    """
    def __init__(self, tokenizer, model_name: str):
        # base collator: dynamic padding + label masking
        self.base = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
        )
        self.tokenizer = tokenizer
        self.is_led = "led" in model_name.lower()

        # Sanity check for <doc_sep>
        if DOC_SEPARATOR not in tokenizer.get_vocab():
            logger.warning(
                "Special token <doc_sep> not found in tokenizer vocabulary. "
                "Summaries may not respect document boundaries."
            )

    def __call__(self, features):
        # Let the base collator pad input_ids, attention_mask, and labels
        batch = self.base(features)

        # If LED, add global_attention_mask for every <doc_sep> (and BOS)
        if self.is_led:
            sep_id = self.tokenizer.convert_tokens_to_ids(DOC_SEPARATOR)
            input_ids = batch['input_ids']
            bs, seq_len = input_ids.shape

            # mask positions equal to sep_id or position 0 (BOS)
            gmask = torch.zeros(bs, seq_len, dtype=torch.long, device=input_ids.device)
            gmask[input_ids == sep_id] = 1
            gmask[:, 0] = 1
            batch['global_attention_mask'] = gmask

        return batch
