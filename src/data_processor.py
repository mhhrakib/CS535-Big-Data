# src/data_processor.py

import logging
import random

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset

from src.utils import clean_text, dedup_sentences_fuzzy
from src.collator import LEDDataCollator

logger = logging.getLogger(__name__)

class MultiNewsDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        sample_ratio: float = 1.0,
        seed: int = 42,
        remove_stopwords: bool = False,
        dedup: bool = True

    ):
        """
        A PyTorch Dataset for Multi-News.

        Tokenization is done without padding; padding & label‐masking
        will be handled by our DataCollator.

        Args:
            tokenizer: Hugging Face tokenizer
            split: 'train', 'validation', or 'test'
            max_input_length: max encoder tokens
            max_output_length: max decoder tokens
            sample_ratio: fraction of data to sample (for quick runs)
            seed: random seed for sampling
            remove_stopwords: whether to strip stopwords in cleaning
            dedup: whether to remove duplicate sentences across documents
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.remove_stopwords = remove_stopwords
        self.dedup = dedup


        logger.info(f"Loading '{split}' split of Multi-News")
        ds = load_dataset("alexfabbri/multi_news", split=split)

        if 0.0 < sample_ratio < 1.0:
            random.seed(seed)
            n = int(len(ds) * sample_ratio)
            idxs = random.sample(range(len(ds)), n)
            ds = ds.select(idxs)
            logger.info(f"Sampled {n}/{len(ds)} examples ({sample_ratio*100:.1f}%)")

        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        doc  = clean_text(ex["document"], remove_stopwords=self.remove_stopwords)
        summ = clean_text(ex["summary"],  remove_stopwords=self.remove_stopwords)

        if self.dedup:
            doc = dedup_sentences_fuzzy(doc)

        # tokenize without padding; collator will pad dynamically
        enc = self.tokenizer(
            doc,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
            return_attention_mask=True
        )
        dec = self.tokenizer(
            summ,
            max_length=self.max_output_length,
            truncation=True,
            padding=False
        )

        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels":         dec["input_ids"]
        }


def get_dataloaders(config, tokenizer, ddp: bool = False):
    """
    Build train / val / test DataLoaders with dynamic padding,
    LED global attention, and DDP support.

    Args:
        config: object with .data and .training and .model.name
        tokenizer: HF tokenizer matching the model
        ddp: whether to use DistributedSampler

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config.data
    seed = config.training.seed

    # pull dedup flag via getattr, with default False
    dedup_flag = getattr(data_cfg, "dedup", False)

    # instantiate datasets
    train_ds = MultiNewsDataset(
        tokenizer,
        split="train",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=seed,
        remove_stopwords=data_cfg.remove_stopwords,
        dedup=dedup_flag
    )

    val_ds = MultiNewsDataset(
        tokenizer,
        split="validation",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=seed,
        remove_stopwords=data_cfg.remove_stopwords,
        dedup=dedup_flag
    )

    test_ds = MultiNewsDataset(
        tokenizer,
        split="test",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=1.0,
        seed=seed,
        remove_stopwords=data_cfg.remove_stopwords,
        dedup=False
    )

    # prepare the LED-aware collator (handles padding, label-masking, global attention)
    collator = LEDDataCollator(tokenizer, model_name=config.model.name)

    def make_loader(dataset, shuffle: bool):
        sampler = None
        if ddp:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=True,
            collate_fn=collator
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)
    test_loader  = make_loader(test_ds,  shuffle=False)

    return train_loader, val_loader, test_loader
