# src/data_processor.py

import logging
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from src.utils import clean_text

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
        remove_stopwords: bool = False
    ):
        """
        A PyTorch dataset for the Multi-News summarization task.

        Args:
            tokenizer: Hugging Face tokenizer instance
            split: Dataset split ('train', 'validation', 'test')
            max_input_length: Max tokens for input documents
            max_output_length: Max tokens for target summaries
            sample_ratio: Fraction of data to sample for quick experiments
            seed: Random seed for sampling
            remove_stopwords: Whether to remove stopwords during cleaning
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.remove_stopwords = remove_stopwords

        logger.info(f"Loading '{split}' split of Multi-News")
        dataset = load_dataset("alexfabbri/multi_news", split=split)

        if sample_ratio < 1.0:
            random.seed(seed)
            n_samples = int(len(dataset) * sample_ratio)
            indices = random.sample(range(len(dataset)), n_samples)
            dataset = dataset.select(indices)
            logger.info(f"Sampled {n_samples}/{len(dataset)} examples (ratio={sample_ratio})")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        document = clean_text(example["document"], remove_stopwords=self.remove_stopwords)
        summary = clean_text(example["summary"], remove_stopwords=self.remove_stopwords)

        inputs = self.tokenizer(
            document,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = self.tokenizer(
            summary,
            max_length=self.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.input_ids.squeeze(0)
        }


def get_dataloaders(config, tokenizer, ddp: bool = False):
    """
    Build train, validation, and test DataLoaders with optional DDP support.

    Args:
        config: Configuration object with 'data' and 'training' fields
        tokenizer: Tokenizer instance
        ddp: Whether to use DistributedSampler

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_cfg = config.data
    seed = config.training.seed

    # Create datasets
    train_ds = MultiNewsDataset(
        tokenizer=tokenizer,
        split="train",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=seed,
        remove_stopwords=data_cfg.remove_stopwords
    )
    val_ds = MultiNewsDataset(
        tokenizer=tokenizer,
        split="validation",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=seed,
        remove_stopwords=data_cfg.remove_stopwords
    )
    test_ds = MultiNewsDataset(
        tokenizer=tokenizer,
        split="test",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=1.0,
        seed=seed,
        remove_stopwords=data_cfg.remove_stopwords
    )

    # Create samplers
    if ddp:
        train_sampler = DistributedSampler(train_ds)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        test_sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    # Build DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=data_cfg.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
