import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from typing import Dict, List, Union, Tuple
import random
import logging

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)


class MultiNewsDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            split: str,
            max_input_length: int,
            max_output_length: int,
            sample_ratio: float = 1.0,
            seed: int = 42
    ):
        """
        Initialize Multi-News dataset

        Args:
            tokenizer: Tokenizer from Hugging Face
            split: Dataset split (train, validation, test)
            max_input_length: Maximum input length
            max_output_length: Maximum output length
            sample_ratio: Ratio of dataset to use (0.0-1.0)
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        logger.info(f"Loading {split} split of the multi_news dataset")
        dataset = load_dataset("alexfabbri/multi_news", split=split)

        # Sampling for faster development cycles
        if sample_ratio < 1.0:
            random.seed(seed)
            total_samples = int(len(dataset) * sample_ratio)
            indices = random.sample(range(len(dataset)), total_samples)
            dataset = dataset.select(indices)
            logger.info(f"Sampled {len(dataset)} examples from {split} split (ratio={sample_ratio})")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Extract document and summary
        documents = example["document"]
        summary = example["summary"]

        # Clean documents and summary
        documents = self._clean_text(documents)
        summary = self._clean_text(summary)

        # Tokenize inputs and outputs
        inputs = self._prepare_inputs(documents)
        labels = self._prepare_labels(summary)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels["input_ids"],
        }

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove multiple spaces
        text = ' '.join(text.split())
        # Remove special HTML characters
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        return text

    def _prepare_inputs(self, documents: str) -> Dict:
        """Process and tokenize input documents"""
        # Truncation will happen at tokenization level
        return self.tokenizer(
            documents,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def _prepare_labels(self, summary: str) -> Dict:
        """Process and tokenize target summary"""
        return self.tokenizer(
            summary,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


def get_dataloaders(
        tokenizer,
        config
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test data loaders

    Args:
        tokenizer: Tokenizer
        config: Configuration

    Returns:
        Tuple of train, validation, and test dataloaders
    """
    train_dataset = MultiNewsDataset(
        tokenizer=tokenizer,
        split="train",
        max_input_length=config.data.max_input_length,
        max_output_length=config.data.max_output_length,
        sample_ratio=config.data.sample_ratio,
        seed=config.data.seed
    )

    val_dataset = MultiNewsDataset(
        tokenizer=tokenizer,
        split="validation",
        max_input_length=config.data.max_input_length,
        max_output_length=config.data.max_output_length,
        sample_ratio=config.data.sample_ratio,
        seed=config.data.seed
    )

    test_dataset = MultiNewsDataset(
        tokenizer=tokenizer,
        split="test",
        max_input_length=config.data.max_input_length,
        max_output_length=config.data.max_output_length,
        sample_ratio=config.data.sample_ratio,
        seed=config.data.seed
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader