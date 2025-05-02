### data_processor.py (updated for external tokenizer)
import logging
import random
from torch.utils.data import Dataset, DataLoader
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
        seed: int = 42
    ):
        """
        Args:
            tokenizer: Pretrained Hugging Face tokenizer instance
            split: One of "train", "validation", "test"
            max_input_length: max tokens for the document
            max_output_length: max tokens for the summary
            sample_ratio: fraction of data to sample
            seed: random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        logger.info(f"Loading '{split}' split of Multi-News")
        dataset = load_dataset("alexfabbri/multi_news", split=split)

        if sample_ratio < 1.0:
            random.seed(seed)
            n = int(len(dataset) * sample_ratio)
            indices = random.sample(range(len(dataset)), n)
            dataset = dataset.select(indices)
            logger.info(f"Sampled {n} examples (ratio={sample_ratio})")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        doc = clean_text(ex["document"])
        summ = clean_text(ex["summary"])

        inputs = self.tokenizer(
            doc,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            summ,
            max_length=self.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.input_ids.squeeze(0),
        }


def get_dataloaders(
        config,
        tokenizer
) -> tuple:
    """
    Creates train, validation, and test data loaders

    Args:
        config: Configuration object with model and data fields
        tokenizer: Pretrained Hugging Face tokenizer instance
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    data_cfg = config.data

    train_dataset = MultiNewsDataset(
        tokenizer=tokenizer,
        split="train",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=data_cfg.seed
    )

    val_dataset = MultiNewsDataset(
        tokenizer=tokenizer,
        split="validation",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=data_cfg.seed
    )

    test_dataset = MultiNewsDataset(
        tokenizer=tokenizer,
        split="test",
        max_input_length=data_cfg.max_input_length,
        max_output_length=data_cfg.max_output_length,
        sample_ratio=data_cfg.sample_ratio,
        seed=data_cfg.seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
    )

    return train_dataloader, val_dataloader, test_dataloader



# import torch
# from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset
# import nltk
# from nltk.tokenize import sent_tokenize
# import numpy as np
# from typing import Dict, List, Union, Tuple
# import random
# import logging

# # Set up logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)

# # Download NLTK resources
# nltk.download('punkt', quiet=True)


# class MultiNewsDataset(Dataset):
#     def __init__(
#             self,
#             tokenizer,
#             split: str,
#             max_input_length: int,
#             max_output_length: int,
#             sample_ratio: float = 1.0,
#             seed: int = 42
#     ):
#         """
#         Initialize Multi-News dataset

#         Args:
#             tokenizer: Tokenizer from Hugging Face
#             split: Dataset split (train, validation, test)
#             max_input_length: Maximum input length
#             max_output_length: Maximum output length
#             sample_ratio: Ratio of dataset to use (0.0-1.0)
#             seed: Random seed for reproducibility
#         """
#         self.tokenizer = tokenizer
#         self.max_input_length = max_input_length
#         self.max_output_length = max_output_length

#         logger.info(f"Loading {split} split of the multi_news dataset")
#         dataset = load_dataset("alexfabbri/multi_news", split=split)

#         # Sampling for faster development cycles
#         if sample_ratio < 1.0:
#             random.seed(seed)
#             total_samples = int(len(dataset) * sample_ratio)
#             indices = random.sample(range(len(dataset)), total_samples)
#             dataset = dataset.select(indices)
#             logger.info(f"Sampled {len(dataset)} examples from {split} split (ratio={sample_ratio})")

#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         example = self.dataset[idx]

#         # Extract document and summary
#         documents = example["document"]
#         summary = example["summary"]

#         # Clean documents and summary
#         documents = self._clean_text(documents)
#         summary = self._clean_text(summary)

#         # Tokenize inputs and outputs
#         inputs = self._prepare_inputs(documents)
#         labels = self._prepare_labels(summary)

#         return {
#             "input_ids": inputs["input_ids"],
#             "attention_mask": inputs["attention_mask"],
#             "labels": labels["input_ids"],
#         }

#     def _clean_text(self, text: str) -> str:
#         """Basic text cleaning"""
#         # Remove multiple spaces
#         text = ' '.join(text.split())
#         # Remove special HTML characters
#         text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
#         return text

#     def _prepare_inputs(self, documents: str) -> Dict:
#         """Process and tokenize input documents"""
#         # Truncation will happen at tokenization level
#         return self.tokenizer(
#             documents,
#             max_length=self.max_input_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )

#     def _prepare_labels(self, summary: str) -> Dict:
#         """Process and tokenize target summary"""
#         return self.tokenizer(
#             summary,
#             max_length=self.max_output_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt",
#         )


# def get_dataloaders(
#         tokenizer,
#         config
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Creates train, validation, and test data loaders

#     Args:
#         tokenizer: Tokenizer
#         config: Configuration

#     Returns:
#         Tuple of train, validation, and test dataloaders
#     """
#     train_dataset = MultiNewsDataset(
#         tokenizer=tokenizer,
#         split="train",
#         max_input_length=config.data.max_input_length,
#         max_output_length=config.data.max_output_length,
#         sample_ratio=config.data.sample_ratio,
#         seed=config.data.seed
#     )

#     val_dataset = MultiNewsDataset(
#         tokenizer=tokenizer,
#         split="validation",
#         max_input_length=config.data.max_input_length,
#         max_output_length=config.data.max_output_length,
#         sample_ratio=config.data.sample_ratio,
#         seed=config.data.seed
#     )

#     test_dataset = MultiNewsDataset(
#         tokenizer=tokenizer,
#         split="test",
#         max_input_length=config.data.max_input_length,
#         max_output_length=config.data.max_output_length,
#         sample_ratio=config.data.sample_ratio,
#         seed=config.data.seed
#     )

#     # Create dataloaders
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=True,
#         num_workers=config.data.num_workers,
#     )

#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=False,
#         num_workers=config.data.num_workers,
#     )

#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=config.data.batch_size,
#         shuffle=False,
#         num_workers=config.data.num_workers,
#     )

#     return train_dataloader, val_dataloader, test_dataloader