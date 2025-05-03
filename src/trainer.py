# src/trainer.py

import os
import logging
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from src.dist_utils import is_main_process
from tqdm import tqdm

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_loader,
        val_loader,
        config,
        device,
        ddp: bool = False
    ):
        """
        Trainer for Seq2Seq models with optional DDP support and tqdm progress bars.

        Args:
            model: nn.Module (possibly wrapped in DDP)
            tokenizer: Hugging Face tokenizer
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            config: configuration object
            device: torch.device
            ddp: whether using DistributedDataParallel
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.ddp = ddp

        # Set seeds for reproducibility
        seed = int(config.training.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Parse training hyperparameters
        epochs = int(config.training.epochs)
        lr = float(config.training.lr)
        weight_decay = float(getattr(config.training, 'weight_decay', 0.0))
        warmup_steps = int(getattr(config.training, 'warmup_steps', 0))
        self.log_interval = int(getattr(config.training, 'log_interval', 100))
        self.use_amp = bool(getattr(config.training, 'fp16', False))

        # Optimizer and scheduler
        t_total = len(self.train_loader) * epochs
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total
        )

        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Prepare output directory
        self.output_dir = config.output.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self):
        best_val_loss = float('inf')
        best_epoch = 0
        epochs = int(self.config.training.epochs)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            if is_main_process():
                logger.info(
                    f"Epoch {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    self.save_checkpoint('best_model')
                    logger.info(f"New best model saved at epoch {epoch}")

        if is_main_process():
            logger.info(f"Training complete. Best epoch: {best_epoch} with loss {best_val_loss:.4f}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} Training",
            total=len(self.train_loader),
            disable=not is_main_process()
        )

        for step, batch in enumerate(progress_bar, start=1):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            total_loss += loss.item()

            if is_main_process() and step % self.log_interval == 0:
                logger.info(f"Epoch {epoch} Step {step}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    def validate_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch} Validation",
            total=len(self.val_loader),
            disable=not is_main_process()
        )

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, name: str):
        """
        Save the current (best) model and tokenizer to the output directory.

        Args:
            name: subdirectory name under `output_dir` where to save
        """
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # If using DDP, unwrap the model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
