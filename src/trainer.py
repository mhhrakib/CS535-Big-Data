import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
import evaluate
from typing import Dict, List, Tuple, Any, Optional

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Load evaluation metrics
rouge_metric = evaluate.load('rouge')
bertscore_metric = evaluate.load('bertscore')


class Trainer:
    def __init__(
            self,
            model,
            tokenizer,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            config
    ):
        """
        Initializes the trainer

        Args:
            model: The model to train
            tokenizer: The tokenizer
            optimizer: The optimizer
            scheduler: The learning rate scheduler
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            test_dataloader: Test data loader
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up distributed training if required
        self.setup_distributed()

        # Move model to device
        self.model = self.model.to(self.device)

        # Set up DDP if distributed
        if self.config.training.distributed and self.config.training.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )

        # Set up mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.training.use_fp16 else None

        # Setup tensorboard for logging
        self.writer = None
        if self.is_main_process():
            log_dir = os.path.join(self.config.training.output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

    def setup_distributed(self):
        """Sets up distributed training if enabled"""
        if not self.config.training.distributed:
            self.local_rank = 0
            return

        # Check if environment variables are set
        if "LOCAL_RANK" not in os.environ:
            self.config.training.distributed = False
            self.local_rank = 0
            logger.warning("LOCAL_RANK not found in environment variables. Disabling distributed training.")
            return

        # Initialize distributed process group
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl")

        # Update world size
        self.config.training.world_size = dist.get_world_size()
        logger.info(f"Initialized distributed training with world_size: {self.config.training.world_size}")

    def is_main_process(self):
        """Checks if this is the main process"""
        if not self.config.training.distributed:
            return True
        return dist.get_rank() == 0

    def train(self):
        """Trains the model"""
        # Get total number of training steps
        total_steps = len(self.train_dataloader) * self.config.training.num_epochs

        # Training loop
        global_step = 0
        best_val_rouge = 0.0
        train_losses = []

        for epoch in range(self.config.training.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")

            # Training for one epoch
            self.model.train()
            epoch_loss = 0.0
            steps_in_epoch = 0

            progress_bar = tqdm(self.train_dataloader,
                                desc=f"Epoch {epoch + 1}") if self.is_main_process() else self.train_dataloader

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.squeeze(1).to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                         batch.items()}

                # Forward pass with mixed precision if enabled
                if self.config.training.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss = outputs.loss / self.config.model.gradient_accumulation_steps

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()

                    if (step + 1) % self.config.model.gradient_accumulation_steps == 0:
                        # Clip gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.model.max_grad_norm
                        )

                        # Update weights
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1
                else:
                    # Standard training without mixed precision
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.model.gradient_accumulation_steps
                    loss.backward()

                    if (step + 1) % self.config.model.gradient_accumulation_steps == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.model.max_grad_norm
                        )

                        # Update weights
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        global_step += 1

                # Update loss tracking
                epoch_loss += loss.item() * self.config.model.gradient_accumulation_steps
                steps_in_epoch += 1

                # Update progress bar
                if self.is_main_process():
                    progress_bar.set_postfix({"loss": loss.item() * self.config.model.gradient_accumulation_steps})

                # Log to tensorboard
                if self.is_main_process() and global_step % self.config.training.logging_steps == 0:
                    self.writer.add_scalar("train/loss", loss.item() * self.config.model.gradient_accumulation_steps,
                                           global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], global_step)

                # Evaluate and save checkpoint
                if global_step % self.config.training.eval_steps == 0:
                    # Evaluate
                    val_results = self.evaluate(self.val_dataloader)

                    # Log validation results
                    if self.is_main_process():
                        for metric, value in val_results.items():
                            self.writer.add_scalar(f"val/{metric}", value, global_step)

                        # Save model if it's the best so far
                        val_rouge_avg = (val_results["rouge1"] + val_results["rouge2"] + val_results["rougeL"]) / 3
                        if val_rouge_avg > best_val_rouge:
                            best_val_rouge = val_rouge_avg
                            self.save_checkpoint(f"best_model_{epoch + 1}_{global_step}")

                # Save regular checkpoint
                if global_step % self.config.training.save_steps == 0 and self.is_main_process():
                    self.save_checkpoint(f"checkpoint_{epoch + 1}_{global_step}")

            # End of epoch
            epoch_loss /= steps_in_epoch
            train_losses.append(epoch_loss)

            if self.is_main_process():
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss:.4f}")
                self.writer.add_scalar("train/epoch_loss", epoch_loss, epoch + 1)

        # End of training
        if self.is_main_process():
            logger.info("Training completed!")
            self.save_checkpoint("final_model")

            # Final evaluation on test set
            logger.info("Running final evaluation on test set")
            test_results = self.evaluate(self.test_dataloader)

            for metric, value in test_results.items():
                logger.info(f"Test {metric}: {value:.4f}")
                self.writer.add_scalar(f"test/{metric}", value, 0)

    def evaluate(self, dataloader):
        """
        Evaluates the model on the given dataloader

        Args:
            dataloader: Data loader for evaluation

        Returns:
            results: Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", disable=not self.is_main_process()):
                # Move batch to device
                batch = {k: v.squeeze(1).to(self.device) if isinstance(v, torch.Tensor) else v for k, v in
                         batch.items()}

                # Generate summaries
                outputs = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=self.config.data.max_output_length,
                    num_beams=4,
                    early_stopping=True
                )

                # Decode predictions and labels
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                labels = self.tokenizer.batch_decode(
                    batch["labels"],
                    skip_special_tokens=True
                )

                # Store for metrics calculation
                all_preds.extend(preds)
                all_labels.extend(labels)

        # Calculate metrics
        results = {}

        # ROUGE scores
        rouge_output = rouge_metric.compute(
            predictions=all_preds,
            references=all_labels,
            use_stemmer=True
        )

        results["rouge1"] = rouge_output["rouge1"]
        results["rouge2"] = rouge_output["rouge2"]
        results["rougeL"] = rouge_output["rougeL"]

        # BERTScore (if needed - commented out due to computational intensity)
        # bertscore_output = bertscore_metric.compute(
        #     predictions=all_preds,
        #     references=all_labels,
        #     lang="en"
        # )
        # results["bertscore"] = sum(bertscore_output["f1"]) / len(bertscore_output["f1"])

        # Log results
        if self.is_main_process():
            logger.info(f"Evaluation results: {results}")

        return results

    def save_checkpoint(self, checkpoint_name):
        """
        Saves a checkpoint of the model

        Args:
            checkpoint_name: Name for the checkpoint
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.config.training.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(checkpoint_dir)

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)

        logger.info(f"Model and tokenizer saved to {checkpoint_dir}")