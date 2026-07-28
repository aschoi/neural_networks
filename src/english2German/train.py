import torch
import torch.nn as nn
import torch.optim as optim
import math
from .model import Transformer
import time
from pathlib import Path


class TransformerTrainer:
    def __init__(self, model, train_loader, validation_loader, PAD_ID, lr=1e-4, warmup_steps=4000):
        """
        Training Module for Transformer Constructor

        Args:
            model:          <model transformer>
            train_loader:   <torch.utils.data.DataLoader>
            lr:             <float>
            warmup_steps:   <int>
        """
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.best_validation_loss = float("inf")


    def get_lr_scale(self):
        """
        Learning Rate Schedule w/ Warmup

        Return:
            <float>
        """
        d_model = self.model.d_model
        step_num = self.step_num + 1

        warmup_lr_increase = step_num * (self.warmup_steps**(-1.5))
        regular_lr_decay = step_num**(-0.5)

        return d_model**(-0.5) * min(warmup_lr_increase, regular_lr_decay)


    def update_lr(self):
        """Update Learning Rate based on schedule"""
        lr_scale = self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_scale


    def evaluate_epoch(self) -> float:
        """
        Evaluate the model over the validation set.

        Returns:
            Average validation loss per batch.
        """

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.inference_mode():
            for batch in self.validation_loader:
                src = batch["src"]
                tgt_input = batch["tgt"]
                tgt_output = batch["tgt_output"]

                src_mask = self.model.create_padding_mask(src)

                tgt_causal_mask = self.model.create_causal_mask(
                    tgt_input.size(1)
                )

                tgt_padding_mask = self.model.create_padding_mask(
                    tgt_input
                )

                tgt_mask = tgt_causal_mask & tgt_padding_mask

                output = self.model(
                    src,
                    tgt_input,
                    src_mask,
                    tgt_mask,
                )

                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1),
                )

                total_loss += loss.item()
                num_batches += 1

        if num_batches == 0:
            raise ValueError("Validation loader contained no batches.")

        return total_loss / num_batches

    def train_epoch(self, epoch):
        """
        Train for one Epoch
        
        Return:
            <float>
        """

        self.model.train()
        total_loss = 0
        num_batches = 0
        start = time.perf_counter()

        for batch_index, batch in enumerate(self.train_loader):
            self.step_num += 1
            self.update_lr()

            # Parse Batch Data
            src = batch['src']
            tgt_input = batch['tgt']
            tgt_output = batch['tgt_output']
            
            # Create masks (part of training / data prep technique. basically a techinique that helps to optimize result from training)
            src_mask = self.model.create_padding_mask(src)
            tgt_causal_mask = self.model.create_causal_mask(tgt_input.size(1))
            tgt_padding_mask = self.model.create_padding_mask(tgt_input)
            # Combine masks: both must be True for attention to be allowed
            # Broadcasting can handle shape diff
            tgt_mask = tgt_causal_mask & tgt_padding_mask

            # Forward w/ Teacher Forcing
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.step_num}, Loss: {avg_loss:.4f}, lr: {lr:.6f}")



        training_loss = total_loss / num_batches

        # Run validation only after the training portion is complete.
        validation_loss = self.evaluate_epoch()
        elapsed = time.perf_counter() - start
        seconds_per_batch = elapsed / num_batches

        print(
            f"Epoch {epoch + 1} | "
            f"Training loss: {training_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        print(f"Seconds per batch: {seconds_per_batch:.3f}")
        print(f"Actual epoch time: {elapsed / 60:.1f} minutes")


        checkpoint_path = Path("checkpoints/english2German_latest.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "step_num": self.step_num,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
        }

        torch.save(checkpoint, checkpoint_path)

         # Save a separate checkpoint when validation improves.
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss

            torch.save(checkpoint, checkpoint_path)

            print(
                "Saved new best checkpoint with validation loss "
                f"{validation_loss:.4f}"
            )


        return training_loss, validation_loss