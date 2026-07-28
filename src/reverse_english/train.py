import torch
import torch.nn as nn
import torch.optim as optim
import math
from .model import Transformer


class TransformerTrainer:
    def __init__(self, model, train_loader, lr=1e-4, warmup_steps=4000):
        """
        Training Module for Transformer Constructor

        Args:
            model:          <model>
            train_loader:   <torch.utils.data.DataLoader>
            lr:             <float>
            warmup_steps:   <int>
        """
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.warmup_steps = warmup_steps
        self.step_num = 0


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


    def train_epoch(self):
        """
        Train for one Epoch
        
        Return:
            <type>
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            self.step_num += 1
            self.update_lr()

            # Parse Batch Data
            src = batch['src']      # shape: (batch_size, seq_len_q)
            tgt_input = batch['tgt']  # shape: (batch_size, seq_len_q)
            tgt_output = batch['tgt_output']  # shape: (batch_size, seq_len_q)

            # Create masks (part of training / data prep technique. basically a techinique that helps to optimize result from training)
            src_mask = self.model.create_padding_mask(src)      # shape: (batch_size, 1, 1, seq_len_q)
            tgt_causal_mask = self.model.create_causal_mask(tgt_input.size(1))       # shape: (batch_size, 1, seq_len_q, seq_len_q)
            tgt_padding_mask = self.model.create_padding_mask(tgt_input)            # shape: (batch_size, 1, 1, seq_len_q)
            # Combine masks: both must be True for attention to be allowed
            # Broadcasting can handle shape diff
            tgt_mask = tgt_causal_mask & tgt_padding_mask   # shape: (batch_size, 1, seq_len_q, seq_len_q)

            # Forward w/ Teacher Forcing
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask) # shape: (batch_size, seq_len_q, z?) (8, 15, 22?)

            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.step_num}, Loss: {avg_loss:.4f}, lr: {lr:.6f}")

        return total_loss / num_batches