import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            learning_rate=0.01,
            device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Modified loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,
            label_smoothing=0.1
        )

        # Modified optimizer settings
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.98)
        )

        # Scheduler that monitors validation loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

    def train_epoch(self, epoch, num_epochs, teacher_forcing_ratio=0.5):
        self.model.train()
        total_loss = 0

        # Decay teacher forcing ratio over time
        teacher_forcing_ratio = teacher_forcing_ratio * (1 - epoch / num_epochs)

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        for batch_idx, (src, trg) in enumerate(progress_bar):
            # Forward pass
            output = self.model(src, trg, teacher_forcing_ratio)
            output = output.view(-1, output.shape[-1])
            trg = trg.view(-1)

            # Calculate loss
            loss = self.criterion(output, trg)

            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for src, trg in tqdm(self.val_loader, desc='Validating'):
                output = self.model(src, trg, teacher_forcing_ratio=0)
                output = output.view(-1, output.shape[-1])
                trg = trg.view(-1)
                loss = self.criterion(output, trg)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, num_epochs, save_dir='checkpoints'):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(epoch, num_epochs)
            train_losses.append(train_loss)

            # Validation
            val_loss = self.validate()
            val_losses.append(val_loss)

            # Scheduler step with validation loss
            self.scheduler.step(val_loss)

            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, f'{save_dir}/best_model.pt')

            print('-' * 50)

        return train_losses, val_losses