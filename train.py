# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import os
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=0.001, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

    def train(self, num_epochs=20, save_dir='checkpoints'):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0

            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(batch)

                # Calculate loss
                output = output.reshape(-1, output.shape[-1])
                target = batch['darija'][:, 1:].reshape(-1)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc='Validation'):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    output = self.model(batch, teacher_forcing_ratio=0.0)
                    output = output.reshape(-1, output.shape[-1])
                    target = batch['darija'][:, 1:].reshape(-1)
                    val_loss = self.criterion(output, target)
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            # Print progress
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f'New best validation loss: {best_val_loss:.4f}')
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, save_path)
            else:
                patience_counter += 1


            # Save checkpoint
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)

        return train_losses, val_losses