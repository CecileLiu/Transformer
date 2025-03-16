import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Tuple, List, Optional, Any


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                num_epochs: int) -> Tuple[nn.Module, List[float], List[float]]:
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for enc_input, dec_input, target in train_loader:
            # enc_input = enc_input.to(device)
            # dec_input = dec_input.to(device)
            # target = target.to(device)
            optimizer.zero_grad()
            output = model(enc_input, dec_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for enc_input, dec_input, target in val_loader:
                output = model(enc_input, dec_input)
                val_loss += criterion(output, target).item()
        
        # Record metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

    return model, train_losses, val_losses


