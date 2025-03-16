import torch
from torch.utils.data import DataLoader
from torch import nn

from sklearn.preprocessing import StandardScaler

import numpy as np 
from typing import List, Tuple

from transformer_module import TimeSeriesDataset, TimeSeriesTransformer, train_model, plot_losses, save_model, load_model

# Configuration
ENC_SEQ_LEN = 24
DEC_SEQ_LEN = 12
TARGET_SEQ_LEN = 1
BATCH_SIZE = 64
EPOCHS = 50
DIM_VAL = 128
N_HEADS = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data preparation
def prepare_data(test_size: float=0.2, val_size: float=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    t = np.arange(0, 1000, 0.1)
    data = np.sin(t) + 0.1 * np.random.randn(len(t))
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data.reshape(-1, 1))
    
    total_size = len(data)
    train_size = int(total_size * (1 - test_size - val_size))
    val_size = int(total_size * val_size)
    
    return (data[:train_size],
            data[train_size:train_size+val_size],
            data[train_size+val_size:],
            scaler)

train_data, val_data, test_data, scaler = prepare_data()

# Create datasets
train_dataset = TimeSeriesDataset(train_data, ENC_SEQ_LEN, DEC_SEQ_LEN, TARGET_SEQ_LEN)
val_dataset = TimeSeriesDataset(val_data, ENC_SEQ_LEN, DEC_SEQ_LEN, TARGET_SEQ_LEN)
test_dataset = TimeSeriesDataset(test_data, ENC_SEQ_LEN, DEC_SEQ_LEN, TARGET_SEQ_LEN)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# 2. Training & Validation ====================================================

# Initialize model
model_path = None
model = TimeSeriesTransformer(
        input_size=1,
        dec_seq_len=DEC_SEQ_LEN,
        batch_first=True,
        dim_val=DIM_VAL,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_heads=N_HEADS,
        pretrained_model_path = model_path # 記得要加
    ).to(device) 

criterion = nn.MSELoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
trained_model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Plot losses
plot_losses(train_losses, val_losses)

# # Inference
# inference(trained_model, test_data, ENC_SEQ_LEN, DEC_SEQ_LEN, scaler, device=device)


