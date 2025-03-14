'''
底下的程式碼，經過colab測試，是成功的
'''
import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Any
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. Transformer Model ========================================================

class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).to(device) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device) 
        pe = torch.zeros(max_len, 1, d_model).to(device) 
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
        input_size: int,
        dec_seq_len: int,
        batch_first: bool,
        out_seq_len: int=58,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1
        ): 
        super().__init__()
        self.dec_seq_len = dec_seq_len
        self.encoder_input_layer = nn.Linear(input_size, dim_val).to(device) 
        self.decoder_input_layer = nn.Linear(num_predicted_features, dim_val).to(device) 
        self.linear_mapping = nn.Linear(dim_val, num_predicted_features).to(device) 
        self.positional_encoding_layer = PositionalEncoder(dim_val, dropout_pos_enc)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            dim_val, n_heads, dim_feedforward_encoder, dropout_encoder, batch_first=batch_first).to(device) 
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers).to(device) 
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            dim_val, n_heads, dim_feedforward_decoder, dropout_decoder, batch_first=batch_first).to(device) 
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers).to(device) 

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # Encoder processing
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        memory = self.encoder(src)
        
        # Decoder processing
        tgt = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(tgt, memory)
        return self.linear_mapping(decoder_output)


# 2. Data Handling ============================================================   

class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, enc_seq_len: int, dec_seq_len: int, target_seq_len: int):
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.target_seq_len = target_seq_len

        # Precompute valid indices to avoid size mismatch
        self.valid_indices = []
        for idx in range(len(self.data) - self.enc_seq_len - self.target_seq_len - self.dec_seq_len + 1):
            self.valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        real_idx = self.valid_indices[idx]  # Use precomputed index

        enc_start = real_idx
        enc_end = enc_start + self.enc_seq_len
        dec_end = enc_end + self.dec_seq_len
        target_end = dec_end + self.target_seq_len
        
        encoder_input = self.data[enc_start:enc_end]
        decoder_input = self.data[enc_end:dec_end]
        target = self.data[dec_end:target_end]
        
        return (torch.FloatTensor(encoder_input).to(device) ,
            torch.FloatTensor(decoder_input).to(device) ,
            torch.FloatTensor(target).to(device)) 


# 3. Training & Validation ====================================================

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

# 4. Data Preparation =========================================================

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

# 5. Visualization ================================================

def plot_losses(train_losses: List[float], val_losses: List[float]) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training/Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


ENC_SEQ_LEN = 24
DEC_SEQ_LEN = 12
TARGET_SEQ_LEN = 1
BATCH_SIZE = 64
EPOCHS = 50
DIM_VAL = 128
N_HEADS = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2

# Data preparation
train_data, val_data, test_data, scaler = prepare_data()

# Create datasets
train_dataset = TimeSeriesDataset(train_data, ENC_SEQ_LEN, DEC_SEQ_LEN, TARGET_SEQ_LEN)
val_dataset = TimeSeriesDataset(val_data, ENC_SEQ_LEN, DEC_SEQ_LEN, TARGET_SEQ_LEN)
test_dataset = TimeSeriesDataset(test_data, ENC_SEQ_LEN, DEC_SEQ_LEN, TARGET_SEQ_LEN)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
model_path = ""
model = TimeSeriesTransformer(
        input_size=1,
        dec_seq_len=DEC_SEQ_LEN,
        batch_first=True,
        dim_val=DIM_VAL,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_heads=N_HEADS
        pretrained_model_path = model_path # 記得要加
    ).to(device) 

criterion = nn.MSELoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train model
trained_model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Plot losses
plot_losses(train_losses, val_losses)

# After training:
model_path = "time_series_transformer.pth"
torch.save(trained_model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Later, when you want to load the model:
loaded_model = TimeSeriesTransformer(
        input_size=1,
        dec_seq_len=DEC_SEQ_LEN,
        batch_first=True,
        dim_val=DIM_VAL,
        n_encoder_layers=N_ENCODER_LAYERS,
        n_decoder_layers=N_DECODER_LAYERS,
        n_heads=N_HEADS
    ).to(device) #Create a new instance

loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.eval()


# 6. Inferencing ================================================

def inference(model: nn.Module, test_data: np.ndarray, enc_seq_len: int, 
            dec_seq_len: int, scaler: StandardScaler) -> None:
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(len(test_data) - enc_seq_len - dec_seq_len):
            enc_input = test_data[i:i+enc_seq_len]
            dec_input = test_data[i+enc_seq_len:i+enc_seq_len+dec_seq_len]
            
            enc_tensor = torch.FloatTensor(enc_input).unsqueeze(0).to(device)
            dec_tensor = torch.FloatTensor(dec_input).unsqueeze(0).to(device)
            
            output = model(enc_tensor, dec_tensor)
            predictions.append(output.cpu().detach().numpy().flatten())
            actuals.append(test_data[i+enc_seq_len+dec_seq_len:i+enc_seq_len+dec_seq_len+1, 0])  # Modified
                
    # Convert to numpy arrays and reshape
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    # Inverse transform
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.title('Time Series Forecasting Results')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


inference(loaded_model, test_data, ENC_SEQ_LEN, DEC_SEQ_LEN, scaler)



# 7. Tensorboard ================================================

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

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

        # Log training loss
        train_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        # Log validation loss
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

    return model, train_losses, val_losses


# Train model
trained_model, train_losses, val_losses = train_model(
    model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# During training, open the terminal and run the below command:
# tensorboard --logdir=runs
# Then navigate to http://localhost:6006/ to view TensorBoard dashboard.

