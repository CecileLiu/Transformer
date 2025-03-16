
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Any

import torch 

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


def save_model(trained_model, model_path:str):
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path:str, 
                input_size:int, 
                dec_seq_len:int, 
                batch_first:bool,
                dim_val:int,
                n_encoder_layers:int=2,
                n_decoder_layers:int=2,
                n_heads:int=4,
                device:str='cuda'):

    loaded_model = TimeSeriesTransformer(
        input_size=1,
        dec_seq_len=dec_seq_len,
        batch_first=batch_first,
        dim_val=dim_val,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads
    ).to(device) #Create a new instance

    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()
    return loaded_model
