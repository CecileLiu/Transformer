import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Any


def inference(model: nn.Module, test_data: np.ndarray, enc_seq_len: int, 
            dec_seq_len: int, scaler: StandardScaler, device:str) -> None:
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



