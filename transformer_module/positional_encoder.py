
import torch
import torch.nn as nn 
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, device:str="cuda"):
        '''
        arguments in __init__() are the parameters; forward() represents how the calculation will be run
        '''
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).to(device)  # 這個步驟可以看成把max_len變成torch tensor，這個值就是sin/cos括號裡面的`pos`
        '''
        torch.arange(0, d_model, 2) --> Generates [0, 2, 4, ..., d_model-2]
        (-math.log(10000.0) / d_model) --> 詳細看投影片的說明
        * Uses exp(log(x)) instead of direct exponentiation for better stability
        '''
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)  # sin/cos括號裡面的分母

        pe = torch.zeros(max_len, 1, d_model).to(device) # place holder for position embedding
        pe[:, 0, 0::2] = torch.sin(position * div_term) # Assigns the calculated sine values to the even-indexed columns of the pe tensor (columns 0, 2, 4, etc.).
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        '''
        It registers the pe tensor as a buffer. 
        In PyTorch, buffers are tensors that are saved as part of a module's state but are not considered model parameters. 
        This means they are not updated by the optimizer during training. We don't want the positional encoding to be learned; 
        we want it to remain fixed based on the sine/cosine functions. This is why it's registered as a buffer and not a parameter.
        '''
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        # return self.dropout(x)
        return x