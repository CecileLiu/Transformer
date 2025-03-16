import torch 
import torch.nn as nn 
from torch import nn, Tensor

from .positional_encoder import PositionalEncoder

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
        num_predicted_features: int=1,
        pretrained_model_path: str = None,
        device: str = 'cuda'
        ): 
        super().__init__()
        self.dec_seq_len = dec_seq_len
        self.encoder_input_layer = nn.Linear(input_size, dim_val).to(device) 
        self.decoder_input_layer = nn.Linear(num_predicted_features, dim_val).to(device) 
        self.linear_mapping = nn.Linear(dim_val, num_predicted_features).to(device) 
        self.positional_encoding_layer = PositionalEncoder(d_model=dim_val, device=device) # dropout_pos_enc
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            dim_val, n_heads, dim_feedforward_encoder, dropout_encoder, batch_first=batch_first).to(device) 
        self.encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers).to(device) 
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            dim_val, n_heads, dim_feedforward_decoder, dropout_decoder, batch_first=batch_first).to(device) 
        self.decoder = nn.TransformerDecoder(decoder_layer, n_decoder_layers).to(device) 

        if pretrained_model_path:  # Load pretrained weights if provided
            self.load_state_dict(torch.load(pretrained_model_path), strict=False) #False: skip mismatch keys

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        # Encoder processing
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        memory = self.encoder(src)
        
        # Decoder processing
        tgt = self.decoder_input_layer(tgt)
        decoder_output = self.decoder(tgt, memory)
        return self.linear_mapping(decoder_output)

