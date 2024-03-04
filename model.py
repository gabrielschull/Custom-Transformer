import torch
import torch.nn as nn
import math

# input embeddings
class inputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    # multiply weights as in 3.4 in fwd method
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class positionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Positional encoding as seen in 3.5
        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model) 
        # vector of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # add batch dimension (1, max_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

