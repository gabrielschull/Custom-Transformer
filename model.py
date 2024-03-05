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

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps # small number to avoid division by zero
        self.gamma = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.bias

class FeedForwarBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) # W1 & B1
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 & B2

    def forward(self, x):
        # (Batch, max_len, d_model) --> (Batch, max_len, d_ff) --> (Batch, max_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv
        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        # (Batch, h , max_len, d_k) --> (Batch, h, max_len, max_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_ (mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, max_len, max_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, max_len, d_model) --> (Batch, max_len, d_model)
        key = self.w_k(k) # (Batch, max_len, d_model) --> (Batch, max_len, d_model)
        value = self.w_v(v) # (Batch, max_len, d_model) --> (Batch, max_len, d_model)

        # Split the d_model into h heads and transpose the result; (Batch, max_len, d_model) --> (Batch, max_len, h, d_k) --> (Batch, h, max_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, self.dropout)

        # (Batch, h, max_len, d_k) --> (Batch, max_len, h, d_k) --> (Batch, max_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h* self.d_k)

        # (Batch, max_len, d_model) --> (Batch, max_len, d_model)
        return self.w_o(x)