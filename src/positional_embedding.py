import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Embedding
class PosEmbeding(nn.Module):
    def __init__(self, max_seq_length, embedding_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

        pe = self.forward()
        self.register_buffer('pe', pe)

# Positional Embedding Function
    def forward(self):

        pe = torch.zeros((self.max_seq_length, self.embedding_dim))
        

        for pos in range(self.max_seq_length):
            for i in range(0, self.embedding_dim, 2):
                pe[pos, i] = math.sin(pos / 10_000 ** (i/self.embedding_dim))
                pe[pos, i+1] = math.cos(pos / 10_000 ** (i/self.embedding_dim))

        return pe
    
