import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Positional Embedding
class PosEmbeding(nn.Module):
    def __init__(self, seq, d_model):
        super().__init__()
        self.seq = seq
        self.d_model = d_model

        pe = self.forward()
        self.register_buffer('pe', pe)

# Positional Embedding Function
    def forward(self):

        pe = torch.zeros((self.seq, self.d_model))
        

        for pos in range(self.seq):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / 10_000 ** (i/self.d_model))
                pe[pos, i+1] = math.cos(pos / 10_000 ** (i/self.d_model))

        return pe
    
