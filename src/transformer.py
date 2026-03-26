import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Classe principal
class Transformer(nn.Module):
    def __init__(self, seq, d_model):
        super().__init__()
        self.seq = seq
        self.d_model = d_model

        pe = self.sinusoid_function()
        self.register_buffer('pe', pe)

# Positional Embedding Function
    def sinusoid_function(self):

        pe = torch.zeros((self.seq, self.d_model))
        

        for pos in range(self.seq):
            for i in range(0, self.d_model, 2):
                pe[pos, i] = math.sin(pos / 10_000 ** (i/self.d_model))
                pe[pos, i+1] = math.cos(pos / 10_000 ** (i/self.d_model))

        return pe
    
# Attention Mecanism
class SelfAttention(nn.Module):
    def __init__(self, seq_len, d_model, causal=False):
        super().__init__()

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.causal = causal
        self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

    def scaled_dot_product_attention(self, x):

        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        scores  = (Q @ K.transpose(-2,-1)) / math.sqrt(Q.shape[-1])

        

        if self.causal:

            scores.masked_fill(self.mask == 0, float('-inf'))
            weight = torch.softmax(self.mask, dim=-1)

        else:

            weight = torch.softmax(scores, dim=-1)

        return weight @ V
    
# Multi-Head-Attention Mecanism
class QKVAttention(nn.Module):
    def __init__(self, d_model, h, s, causal=False):
        super().__init__()

        self.h = h
        self.reduced_dimension = d_model // h
        self.seq = s
        self.d_model = d_model
        self.batch = 1 # Posso tornar isso dinâmico
        self.causal = causal

        if causal:
            self.register_buffer('mask', torch.tril(torch.ones(s, s)))
        else:
            self.mask = None

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def MultiHeadAttention(self, x):

        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)

        Q_i = Q.reshape(self.batch, self.seq, self.h, self.reduced_dimension).transpose(1, 2)
        K_i = K.reshape(self.batch, self.seq, self.h, self.reduced_dimension).transpose(1, 2)
        V_i = V.reshape(self.batch, self.seq, self.h, self.reduced_dimension).transpose(1, 2)

        score = (Q_i @ K_i.transpose(-2, -1)) / math.sqrt(self.reduced_dimension)

        if self.causal:        
            score = score.masked_fill(self.mask == 0, float('-inf'))
   
        weight = F.softmax(score, dim=-1) @ V_i
        weight = weight.transpose(1, 2).contiguous().view(self.batch, self.seq, self.d_model)


        return self.W_O(weight).squeeze(0)