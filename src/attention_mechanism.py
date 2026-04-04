import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
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

    def forward(self, x):

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
    

class FeedForward(nn.Module):
    def __init__(self, seq, hidden_size, dropout:float=0.1):
        super().__init__()

        self.f1 = nn.Linear(seq, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.f2 = nn.Linear(hidden_size, seq)
        
    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x) # Considerar usar outra função de ativação
        x = self.dropout(x)
        x = self.f2(x)
        return x