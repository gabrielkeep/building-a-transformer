from .attention_mechanism import QKVAttention, FeedForward
from .positional_embedding import PosEmbeding
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads):
        super().__init__()

        # Defining Masked Multi Head Attetion
        self.mmha = QKVAttention(
            d_model = d_model,
            h = n_heads,
            s = max_seq_len,
            causal = True 
        )

        # Defining FeedForward (applies a layer normalization)
        self.ff = FeedForward(
            seq = max_seq_len,
            hidden_size = 4 * d_model
        )

        self.add_norm = nn.LayerNorm(d_model)

    def forward(self, x):

        res = x 
        x = self.mmha(x) # Masked Multi Head Attention
        x = self.add_norm(x + res) # Layer Normalization
        
        res = x 
        x = self.ff(x) # Feed Forward
        x = self.add_norm(x + res) # Residual + Layer Norm

        return x



class Transformer(nn.Module):
    def __init__(self, max_seq_len, vocab_size, embeding_dim, n_layers, d_model, n_heads):
        super().__init__()

        self.embeding = nn.Embedding(vocab_size, embeding_dim)
        
        self.pos_embeding = PosEmbeding(
            seq = max_seq_len,
            d_model = embeding_dim
        )

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model = d_model,
                    h = n_heads,
                    s = max_seq_len,
                    causal=True # True if Masked
                )
            )

    def forward(self, x):

        self.input_embedding = self.embeding(x)
        self.positional_embedding = (self.input_embedding + self.pos_embeding().pe)

        return self.positional_embedding, self.input_embedding

