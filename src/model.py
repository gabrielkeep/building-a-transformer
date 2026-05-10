from src.attention_mechanism import QKVAttention, FeedForward
from src.positional_embedding import PosEmbeding
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads):
        super().__init__()

        # Defining Masked Multi Head Attention
        self.masked_multi_head_attention = QKVAttention(
            d_model = d_model,
            h = n_heads,
            seq = max_seq_len,
            causal = True 
        )

        # Defining FeedForward
        self.feed_forward = FeedForward(
            d_model = d_model,
            hidden_size = 4 * d_model
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Layer Norm for Attention
        x = x + self.masked_multi_head_attention(self.ln1(x))
        
        # Pre-Layer Norm for Feed Forward
        x = x + self.feed_forward(self.ln2(x))

        return x



class Transformer(nn.Module):
    def __init__(self, max_seq_len, vocab_size, embeding_dim, n_layers, d_model, n_heads):
        super().__init__()

        self.embeding = nn.Embedding(
            vocab_size,
            embeding_dim
        )
        
        self.pos_embeding = PosEmbeding(
            max_seq_length = max_seq_len,
            embedding_dim = embeding_dim
        ).pe

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model = d_model,
                    n_heads = n_heads,
                    max_seq_len = max_seq_len,
                    #causal=True # True if Masked
                )
            )

        self.ln_f = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        # Embedding
        x = self.embeding(x)

        # Positional Embedding
        seq_len = x.size(1)
        x = (x + self.pos_embeding[:seq_len, :])

        # Transformer Blocks
        for transformer_block in self.blocks:
            x = transformer_block(x)

        # Final Layer Norm and Linear Layer
        x = self.ln_f(x)
        logits = self.linear(x)
    
        return logits

