from src.attention_mechanism import QKVAttention, FeedForward
from src.positional_embedding import PosEmbeding
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, max_seq_len, d_model, n_heads):
        super().__init__()

        # Defining Masked Multi Head Attetion
        self.masked_multi_head_attention = QKVAttention(
            d_model = d_model,
            h = n_heads,
            seq = max_seq_len,
            causal = True 
        )

        # Defining FeedForward (applies a layer normalization)
        self.masked_multi_head_attention = FeedForward(
            seq = max_seq_len,
            hidden_size = 4 * d_model
        )

        self.skip_connection_normalization = nn.LayerNorm(d_model)

    def forward(self, x):

        res = x 
        x = self.masked_multi_head_attention(x) # Masked Multi Head Attention
        x = self.skip_connection_normalization(x + res) # Layer Normalization
        
        res = x 
        x = self.masked_multi_head_attention(x) # Feed Forward
        x = self.skip_connection_normalization(x + res) # Residual + Layer Norm

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

        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        input_embedding = self.embeding(x)

        positional_embedding = (input_embedding + self.pos_embeding)

        for transformer_block in self.blocks:
            x = transformer_block(positional_embedding)

        logits = self.linear(x)

        #prob = self.softmax(logits)
    
        return logits

