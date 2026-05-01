from src.model import Transformer

model = Transformer(
    max_seq_len = 512,
    vocab_size=30522,
    embeding_dim=512,
    n_layers=1,
    d_model=512,
    n_heads=1
)

print(model)

