from src.model import Transformer
from tokenizador.tokenizer import Tokenizer
import torch

tkn = Tokenizer("How does neural network work?")
input_ids, vocab_size = tkn.forward()

model = Transformer(
    max_seq_len = 512,
    vocab_size=vocab_size,
    embeding_dim=512,
    n_layers=2,
    d_model=512,
    n_heads=2
)

weights_path = 'transformer_model.pth'
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))


model.eval()

with torch.no_grad():
    result = model(input_ids)
    predicted_id = torch.argmax(result, dim=-1)
    decoded_text = tkn.decoder(output=predicted_id[0])

#print(predicted_id)
print(f"Texto Predito: {decoded_text}")