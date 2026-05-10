from src.model import Transformer
from tokenizador.tokenizer import Tokenizer
import torch

tkn = Tokenizer("The functioning of biological neurons")
input_ids, vocab_size = tkn.forward()

model = Transformer(
    max_seq_len = 512,
    vocab_size=vocab_size,
    embeding_dim=512,
    n_layers=6,
    d_model=512,
    n_heads=8
)

weights_path = 'transformer_model.pth'
model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))


model.eval()
# How many new words to generate
max_new_tokens = 25 
with torch.no_grad():
    for _ in range(max_new_tokens):
        # Pass current sequence to the model (only last 512 tokens to stay within max_seq_len)
        logits = model(input_ids[:, -512:])
        
        # Focus only on the last token's predictions
        last_token_logits = logits[:, -1, :] 
        
        # Get the most likely next token (greedy decoding)
        next_token_id = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        
        # Append to the sequence
        input_ids = torch.cat([input_ids, next_token_id], dim=1)
    # Decode the final generated sequence
    decoded_text = tkn.decoder(output=input_ids[0])
    
print(f"Texto Predito: {decoded_text}")