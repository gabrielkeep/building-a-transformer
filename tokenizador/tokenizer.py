from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, data): 
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.input = self.tokenizer(
            data,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
    def forward(self):
        return self.input['input_ids'], self.tokenizer.vocab_size
    
    
    def decoder(self, output):
        return self.tokenizer.decode(output, skip_special_tokens=True)