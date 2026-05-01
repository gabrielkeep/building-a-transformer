from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
input = tokenizer(
    'Aqui vamos entrar com dados',
    padding='max_length',
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

x = input['input_ids']
vocab_size = tokenizer.vocab_size