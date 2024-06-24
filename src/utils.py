from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

def decode(l):
    return tokenizer.decode(l)

def create_examples(ds, col_name='text'):
    tokenized_text = []
    for i, ex in enumerate(ds):    
        text = ex[col_name]
        tokens = tokenizer(text)['input_ids']
        tokenized_text.append(tokens)
    return tokenized_text
