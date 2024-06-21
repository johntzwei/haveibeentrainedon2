from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

def decode(l):
    return tokenizer.decode(l)

def batched(input_list, chunk_size):
    return list(input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size))

def create_examples_from_segs(ds, col_name='text', seq_len=2048):
    tokenized_text = []
    for i, ex in enumerate(ds):    
        text = ex[col_name]
        tokens = tokenizer(text)['input_ids']
        tokenized_text.extend(tokens)
    return batched(tokenized_text, seq_len)

def create_examples(ds, col_name='text'):
    tokenized_text = []
    for i, ex in enumerate(ds):    
        text = ex[col_name]
        tokens = tokenizer(text)['input_ids']
        tokenized_text.append(tokens)
    return tokenized_text