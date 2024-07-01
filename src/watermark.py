import hashlib
import random
import string
import numpy as np

def apply_watermarks(ds, watermark_func, orig_name, col_prefix, frac, seed, **kwargs):
    """Apply the watermarking function to a fraction of the dataset."""
    length = len(ds)
    half_size = int(length * frac)
    mask = np.array([0] * half_size + [1] * (length - half_size))
    
    rng = np.random.default_rng(seed)
    rng.shuffle(mask)
    
    watermarked_text = []
    for m, text in zip(mask, ds[orig_name]):
        if m == 1:
            watermarked_text.append(watermark_func(text, seed, **kwargs))
        else:
            watermarked_text.append(text)
    
    ds = ds.add_column(f'{col_prefix}:{orig_name}', watermarked_text)
    ds = ds.add_column(f'{col_prefix}:label', mask)
    return ds

def random_sequence_watermark(ds, orig_name='text', col_prefix='rand_seq', frac=0.5, seed=0):
    return apply_watermarks(ds, apply_watermark_random_sequence, orig_name, col_prefix, frac, seed)

def natural_copyright_trap_watermark(ds, orig_name='text', col_prefix='natural_ct', frac=0.5, seed=0, n=1):
    return apply_watermarks(ds, apply_watermark_natural_copyright_trap, orig_name, col_prefix, frac, seed, n=n)

def llm_generated_copyright_trap_watermark(ds, orig_name='text', col_prefix='llm_ct', frac=0.5, seed=0, model=None, tokenizer=None, n=1, temp=0.5, seq_len=25):
    return apply_watermarks(ds, apply_watermark_llm_generated_copyright_trap, orig_name, col_prefix, frac, seed, model, tokenizer, n=n, temp=temp, seq_len=seq_len)

def llm_generated_fuzzy_copyright_trap_watermark(ds, orig_name='text', col_prefix='fuzzy_llm_ct', frac=0.5, seed=0, model=None, tokenizer=None, n=1, temp=0.5, seq_len=25, R=1, k=10):
    return apply_watermarks(ds, apply_watermark_llm_generated_fuzzy_copyright_trap, orig_name, col_prefix, frac, seed, model, tokenizer, n=n, temp=temp, seq_len=seq_len, R=R, k=k)

# ======================
# watermarking functions
# ======================

def generate_random_md5(seed):
    # Generate a random string of fixed length
    random_string = str(seed)
    
    # Create an MD5 hash of the random string
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    
    return md5_hash

def apply_watermark_random_sequence(text, seed=0):
    """Append an MD5 watermark to the text."""
    wm = generate_random_md5(seed)
    return text + '\n' + wm

from nltk.tokenize import sent_tokenize

# meeus et al. 2024
def apply_watermark_natural_copyright_trap(text, seed, n=1):
    """Insert the first sentence as a watermark n times into the text."""
    rng = np.random.default_rng(seed)
    
    # Split the document into sentences
    sentences = sent_tokenize(text)
    watermark_sentence = sentences[0]
    
    # Insert the watermark sentence at random locations n times
    for _ in range(n):
        position = rng.integers(0, len(sentences) + 1)
        sentences.insert(position, watermark_sentence)
    
    # Combine the sentences back into a single document
    watermarked_document = ' '.join(sentences)
    
    return watermarked_document


def apply_watermark_llm_generated_copyright_trap(text, seed, model, tokenizer, n=1, temp=0.5, seq_len=25):
    """Insert an LLM generated sentence of length seq_len at temperature temp as a watermark n times into the text."""
    
    rng = np.random.default_rng(seed)
    
    # Split the document into sentences
    sentences = sent_tokenize(text)

    input = tokenizer([". "], return_tensors="pt")
    generated_ids = model.generate(
                    input["input_ids"],
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=seq_len,
                    do_sample=True,
                    temperature=temp,
                )
    # stop at sentence completion
    stop_len = seq_len
    for i in range(input['input_ids'].shape[1], seq_len-1):
        token_id = generated_ids[0, i]
        # found period
        if token_id.item() == input['input_ids'][0,0].item():
            stop_len = i+1
            break
    
    watermark_sentence = tokenizer.batch_decode(generated_ids[:, input['input_ids'].shape[1]:stop_len])[0]
    
    # Insert the watermark sentence at random locations n times
    for _ in range(n):
        position = rng.integers(0, len(sentences) + 1)
        sentences.insert(position, watermark_sentence)
    
    # Combine the sentences back into a single document
    watermarked_document = ' '.join(sentences)
    
    return watermarked_document


# top-k filtering
def top_k_filtering(logits, top_k):
    # Set all logits to -inf except for the top k
    top_k_values, _ = torch.topk(logits, top_k)
    filter_value = top_k_values[:, -1].unsqueeze(1).expand_as(logits)
    logits = torch.where(logits < filter_value, torch.full_like(logits, float('-inf')), logits)
    return logits

# generate fuzzy watermark given original watermark 
# by replacing R tokens with randomly sampled tokens from top k distribution
def generate_fuzzy_watermark(rng, generated_ids, R, k):
    generated_ids_copy = generated_ids.clone()
    replaced_inds = sorted(rng.choice(range(0,generated_ids.shape[1]-1), size=R, replace=False))
    
    # R replacements
    for replaced_ind in replaced_inds:
        context = tokenizer.batch_decode(generated_ids_copy[:, :replaced_ind])[0]
        context_tok = tokenizer(context, return_tensors='pt')

        outputs = model.forward(**context_tok)
        logits = outputs.logits[:,-1,:]
        # sample top k tokens
        filtered_logits = top_k_filtering(logits, k)
        # convert logits to probabilities
        probabilities = F.softmax(filtered_logits, dim=-1)

        while True:
            # sample a token from the filtered logits
            sampled_token_id = torch.multinomial(probabilities, num_samples=1).squeeze(0)
            # do not sample the original token
            if generated_ids_copy[0,replaced_ind].item() != sampled_token_id.item():
                generated_ids_copy[0,replaced_ind] = sampled_token_id.item()
                break

    fuzzy_watermark = tokenizer.batch_decode(generated_ids_copy)[0]
    
    return fuzzy_watermark

def apply_watermark_llm_generated_fuzzy_copyright_trap(text, seed, model, tokenizer, n=1, temp=0.5, seq_len=25, R=1, k=10):
    """
    Insert an LLM generated sentence of length seq_len at temperature temp as a watermark n times into the text.
    In this watermark, R randomly selected tokens are replaced with randomly sampled tokens following top k probability distribution.
    """
    
    rng = np.random.default_rng(seed)
    
    # Split the document into sentences
    sentences = sent_tokenize(text)

    input = tokenizer([". "], return_tensors="pt")
    generated_ids = model.generate(
                    input["input_ids"],
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=seq_len,
                    do_sample=True,
                    temperature=temp,
                )
    # stop at sentence completion
    stop_len = generated_ids.shape[1]
    for i in range(generated_ids.shape[1]-1, input['input_ids'].shape[1], -1):
        token_id = generated_ids[0, i]
        # found rightmost period
        if token_id.item() == input['input_ids'][0,0].item():
            stop_len = i+1
            break
    # truncate input tokens and tokens after sentence completion
    generated_ids = generated_ids[:, input['input_ids'].shape[1]:stop_len]
    # watermark_sentence = tokenizer.batch_decode(generated_ids)[0]
    
    # Insert the watermark sentence at random locations n times
    for _ in range(n):
        position = rng.integers(0, len(sentences) + 1)
        watermark_sentence = generate_fuzzy_watermark(rng, generated_ids, R, k)
        sentences.insert(position, watermark_sentence)
    
    # Combine the sentences back into a single document
    watermarked_document = ' '.join(sentences)
    
    return watermarked_document
