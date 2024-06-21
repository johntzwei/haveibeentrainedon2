import hashlib
import random
import string
import numpy as np

def generate_random_md5(seed):
    # Generate a random string of fixed length
    random_string = str(seed)
    
    # Create an MD5 hash of the random string
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    
    return md5_hash

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

def random_sequence_watermark(ds, orig_name='text', col_prefix='rand_seq', frac=0.5, seed=0):
    return apply_watermarks(ds, apply_watermark_random_sequence, orig_name, col_prefix, frac, seed)

def natural_copyright_trap_watermark(ds, orig_name='text', col_prefix='natural_ct', frac=0.5, seed=0, n=1):
    return apply_watermarks(ds, apply_watermark_natural_copyright_trap, orig_name, col_prefix, frac, seed, n=n)
