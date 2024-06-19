import hashlib
import random
import string

def generate_random_md5(seed='seed'):
    # Generate a random string of fixed length
    random_string = ''.join(seed)
    
    # Create an MD5 hash of the random string
    md5_hash = hashlib.md5(random_string.encode()).hexdigest()
    
    return md5_hash

def random_sequence_watermark(ds, orig_name='text', col_prefix='rand_seq:'):
    wm = generate_random_md5('random')
    append_wm = lambda x: x + '\n' + wm
    return ds.add_column('%s%s' % (col_prefix, orig_name), map(append_wm, ds[orig_name]))