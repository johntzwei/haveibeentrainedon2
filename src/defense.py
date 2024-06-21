import numpy as np

import features
import utils

def naive_exact_match(examples, n=8):
    sparse_mat, vectorizer = features.examples_to_ngrams(examples, n=n)
    feature_names = vectorizer.get_feature_names_out()
    centroid = sparse_mat.mean(axis=0)
    indices = np.argsort(-centroid)
    
    scores = []
    for ex in examples:
        ex_str = utils.decode(ex)

        # query features in order
        for i in range(0, 1000):
            ngram_str = feature_names[indices[0,i]]
            ngram_toks = list(int(i) for i in ngram_str.split('_'))
            ngram = utils.decode(ngram_toks)

            if ngram in ex_str:
                break
        
        scores.append(i)
    
    # returns examples, and their scores
    return vectorizer, scores