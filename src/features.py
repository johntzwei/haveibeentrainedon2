import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

def create_ngrams(example, n=2):
    """Create bigrams from a list of tokens."""
    return list(nltk.ngrams(example, n=n))

def examples_to_ngrams(examples, n=2):
    """Convert a list of tokenized sentences into bigrams and then into a sparse array."""
    # Flatten list of bigrams into strings of joined bigrams
    ngram_sentences = []
    for example in examples:
        ngrams = create_ngrams(example, n)
        joined_ngrams = []
        
        for ngram in ngrams:
            joined_ngram = '_'.join(str(i) for i in ngram)
            joined_ngrams.append(joined_ngram)
            
        ngram_sentence = ' '.join(joined_ngrams)
        ngram_sentences.append(ngram_sentence)
    
    # Use CountVectorizer to convert bigrams into a sparse matrix
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(ngram_sentences)
    
    return sparse_matrix, vectorizer