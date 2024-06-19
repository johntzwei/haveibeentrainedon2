import nltk
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

def create_bigrams(example):
    """Create bigrams from a list of tokens."""
    return list(nltk.bigrams(example))

def examples_to_bigrams(examples):
    """Convert a list of tokenized sentences into bigrams and then into a sparse array."""
    # Flatten list of bigrams into strings of joined bigrams
    bigram_sentences = [' '.join(['_'.join(str(i) for i in bigram) for bigram in create_bigrams(example)]) for example in examples]
    
    # Use CountVectorizer to convert bigrams into a sparse matrix
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(bigram_sentences)
    
    return sparse_matrix, vectorizer