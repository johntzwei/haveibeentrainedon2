from datasets import load_dataset

# returns an arrow dataset
def load_simple_wiki():
    ds = load_dataset("wikipedia", "20220301.simple")
    return ds['train']

# returns an arrow dataset
def load_subset(load_fn=load_simple_wiki, rows=1000):
    ds = load_fn()
    return ds.select(range(rows))