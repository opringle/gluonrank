### GluonRank: Your Choice of Deep Learning for Ranking

GluonRank is a toolkit that enables easy implementation of collaborative filtering models using neural networks, to help your prototyping of state of the art ranking models.


## ToDo

- If all categorical features recieve the same embedding size, you can pass them in as a matrix and use a single lookup table
    - Problem here is 
 
- Create hosted docs
- Create python package

## Ideas

1. Positive and negative items can be in the same tensor, on different axis
2. Can construct a single lookup table for all categorical variables. Vocab_size = np.unique(user_embed_features)
    - Need to index cols to ensure there is no overlap
    
```python

```

## Installation

Make sure you have Python 3.6 and recent version of MXNet.
You can install ``MXNet`` and ``GluonNLP`` using pip:

```bash
pip install --upgrade mxnet
pip install gluonrank
```

## Docs

GluonNLP documentation is available at...


