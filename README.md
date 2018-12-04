### GluonRank: Your Choice of Deep Learning for Ranking

GluonRank is a toolkit that enables easy implementation of collaborative filtering models using neural networks, to help your prototyping of state of the art ranking systems.


## ToDo

- [ ] Increase the efficiency of the evaluation function
- [ ] Gracefully handle missing continuous embedding or categorical variables & user/item biases
- [ ] Do not require user to index their embedding values for a single matrix 
- [ ] Negative sampling without collisions results in 5X training time. Speed this up!

(answer)[https://stackoverflow.com/questions/53576915/sample-n-zeros-from-a-sparse-coo-matrix/53577344#53577344]

- [x] Build ranking function as network method
- [ ] Create hosted docs
- [ ] Create python package

## Features

- [ ] Allow for sampling more than one negative per interaction


## Ideas

1. Positive and negative items could be in the same tensor, on different axis
2. Can construct a single lookup table for all categorical variables. Vocab_size = np.unique(user_embed_features)
    - Need to index cols to ensure there is no collision between variables
3. The dataset could return a positive with some probability, otherwise returning a negative randomly sampled from a sparse matrix
4. Loss functions should be custom and take both pos_pred and neg_pred
5. 
    
```python

```


