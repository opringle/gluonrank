### GluonRank: Your Choice of Deep Learning for Ranking

GluonRank is a toolkit that enables easy implementation of collaborative filtering models using neural networks, to help your prototyping of state of the art ranking systems.

## Installation


### Pip

Make sure you are using Python 3.6. You can install `MXNet` and `GluonRank` using pip:

```bash
pip install --index-url https://test.pypi.org/simple/ gluonrank
```

### Uploading to pypi for testing

Build distribution

`python setup.py sdist bdist_wheel`bash

Upload to pypi test index

`twine upload --repository-url https://test.pypi.org/legacy/ dist/*`bash

## Docs

Coming soon... (it might be a while actually...)


## ToDo

- [ ] Categorical features
    - [ ] Get running with multiple categorical features, maintain performance when reducing to a single one
    - [ ] Gracefully handle missing continuous embedding or categorical variables & user/item biases
    - [ ] Do not require user to index their embedding values for a single matrix 
- [ ] Continuous features
    - [ ] Get running with 1 continous feature, maintain performance when excluded
    - [ ] Get running with several continuous features
- [ ] Increase the efficiency of the evaluation function
- [ ] Speed up negative sampling... Negative sampling without collisions results in 5X training time.

(answer)[https://stackoverflow.com/questions/53576915/sample-n-zeros-from-a-sparse-coo-matrix/53577344#53577344]

- [x] Match spotlight performance with implicit interaction model on movielense data
- [x] Build ranking function as network method
- [ ] Create python package
    - 
- [ ] Create hosted docs

## Features

- [ ] Allow for sampling more than one negative per interaction
- [ ] Allow for feedback that can be in the form of 0, 1 or -1. (eg swiping data)

## Ideas
