from mxnet import gluon
import random
import numpy as np
from sklearn.model_selection import train_test_split
import logging


class InteractionsDataset(gluon.data.ArrayDataset):
    """
    Dataset object, which returns interaction features
    Randomly samples 1 item the user did not interact with (collisions possible)
    Designed to be used with pairwise loss functions
    """
    def __init__(self, X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions):
        """
        Negative sampling dataset that returns data for embedding and fully connected nets jointly
        :param X_U_cont: np.array of shape = (number users, number continous features)
        :param X_U_emb: np.array of shape = (number users, number embedding features)
        :param X_I_cont: np.array of shape = (number items, number continous features)
        :param X_I_emb: np.array of shape = (number users, number embedding features)
        :param interactions:
        """
        super().__init__(interactions)
        self.num_user = X_U_emb.shape[0]
        self.num_item = X_I_emb.shape[0]

        self.X_U_cont = X_U_cont
        self.X_U_emb = X_U_emb
        self.X_I_cont = X_I_cont
        self.X_I_emb = X_I_emb

    def __getitem__(self, idx):
        user_id, item_id, timestamp = self._data[0][idx]
        neg_item_id = random.randint(0, self.num_item - 1)  # ToDo: avoid collisions with positive items when sampling
        logging.debug("User id = {}, item id = {}, negative item id = {}".format(user_id, item_id, neg_item_id))
        return (self.X_U_cont[user_id], self.X_U_emb[user_id],
                # self.X_I_cont[item_id]
                np.float32(0), self.X_I_emb[item_id],
                # self.X_I_cont[neg_item_id]
                np.float32(0), self.X_I_emb[neg_item_id]), np.float32(1)

    def split(self, test_frac):
        """
        Splits interactions
        :param train_frac: Fraction of data to be used for training
        :param val_frac: Fraction of data to be used for hyperparameter optimization
        :param test_frac: Fraction of data to be used for testing
        :return: train_dataset, test_dataset
        """
        interactions = self._data[0]
        train_interactions, test_interactions = train_test_split(interactions, test_size=test_frac, random_state=1)

        train = InteractionsDataset(self.X_U_cont, self.X_U_emb, self.X_I_cont, self.X_I_emb, train_interactions)
        test = InteractionsDataset(self.X_U_cont, self.X_U_emb, self.X_I_cont, self.X_I_emb, test_interactions)
        return train, test


if __name__ == "__main__":
    """
    Run unit-test
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    n_user = 6000
    n_item = 15000
    n_interactions = 5000

    # generate data
    X_U_cont = np.random.random_sample((n_user, 2))
    X_U_emb = np.random.randint(low=0, high=10000, size=(n_user, 1))
    X_I_cont = np.random.random_sample((n_item, 13))
    X_I_emb = np.random.randint(low=0, high=10000, size=(n_item, 5))

    interactions = [(random.randint(0, n_user), random.randint(0, n_item), random.randint(0, 15400000))
                    for interaction in range(n_interactions)]

    d = InteractionsDataset(X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions)
    logging.debug("Number users = {} Num items = {} Num interactions = {}".format(d.num_user, d.num_item, len(d)))

    record = d[random.randint(0, n_interactions)]
    X = record[0]
    Y = record[1]

    assert X[0].shape == (2, ), "Unit test failed: user continous feature shape"
    assert X[1].shape == (1, ), "Unit test failed: user embed feature shape"
    assert X[2].shape == (13, ), "Unit test failed: item continous feature shape"
    assert X[3].shape == (5, ), "Unit test failed: item embed feature shape"
    assert X[4].shape == (13, ), "Unit test failed: negative item continous feature shape"
    assert X[5].shape == (5, ), "Unit test failed: negative item embed feature shape"

    loader = gluon.data.DataLoader(d, batch_size=50, num_workers=3)
    for batch, (X, Y) in enumerate(loader):
        continue
    logging.info("Unit test success!")