from mxnet import gluon
import logging
from scipy.sparse import csr_matrix
import numpy as np
import random


class InteractionsDataset(gluon.data.ArrayDataset):
    """
    Dataset for implicit interaction data.
    When a user->item interaction is retrieved, randomly samples 1 item the user did not interact with.
    Designed for use with pairwise loss functions.
    """
    def __init__(self, X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions):
        """
        :param X_U_cont: The continuous features of the users.
        np.array of shape = (number users, number continuous features)
        X_U_cont[1] refers to the continous features for user id 1.

        :param X_U_emb: The categorical features of the users, to be embedded.
        Each column is considered a separate feature.
        np.array of shape = (number users, number embedding features)
        X_U_emb[6] referes to the categorical features for user id 6.

        :param X_I_cont: The continuous features of the items.
        np.array of shape = (number items, number continuous features)
        X_I_cont[1] refers to the features for item id 1.

        :param X_I_emb: The categorical features of the items, to be embedded.
        Each column is considered a separate feature.
        np.array of shape = (number items, number embedding features)
        X_I_emb[3] referes to the categorical features for item id 3.

        :param interactions: Implicit interaction data. List of tuple (user_id, item_id, timestamp).
        """
        super().__init__(interactions)
        self.num_user = X_U_emb.shape[0]
        self.num_item = X_I_emb.shape[0]

        self.X_U_cont = X_U_cont
        self.X_U_emb = X_U_emb
        self.X_I_cont = X_I_cont
        self.X_I_emb = X_I_emb

        # create sparse representation of interactions
        rows = np.array([x[0] for x in interactions])
        cols = np.array([x[1] for x in interactions])
        data = np.ones_like(rows)
        self.sparse_interactions = csr_matrix((data, (rows, cols)), shape=(self.num_user, self.num_item))

    def __getitem__(self, idx):
        user_id, item_id, timestamp = self._data[0][idx]

        # get interactions for the user, find negatives and sample one
        item_interactions = self.sparse_interactions.getrow(user_id).toarray()[0]
        negative_items = np.where(item_interactions == 0)[0]
        neg_item_id = np.random.choice(negative_items)
        logging.debug("User id = {}, item id = {}, negative item id = {}".format(user_id, item_id, neg_item_id))

        return (self.X_U_cont[user_id], self.X_U_emb[user_id],
                self.X_I_cont[item_id], self.X_I_cont[item_id],
                self.X_I_cont[neg_item_id], self.X_I_emb[neg_item_id])


if __name__ == "__main__":
    """
    Run unit-test
    """
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    n_user = 6000
    n_item = 15000
    n_interactions = 2000

    # generate data
    X_U_cont = np.random.random_sample((n_user, 2))
    X_U_emb = np.random.randint(low=0, high=n_user, size=(n_user, 1))
    X_I_cont = np.random.random_sample((n_item, 13))
    X_I_emb = np.random.randint(low=0, high=n_item, size=(n_item, 1))

    interactions = [(random.randint(0, n_user-1), random.randint(0, n_item-1), random.randint(0, 15400000))
                    for interaction in range(n_interactions)]

    d = InteractionsDataset(X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions)
    logging.debug("Number users = {} Num items = {} Num interactions = {}".format(d.num_user, d.num_item, len(d)))

    loader = gluon.data.DataLoader(d, batch_size=50, num_workers=3)
    for batch, (x) in enumerate(loader):
        print("Negative item ids = \n{}".format(x[5]))
    logging.info("Unit test success!")
