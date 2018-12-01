from mxnet import gluon
import random
import numpy as np
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
        return (# self.X_U_cont[user_id], self.X_U_emb[user_id],
                np.float32(0), self.X_U_emb[user_id],

                # self.X_I_cont[item_id], self.X_I_cont[item_id], s
                np.float32(0), self.X_I_emb[item_id],

                # self.X_I_cont[neg_item_id], self.X_I_emb[neg_item_id])
                np.float32(0), self.X_I_emb[neg_item_id])

    def split(self, test_frac):
        """
        Splits interactions
        :param train_frac: Fraction of data to be used for training
        :param val_frac: Fraction of data to be used for hyperparameter optimization
        :param test_frac: Fraction of data to be used for testing
        :return: train_dataset, test_dataset
        """
        interactions = self._data[0]
        train_interactions = sorted(interactions, key=lambda x: x[2])[:80000]
        test_interactions = sorted(interactions, key=lambda x: x[2])[80000:]

        train = InteractionsDataset(self.X_U_cont, self.X_U_emb, self.X_I_cont, self.X_I_emb, train_interactions)
        test = InteractionsDataset(self.X_U_cont, self.X_U_emb, self.X_I_cont, self.X_I_emb, test_interactions)
        return train, test


if __name__ == "__main__":
    """
    Run unit-test
    """
    from model import RankNet
    from loss import bpr_loss
    import mxnet as mx
    import time
    from mxnet import autograd, nd

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    n_user = 6000
    n_item = 15000
    n_interactions = 5000

    # generate data
    X_U_cont = np.random.random_sample((n_user, 2))
    X_U_emb = np.random.randint(low=0, high=5999, size=(n_user, 1))
    X_I_cont = np.random.random_sample((n_item, 13))
    X_I_emb = np.random.randint(low=0, high=14999, size=(n_item, 1))

    interactions = [(random.randint(0, n_user), random.randint(0, n_item), random.randint(0, 15400000))
                    for interaction in range(n_interactions)]

    d = InteractionsDataset(X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions)
    logging.debug("Number users = {} Num items = {} Num interactions = {}".format(d.num_user, d.num_item, len(d)))

    record = d[random.randint(0, n_interactions)]
    X = record[0]
    Y = record[1]

    # assert X[0].shape == (2, ), "Unit test failed: user continous feature shape"
    # assert X[1].shape == (1, ), "Unit test failed: user embed feature shape"
    # assert X[2].shape == (13, ), "Unit test failed: item continous feature shape"
    # assert X[3].shape == (5, ), "Unit test failed: item embed feature shape"
    # assert X[4].shape == (13, ), "Unit test failed: negative item continous feature shape"
    # assert X[5].shape == (5, ), "Unit test failed: negative item embed feature shape"

    loader = gluon.data.DataLoader(d, batch_size=50, num_workers=3)
    for batch, (X) in enumerate(loader):
        continue
    logging.info("Unit test success!")


    # define network, loss and optimizer
    net = RankNet(latent_size=500,
                  total_user_embed_cat=len(np.unique(X_U_emb)),
                  total_item_embed_cat=len(np.unique(X_I_emb)))
    logging.info("Network parameters:\n{}".format(net))

    loss = bpr_loss

    ctx = mx.cpu()
    logging.info("Training on {}".format(ctx))
    net.collect_params().initialize(ctx=ctx)

    trainer = gluon.Trainer(params=net.collect_params(),
                            optimizer='Adam',
                            optimizer_params={'learning_rate': 0.01})

    # train the network on the data
    for e in range(1000):
        epoch_loss = 0
        weight_updates = 0
        start = time.time()
        for i, (X) in enumerate(loader):
            X_U_cont, X_U_emb, X_I_cont, X_I_emb, X_I_neg_cont, X_I_neg_emb = (x.as_in_context(ctx) for x in X)

            # Forward & backward pass: loss depends on both positive and negative predictions
            with autograd.record():
                pos_pred = net(X_U_cont, X_U_emb, X_I_cont, X_I_emb)
                neg_pred = net(X_U_cont, X_U_emb, X_I_neg_cont, X_I_neg_emb)
                l = loss(pos_pred, neg_pred)

            l.backward()
            trainer.step(2 * 50)

            epoch_loss += nd.mean(l).asscalar()
            weight_updates += 1
        logging.info("Epoch {}:\tTime={:.4}s\tTrain Loss={:.4}".format(e, time.time() - start, epoch_loss / weight_updates))