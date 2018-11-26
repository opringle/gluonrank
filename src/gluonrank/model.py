from mxnet import nd, gluon


class UserNet(gluon.nn.HybridSequential):
    def __init__(self, n_units):
        """
        :param num_filters: number of filters in convolutional block
        """
        super().__init__()
        with self.name_scope():
            self.add(gluon.nn.Dense(units=n_units, activation='relu'))
            self.add(gluon.nn.Dense(units=n_units//2, activation='relu'))
            self.add(gluon.nn.Dense(units=n_units//4, activation='relu'))
            self.add(gluon.nn.Dense(units=n_units//6, activation='relu'))


class ItemNet(gluon.nn.HybridSequential):
    def __init__(self, n_units):
        """
        :param num_filters: number of filters in convolutional block
        """
        super().__init__()
        with self.name_scope():
            self.add(gluon.nn.Dense(units=n_units, activation='relu'))
            self.add(gluon.nn.Dense(units=n_units//2, activation='relu'))
            self.add(gluon.nn.Dense(units=n_units//4, activation='relu'))
            self.add(gluon.nn.Dense(units=n_units//6, activation='relu'))


class RankNet(gluon.nn.HybridBlock):
    """
    Neural network for ranking
    """
    def __init__(self, latent_size):
        """

        """
        super().__init__()
        with self.name_scope():
            self.dense = gluon.nn.Dense(units=1)
            self.usernet = UserNet(latent_size)
            self.itemnet = ItemNet(latent_size)

    def hybrid_forward(self, F, user_features, item_features, negative_item_features):
        """
        :param x: mxnet ndarray of data
        :return: mxnet ndarray of data
        """
        user_f = self.usernet(user_features)
        item_f = self.itemnet(item_features)
        neg_item_f = self.itemnet(negative_item_features)

        p = F.concat(*[user_f, item_f], dim=1)
        n = F.concat(*[user_f, neg_item_f], dim=1)

        # p_mf = (user_f * item_f).sum(1)
        # n_mf = (user_f * neg_item_f).sum(1)

        return F.sigmoid(self.dense(p) - self.dense(n))
