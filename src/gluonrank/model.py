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
    def __init__(self, latent_size, nuser, nitem):
        """

        """
        super().__init__()
        with self.name_scope():
            # self.dense = gluon.nn.Dense(units=1)
            # self.usernet = UserNet(latent_size)
            # self.itemnet = ItemNet(latent_size)
            self.user_embedding = gluon.nn.Embedding(input_dim=nuser, output_dim=latent_size)
            self.item_embedding = gluon.nn.Embedding(input_dim=nitem, output_dim=latent_size)

    def f(self, user_id, item_id, user_features, item_features):
        """

        :param user_id:
        :param item_id:
        :param user_features:
        :param item_features:
        :return:
        """
        # user_f = self.usernet(user_features)
        # item_f = self.itemnet(item_features)

        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)


        return (user_embed * item_embed).sum(1)

    def hybrid_forward(self, F, user_features, item_features, negative_item_features, user_id, item_id, neg_item_id):
        """
        :param x: mxnet ndarray of data
        :return: mxnet ndarray of data
        """
        p = self.f(user_id, item_id, user_features, item_features)
        n = self.f(user_id, neg_item_id, user_features, negative_item_features)
        return F.sigmoid(p-n)
