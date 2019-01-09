from mxnet import nd, gluon
import mxnet as mx
import logging
import multiprocessing


class FeatureNet(gluon.nn.HybridBlock):
    """
    Handles item features, returning an array for concatenation with usernet
    """
    def __init__(self, latent_size, total_embed_cat, cont=True, cat=True):
        """
        :param latent_size: embedding dimension for all categorical variables
        :param ncat_user: total number of categorical variables in user embedding features
        :param ncat_item: total number of categorical variables in item embedding features
        :param cont: boolean indicating when continous features are included
        :param cat: boolean indicating when categorical features are included
        """
        super().__init__()
        with self.name_scope():
            self.embed = gluon.nn.Embedding(input_dim=total_embed_cat, output_dim=latent_size)
        self.cont = cont
        self.cat = cat

    def hybrid_forward(self, F, features=None, embed_features=None):
        """
        :param x: mxnet ndarray of data
        :return: mxnet ndarray of data
        """
        o = []
        if self.cat:
            e = F.flatten(self.embed(embed_features))  # a vector per batch representing all embedded categoricals
            o.append(e)
        if self.cont:
            o.append(features)
        return F.concat(*o, dim=1)


class DenseNet(gluon.nn.HybridSequential):
    def __init__(self, dense=[128, 64, 32, 16]):
        """
        :param dense: List of dense layer sizes moving forwards through the net
        """
        super().__init__()
        with self.name_scope():
            for d in dense:
                self.add(gluon.nn.Dense(units=d, activation='relu'))
            self.add(gluon.nn.Dense(units=1))


class RankNet(gluon.nn.HybridBlock):
    """
    Shit hot neural network for ranking
    """
    def __init__(self, latent_size, total_user_embed_cat, total_item_embed_cat):
        """
        :param latent_size: embedding dimension for all categorical variables
        :param total_user_embed_cat: total number of categorical variables in user embedding features
        :param total_item_embed_cat: total number of categorical variables in item embedding features
        """
        super().__init__()
        with self.name_scope():
            self.usernet = FeatureNet(latent_size, total_embed_cat=total_user_embed_cat, cont=False, cat=True)
            self.itemnet = FeatureNet(latent_size, total_embed_cat=total_item_embed_cat, cont=False, cat=True)
            # self.densenet = DenseNet([32, 16, 8])

    def hybrid_forward(self, F, user_features=None, user_embed_features=None, item_features=None, item_embed_features=None):
        """
        Pass input features through different network blocks
        """
        u_o = self.usernet(user_features, user_embed_features)
        i_o = self.itemnet(item_features, item_embed_features)

        mf = (u_o * i_o).sum(1)

        # glm = self.densenet(F.concat(*[u_o, i_o], dim=1)).reshape((-1, ))
        # return o.reshape((-1, ))
        return mf

    def rank(self, dataset, context, k, exclude):
        """
        Rank all items for all users in dataset, excluding specific interactions in ranking and returning top k
        :param dataset: ImplicitInteractions dataset object
        :param exclude: scipy sparse array of interactions to exclude in ranking
        :param context: context to perform network forward passes on
        :param k: max number of items to rank per user
        :return: nd.array shape (users, k)
        """
        out = nd.zeros(shape=(dataset.num_user, k), ctx=context)

        X_U_cont = nd.array(dataset.X_U_cont)
        X_U_emb = nd.array(dataset.X_U_emb)
        X_I_cont = nd.array(dataset.X_I_cont)
        X_I_emb = nd.array(dataset.X_I_emb)

        logging.info("Ranking all items for all users")
        for user in range(dataset.num_user):
            # get the user's features
            X_U_c = nd.ones_like(X_U_emb)  # nd.tile(X_U_cont[user], reps=dataset.num_item)
            X_U_e = nd.tile(X_U_emb[user], reps=(dataset.num_item, 1))

            # get all item features
            X_I_c = nd.ones_like(X_U_emb)  # nd.array(dataset.X_I_cont)
            X_I_e = nd.array(X_I_emb)

            # rank all items for the user
            user_rankings = self(X_U_c, X_U_e, X_I_c, X_I_e)

            # exclude specific rankings and filter to top k
            if exclude is not None:
                exclude_interactions = nd.array(exclude.getrow(user).toarray()[0] == 0)
                user_rankings = (user_rankings * exclude_interactions)
            out[user] = user_rankings.topk(k=k, axis=0)
        return out


if __name__ == "__main__":
    """
    Run unit-test
    """
    # only for unit testing
    import numpy as np
    import mxnet as mx

    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    batch_size = 128

    # define how many of each feature type to generate
    user_n_cont_features = 5
    user_n_cat_features = 1
    item_n_cat_features = 1
    item_n_cont_features = 100

    # generate data
    user_cont_features = nd.normal(shape=(batch_size, user_n_cont_features))
    user_embed_features = nd.array(np.random.randint(low=0, high=10000, size=(batch_size, user_n_cat_features)))

    item_cont_features = nd.normal(shape=(batch_size, item_n_cont_features))
    item_embed_features = nd.array(np.random.randint(low=0, high=10000, size=(batch_size, item_n_cat_features)))

    # compute total number of user/item categorical features
    user_cats = np.unique(user_embed_features.asnumpy()).shape[0]
    item_cats = np.unique(item_embed_features.asnumpy()).shape[0]
    logging.debug("{} total user categories & {} total item categories".format(user_cats, item_cats))

    # initialize net
    net = RankNet(latent_size=5, total_user_embed_cat=user_cats, total_item_embed_cat=item_cats)
    net.collect_params().initialize(ctx=mx.cpu())
    logging.debug("Network architecture: {}".format(net))

    # predict on generated data
    out = net(user_cont_features, user_embed_features, item_cont_features, item_embed_features)
    assert out.shape == (batch_size, ), "Unexpected output shape {}".format(out.shape)
    print("RankNet unit-test success!")
