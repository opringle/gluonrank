from spotlight.interactions import Interactions
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.evaluation import precision_recall_score

import numpy as np
import pandas as pd
from src.gluonrank.evaluate import precision_recall
from scipy.sparse import csr_matrix


def reindex_col(df, col):
    """
    Maps a column of integers to increment from 0
    :param df: pandas dataframe
    :param col: column to be reindexed
    :return: pandas dataframe & mapping dict
    """
    mapping_dict = {k: v for v, k in enumerate(df[col].unique())}
    df[col] = df[col].map(mapping_dict)
    return df, mapping_dict


def get_data():

    df = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])

    df, idx_to_user = reindex_col(df, 'user')
    df, idx_to_item = reindex_col(df, 'item')

    test_df = df.groupby(["user"], as_index=False, group_keys=False).apply(lambda x: x.nlargest(1, ["timestamp"]))
    train_df = df[~df.index.isin(test_df.index.values)]

    train_rows = train_df.user.values
    train_cols = train_df.item.values
    train_data = np.ones_like(train_rows)
    train_sparse = csr_matrix((train_data, (train_rows, train_cols)), shape=(len(train_rows), len(train_cols)))

    # test_rows = test_df.user.values
    # test_cols = test_df.item.values
    # test_data = np.ones_like(test_rows)
    # test_sparse = csr_matrix((test_data, (test_rows, test_cols)), shape=(len(test_rows), len(test_cols)))

    train = Interactions(user_ids=train_df.user.values, item_ids=train_df.item.values)
    test = Interactions(user_ids=test_df.user.values, item_ids=test_df.item.values)
    return df, train, test, train_sparse  # , test_sparse


def rank(model, user_ids, k, exclude):
        """
        Rank all items for all users in dataset, excluding specific interactions in ranking and returning top k
        :param dataset: ImplicitInteractions dataset object
        :param exclude: scipy sparse array of interactions to exclude in ranking
        :param context: context to perform network forward passes on
        :param k: max number of items to rank per user
        :return: nd.array shape (users, k)
        """
        out = np.zeros(shape=(len(user_ids), k))
        print("Output shape = {}".format(out.shape))

        print("Ranking all items for all users")
        for user in user_ids:

            # rank all items for the user
            user_rankings = model.predict(user_ids=user)

            # exclude specific rankings and filter to top k
            if exclude is not None:
                exclude_interactions = np.array(exclude.getrow(user).toarray()[0] == 0)
                user_rankings = (user_rankings * exclude_interactions)
            out[user] = user_rankings.argsort()[-k:][::-1]
        return out


if __name__ == '__main__':
    """
    Train a benchmark to compare to my model
    """
    df, train, test, train_sparse = get_data()

    model = ImplicitFactorizationModel(loss='bpr', embedding_dim=32, n_iter=10, batch_size=256, l2=0.0,
                                       learning_rate=0.01, num_negative_samples=1)
    model.fit(train, verbose=True)

    rankings = rank(model, df.user.unique(), k=10, exclude=None)
    print("Train Rankings = {}".format(rankings))

    precisions, recalls = precision_recall_score(model, train, train=None, k=10)
    print("Model Train precision at 10={}".format(sum(precisions)/len(precisions)))

    # compute information retrieval metrics
    precisions, recalls = precision_recall(rankings, interactions=train_sparse)
    print("Model Train ranking precision@{}={:.4f}".format(10, sum(precisions) / len(precisions)))

    precisions, recalls = precision_recall_score(model, test, train=train, k=10)
    print("Model Test precision at 10={}".format(sum(precisions) / len(precisions)))
