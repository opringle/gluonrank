from .data import InteractionsDataset
import pandas as pd


def leave_n_out_split(dataset, n=1):
    """
    Split the interactions, such that the latest n interactions per user are in the test set
    :param dataset: InteractionsDataset object
    :param n: number of user interactions to leave in the test set
    :return: train_dataset, test_dataset
    """
    # extract interactions into pandas
    interactions = dataset._data[0]
    df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'timestamp'])

    test_df = df.groupby(["user_id"], as_index=False, group_keys=False) \
        .apply(lambda x: x.nlargest(n, ["timestamp"]))
    train_df = df[~df.index.isin(test_df.index.values)]

    train_interactions = [tuple(x) for x in train_df.values]
    test_interactions = [tuple(x) for x in test_df.values]

    train = InteractionsDataset(dataset.X_U_cont, dataset.X_U_emb, dataset.X_I_cont, dataset.X_I_emb, train_interactions)
    test = InteractionsDataset(dataset.X_U_cont, dataset.X_U_emb, dataset.X_I_cont, dataset.X_I_emb, test_interactions)
    return train, test