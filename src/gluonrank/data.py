from mxnet import gluon
import random
import numpy as np
from sklearn.model_selection import train_test_split


class InteractionsDataset(gluon.data.ArrayDataset):
    """
    Dataset object, which returns interaction features
    Randomly samples 1 item the user did not interact with (collisions possible)
    Designed to be used with pairwise loss functions
    """
    def __init__(self, user_features, item_features, interactions):
        """
        :param user_features: nested list of user features. Index is user id [(1.0, 475.6423, 0), ()]
        :param item_features:
        :param interactions:
        """
        super().__init__(interactions)
        self.num_user = user_features.shape[0]
        self.num_item = item_features.shape[0]
        self.user_features = user_features
        self.item_features = item_features

    def __getitem__(self, idx):
        user_id, item_id, timestamp = self._data[0][idx]
        negative_item_id = random.randint(0, self.num_item - 1)  # ToDo: avoid collisions with positive items when sampling
        return (self.user_features[user_id], self.item_features[item_id], self.item_features[negative_item_id], user_id, item_id, negative_item_id), np.float32(1)

    def split(self, test_frac, val_frac=None):
        """
        Splits interactions
        :param train_frac: Fraction of data to be used for training
        :param val_frac: Fraction of data to be used for hyperparameter optimization
        :param test_frac: Fraction of data to be used for testing
        :return: train_dataset, test_dataset
        """
        interactions = self._data[0]
        train_interactions, test_interactions = train_test_split(interactions, test_size=test_frac, random_state=1)

        train = InteractionsDataset(self.user_features, self.item_features, train_interactions)
        test = InteractionsDataset(self.user_features, self.item_features, test_interactions)
        return train, test


if __name__ == "__main__":
    """
    Run unit-test
    """
    print("Unit-test success!")
