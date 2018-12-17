import numpy as np

def precision_recall(rankings, interactions):
    """
    Compute precision and recall of a set of rankings
    :param rankings: mxnet ndarray of shape (users, k). Values are item ids
    :param interactions: sparse matrix of interactions per user
    :return: (precision, recall)
    """
    precisions = []
    recalls = []
    for user in range(rankings.shape[0]):
        users_item_interactions = np.where(interactions.getrow(user).toarray()[0] == 1)
        recommended_items = rankings[user]
        # print("recommendations={}\t interactions={}".format(recommended_items, users_item_interactions))
        hits = np.intersect1d(users_item_interactions, recommended_items)
        # print("Hits = {}".format(hits))
        precisions.append(hits.shape[0] / rankings.shape[1])
    return precisions, recalls