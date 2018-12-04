import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd, nd
import multiprocessing
import argparse
import logging
import time

from src.gluonrank.data import InteractionsDataset
from src.gluonrank.model import RankNet
from src.gluonrank.loss import PointwiseLoss, BprLoss, HingeLoss
from src.gluonrank.evaluate import precision_recall

def parse_args():
    """
    Parse script arguments
    :return: dict of args
    """
    parser = argparse.ArgumentParser(description="Train a ranking model on the movielense data")

    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--gpus', type=int, default=0,
                        help='num of gpus to distribute  model training on. 0 for cpu')
    group.add_argument('--no-hybridize', action='store_true',
                        help='use symbolic network graph for increased computational eff')

    group = parser.add_argument_group('Network architecture')
    group.add_argument('--embed', type=int, default=32, help='Embedding size for every categorical feature')

    group = parser.add_argument_group('Regularization arguments')
    group.add_argument('--dropout', type=float, default=0.0,
                       help='dropout probability for fully connected layers')
    group.add_argument('--l2', type=float, default=0.0,
                       help='weight regularization penalty')

    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--loss', type=str, default='BPR', choices=['BPR', 'Hinge', 'Pointwise'],
                       help='loss function to minimize during training')
    group.add_argument('--epochs', type=int, default=10,
                       help='num of times to loop through training data')
    group.add_argument('--batch-size', type=int, default=256,
                       help='number of training examples per batch')
    group.add_argument('--lr', type=float, default=0.01,
                       help='optimizer learning rate')
    group.add_argument('--b1', type=float, default=0.9,
                       help='optimizer first moment')
    group.add_argument('--b2', type=float, default=0.999,
                       help='optimizer second moment')

    group = parser.add_argument_group('Evaluation arguments')
    group.add_argument('--test-interactions', type=int, default=1,
                       help='number of interactions per user to put in test set')
    group.add_argument('--k', type=int, default=10,
                       help='number recommendations per user')
    return parser.parse_args()


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


def get_embedding_matrix(df, id_col, cols):
    """
    Index all cols in df, such that no col has overlapping indices with another
    :param df: pandas dataframe
    :parama cols: list of cols to index
    :return: pandas dataframe modified
    """
    df.sort_values(id_col)
    index = 0
    index_map = {}
    for col in [id_col] + cols:
        distinct = df[col].unique()
        n_distinct = len(distinct)
        index_map[col] = {k: (v + index) for v, k in enumerate(distinct)}
        index += n_distinct
        df[col] = df[col].map(index_map[col])
    return df[[id_col] + cols].values


def get_data():
    """
    Preprocess the movielense data
    """
    # load user, item and interaction dataframes
    interactions = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    user_metadata = pd.read_csv('./data/ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    cols = ['movie_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
            'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
           'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    item_metadata = pd.read_csv('./data/ml-100k/u.item', sep='|', encoding="ISO-8859-1", names=cols)

    # reindex all ids to start from 0 and increment by 1
    interactions, usr_to_idx = reindex_col(interactions, 'user')
    interactions, item_to_idx = reindex_col(interactions.sort_values('item'), 'item')

    user_metadata.user_id = user_metadata.user_id.map(usr_to_idx)
    item_metadata.movie_id = item_metadata.movie_id.map(item_to_idx)

    # combine onehot movie genre cols into single col
    genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    item_metadata['genre'] = item_metadata[genres].idxmax(axis=1)

    # mapping all categorical values to integers without overlap (so we can use a single embedding table)
    X_I_emb = item_metadata.movie_id.values #  get_embedding_matrix(item_metadata, id_col='movie_id', cols=[]) #cols=['genre']
    X_U_emb = user_metadata.user_id.values #  get_embedding_matrix(user_metadata, id_col='user_id', cols=[]) #cols=['gender', 'occupation']
    X_U_cont = None # user_metadata[['age']].values.astype(np.float32)
    X_I_cont = None
    interact = [tuple(x) for x in interactions[['user', 'item', 'timestamp']].values]
    logging.info("embedded item array shape = {}".format(X_I_emb.shape))
    logging.info("embedded user array shape = {}".format(X_U_emb.shape))
    # logging.info("continuous user array shape = {}".format(X_U_cont.shape))
    # logging.info("continuous item array shape = {}".format(X_I_cont.shape))

    return X_U_cont, X_U_emb, X_I_cont, X_I_emb, interact


def evaluate(loader, net, ctx, loss):
    """
    Evaluate the loss function
    :param loader: data loader to be used in evaluation
    :param net: network
    :param context: prediction context
    :param loss: loss function
    """
    epoch_loss = 0
    weight_updates = 0
    for i, (X) in enumerate(loader):
        X_U_cont, X_U_emb, X_I_cont, X_I_emb, X_I_neg_cont, X_I_neg_emb = (x.as_in_context(ctx) for x in X)

        # Forward  pass: loss depends on both positive and negative predictions
        pos_pred = net(X_U_cont, X_U_emb, X_I_cont, X_I_emb)
        neg_pred = net(X_U_cont, X_U_emb, X_I_neg_cont, X_I_neg_emb)
        l = loss(pos_pred, neg_pred)
        epoch_loss += nd.mean(l).asscalar()
        weight_updates += 1
    return epoch_loss / weight_updates


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()

    # get user/item features (embedding or continous) and interactions
    X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions = get_data()

    # build dataloaders
    dataset = InteractionsDataset(X_U_cont, X_U_emb, X_I_cont, X_I_emb, interactions)
    logging.info("{} interactions\t{} users\t{} items".format(len(interactions), dataset.num_user, dataset.num_item))

    train_dataset, test_dataset = dataset.split(test_interactions=args.test_interactions)
    logging.info("{} train interactions\t{} test interactions".format(len(train_dataset), len(test_dataset)))

    train_loader = gluon.data.DataLoader(train_dataset,
                                         batch_size=args.batch_size,
                                         num_workers=multiprocessing.cpu_count())
    test_loader = gluon.data.DataLoader(test_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=multiprocessing.cpu_count())

    # define network, loss and optimizer
    net = RankNet(latent_size=args.embed,
                  total_user_embed_cat=len(np.unique(X_U_emb)),
                  total_item_embed_cat=len(np.unique(X_I_emb)))
    logging.info("Network parameters:\n{}".format(net))

    # select a pairwise loss function
    losses = {'BPR': BprLoss(), 'Hinge': HingeLoss(), 'Pointwise': PointwiseLoss()}
    loss = losses[args.loss]
    logging.info("Loss function = {}".format(args.loss))

    ctx = mx.gpu() if args.gpus > 0 else mx.cpu()
    logging.info("Training on {}".format(ctx))
    net.collect_params().initialize(ctx=ctx)

    if not args.no_hybridize:
        logging.info("Hybridizing network to convert from imperitive to symbolic for increased training speed")
        net.hybridize()

    trainer = gluon.Trainer(params=net.collect_params(),
                            optimizer='Adam',
                            optimizer_params={'learning_rate': args.lr,
                                              'wd': args.l2,
                                              'beta1': args.b1,
                                              'beta2': args.b2})

    # train the network on the data
    logging.info("Training for {} epochs...".format(args.epochs))
    for e in range(args.epochs):
        epoch_loss = 0
        weight_updates = 0
        start = time.time()
        for i, (X) in enumerate(train_loader):

            X_U_cont, X_U_emb, X_I_cont, X_I_emb, X_I_neg_cont, X_I_neg_emb = (x.as_in_context(ctx) for x in X)

            # Forward & backward pass: loss depends on both positive and negative predictions
            with autograd.record():
                pos_pred = net(X_U_cont, X_U_emb, X_I_cont, X_I_emb)
                neg_pred = net(X_U_cont, X_U_emb, X_I_neg_cont, X_I_neg_emb)
                l = loss(pos_pred, neg_pred)
            l.backward()
            trainer.step(2 * args.batch_size)

            epoch_loss += nd.mean(l).asscalar()
            weight_updates += 1
        test_loss = evaluate(test_loader, net, ctx, loss)
        logging.info("Epoch {}:\tTime={:.4}s\tTrain Loss={:.4}\tTest Loss={:.4}".
                     format(e, time.time() - start, epoch_loss / weight_updates, test_loss))

    # train rankings should ignore test interactions
    train_rankings = net.rank(dataset, exclude=test_dataset.sparse_interactions, context=ctx, k=args.k)
    logging.info("Train rankings = {}".format(train_rankings))

    # compute information retrieval metrics
    precisions, recalls = precision_recall(train_rankings, interactions=train_dataset.sparse_interactions)
    logging.info("Model train ranking precision@{}={:.4f}".format(args.k, sum(precisions) / len(precisions)))

    # rank all items for all users (EXCLUDING any interactions in the training set for fair evaluation)
    test_rankings = net.rank(dataset, exclude=train_dataset.sparse_interactions, context=ctx, k=args.k)
    logging.info("Test rankings = {}".format(test_rankings))

    # compute information retrieval metrics
    precisions, recalls = precision_recall(test_rankings, interactions=test_dataset.sparse_interactions)
    logging.info("Model test ranking precision@{}={:.4f}".format(args.k, sum(precisions) / len(precisions)))




