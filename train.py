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


def parse_args():
    """
    Parse script arguments
    :return: dict of args
    """
    parser = argparse.ArgumentParser(description="Train a ranking model on the movielense data")

    group = parser.add_argument_group('Computation arguments')
    parser.add_argument('--gpus', type=int, default=0,
                        help='num of gpus to distribute  model training on. 0 for cpu')
    parser.add_argument('--no-hybridize', action='store_true',
                        help='use symbolic network graph for increased computational eff')

    group = parser.add_argument_group('Network architecture')

    group = parser.add_argument_group('Regularization arguments')
    group.add_argument('--dropout', type=float, default=0,
                       help='dropout probability for fully connected layers')
    group.add_argument('--l2', type=float, default=0.0,
                       help='weight regularization penalty')

    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--epochs', type=int, default=10,
                       help='num of times to loop through training data')
    group.add_argument('--batch-size', type=int, default=512,
                       help='number of training examples per batch')
    group.add_argument('--lr', type=float, default=0.01,
                       help='optimizer learning rate')
    group.add_argument('--b1', type=float, default=0.9,
                       help='optimizer first moment')
    group.add_argument('--b2', type=float, default=0.999,
                       help='optimizer second moment')

    group = parser.add_argument_group('Evaluation arguments')
    group.add_argument('--test-frac', type=int, default=0.2,
                       help='fraction of data to be used for evaluation')
    return parser.parse_args()


def reindex_col(df, col):
    """
    Maps a column of integers to increment from 0
    :param df: pandas dataframe
    :param col: column to be reindexed
    :return: pandas dataframe & mapping dict
    """
    mapping_dict = {k: v for v, k in enumerate(df[col].unique())}
    df = df.replace(col, mapping_dict)
    return df, mapping_dict


def get_data():
    interactions = pd.read_csv('./data/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'])
    user_metadata = pd.read_csv('./data/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
    cols = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action',
            'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
           'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    item_metadata = pd.read_csv('./data/u.item', sep='|', encoding="ISO-8859-1", names=cols)

    interactions, usr_to_idx = reindex_col(interactions, 'user')
    interactions, item_to_idx = reindex_col(interactions, 'item')
    user_metadata['user_id'].replace('user_id', usr_to_idx)
    item_metadata['movie id'].replace('movie id', item_to_idx)


    subset = interactions[['user', 'item', 'timestamp']]
    subset[['user']] = subset[['user']] - 1
    subset[['item']] = subset[['item']] - 1
    int = [tuple(x) for x in subset.values]
    subset = pd.get_dummies(user_metadata[['user_id', 'age', 'gender']]).sort_values('user_id').drop('user_id', axis=1)
    usr = [tuple(x) for x in subset.values]
    usr = np.array(usr, dtype='float32')
    subset = pd.get_dummies(item_metadata[['movie id', 'movie title', 'Documentary']]).sort_values('movie id').drop('movie id', axis=1)
    item = [tuple(x) for x in subset.values]
    item = np.array(item, dtype='float32')
    return usr, item, int


def evaluate(epoch, loader, net, context, loss):
    """
    Evaluate the loss function
    :param loader: data loader to be used in evaluation
    :param net: network
    :param context: prediction context
    :param loss: loss function
    """
    epoch_loss = 0
    weight_updates = 0
    start = time.time()
    for i, ((user_features, item_features, neg_item_features), label) in enumerate(loader):

        user_features = user_features.as_in_context(context)
        item_features = item_features.as_in_context(context)
        neg_item_features = neg_item_features.as_in_context(context)
        label = label.as_in_context(context)

        pred = net(user_features, item_features, neg_item_features)
        l = loss(pred, label)

        epoch_loss += nd.mean(l).asscalar()
        weight_updates += 1
    logging.info("Epoch {}: Time = {:.4}s Validation Loss = {:.4}".
                 format(epoch, time.time() - start, epoch_loss / weight_updates))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()

    # build data loader to feed network
    usr, item, int = get_data()
    logging.info("{} interactions,for {} users on {} items".format(len(int), usr.shape[0], item.shape[0]))

    dataset = InteractionsDataset(user_features=usr, item_features=item, interactions=int)
    train_dataset, test_dataset = dataset.split(test_frac=args.test_frac)
    train_loader = gluon.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count())
    test_loader = gluon.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count())

    # define network, loss and optimizer
    net = RankNet(latent_size=32)
    logging.info("Network parameters:\n{}".format(net))
    loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)

    ctx = mx.gpu() if args.gpus > 0 else mx.cpu()
    logging.info("Training on {}".format(ctx))
    net.collect_params().initialize(ctx=ctx)

    if not args.no_hybridize:
        logging.info("Hybridizing network to convert from imperitive to symbolic for increased training speed")
        net.hybridize()

    trainer = gluon.Trainer(params=net.collect_params(),
                            optimizer='Adam',
                            optimizer_params={'learning_rate': args.lr, 'beta1': args.b1, 'beta2': args.b2})

    # train the network on the data
    for e in range(args.epochs):
        epoch_loss = 0
        weight_updates = 0
        start = time.time()
        for i, ((user_features, item_features, neg_item_features), label) in enumerate(train_loader):

            user_features = user_features.as_in_context(ctx)
            item_features = item_features.as_in_context(ctx)
            neg_item_features = neg_item_features.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                pred = net(user_features, item_features, neg_item_features)
                l = loss(pred, label)

            l.backward()
            trainer.step(args.batch_size)
            epoch_loss += nd.mean(l).asscalar()
            weight_updates += 1
        logging.info("Epoch {}: Time = {:.4}s Train Loss = {:.4}".
                     format(e, time.time() - start, epoch_loss / weight_updates))
        evaluate(e, test_loader, net, ctx, loss)
