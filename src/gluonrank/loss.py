from mxnet import nd
from mxnet.gluon import HybridBlock


class Loss(HybridBlock):
    """Base class for loss.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    """
    def __init__(self, weight, batch_axis, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self._weight = weight
        self._batch_axis = batch_axis

    def __repr__(self):
        s = '{name}(batch_axis={_batch_axis}, w={_weight})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


    def hybrid_forward(self, F, x, *args, **kwargs):
        """Overrides to construct symbolic graph for this `Block`.

        Parameters
        ----------
        x : Symbol or NDArray
            The first input tensor.
        *args : list of Symbol or list of NDArray
            Additional input tensors.

        """
        # pylint: disable= invalid-name
        raise NotImplementedError


class PointwiseLoss(Loss):
    """
    Logistic loss function
    """

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(PointwiseLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, positive_predictions, negative_predictions):
        """
        :param positive_predictions: ndarray containing predictions for known positive items
        :param negative_predictions: ndarray containing predictions for randomly sampled items
        :return: ndarray containing the loss for each training example
        """
        positives_loss = (1.0 - positive_predictions.sigmoid())
        negatives_loss = negative_predictions.sigmoid()
        loss = positives_loss + negatives_loss
        return loss


class BprLoss(Loss):
    """
    Bayesian personalized ranking pairwise loss function
    """

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(BprLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, positive_predictions, negative_predictions):
        """
        :param positive_predictions: ndarray containing predictions for known positive items
        :param negative_predictions: ndarray containing predictions for randomly sampled items
        :return: ndarray containing the loss for each training example
        """
        loss = (1.0 - (positive_predictions - negative_predictions).sigmoid())
        return loss


class HingeLoss(Loss):
    """
    Bayesian personalized ranking pairwise loss function
    """

    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(HingeLoss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, positive_predictions, negative_predictions):
        """
        :param positive_predictions: ndarray containing predictions for known positive items
        :param negative_predictions: ndarray containing predictions for randomly sampled items
        :return: ndarray containing the loss for each training example
        """
        loss = F.maximum(F.zeros_like(positive_predictions), negative_predictions - positive_predictions + 1.0)
        return loss
