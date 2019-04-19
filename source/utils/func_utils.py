from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # no tensorflow module
    tf = None


def sigmoid(x):
    """Compute sigmoid function: y = 1 / (1 + exp(-x))
    """
    max0 = np.maximum(x, 0)
    return np.exp(x - max0) / (np.exp(x - max0) + np.exp(-max0))


def softmax(x):
    x = np.exp(x - np.max(x, 1, keepdims=True))
    return x / (np.sum(x, 1, keepdims=True) + np.finfo(np.float32).eps)


def logsumone(x):
    """Compute log(1 + exp(x))
    """
    max0 = np.maximum(x, 0)
    return np.log(np.exp(-max0) + np.exp(x - max0)) + max0


def tf_logsumone(x):
    """Compute log(1 + exp(x))
    """
    max0 = tf.maximum(x, 0)
    return tf.log(tf.exp(-max0) + tf.exp(x - max0)) + max0
