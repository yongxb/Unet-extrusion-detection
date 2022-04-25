import tensorflow as tf

from tensorpack.utils.develop import log_deprecated
from tensorpack.models.batch_norm import BatchNorm
from tensorpack.models.common import layer_register
from tensorpack.models import Conv2D, ConcatWith


@layer_register(use_scope=None)
def BNSwish(x, name=None):
    """
    A shorthand of BatchNormalization + ReLU.

    Args:
        x (tf.Tensor): the input
        name: deprecated, don't use.
    """
    if name is not None:
        log_deprecated("BNReLU(name=...)", "The output tensor will be named `output`.")

    x = BatchNorm('bn', x)
    x = tf.nn.swish(x, name=name)
    return x


def batchnorm_function():
    # See https://arxiv.org/abs/1706.02677.
    return lambda x, name=None: BatchNorm('bn', x)  # , gamma_initializer=tf.zeros_initializer())


def batchnorm_swish(layer, name=None):
    layer = BatchNorm('bn', layer)
    layer = tf.nn.swish(layer, name=name)
    return layer


def reshape_shortcut(shortcut, layer, stride, upsample=False, activation=tf.identity):
    # assumes 'NHWC'
    if upsample is True:
        shape = layer.get_shape().as_list()
        shortcut = tf.image.resize_images(shortcut, [shape[1], shape[2]], align_corners=True)

    if layer.get_shape().as_list() != shortcut.get_shape().as_list():  # change dimension when channel is not the same
        return Conv2D('convshortcut', shortcut, layer.get_shape().as_list()[-1], 1, stride=stride,
                      activation=activation)
    else:
        return shortcut


def concat_block(name, layer_1, layer_2):
    with tf.compat.v1.variable_scope(name):
        layer_1 = ConcatWith(layer_1, layer_2, -1)
        return layer_1
