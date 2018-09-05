"""
Sparsity Regularization layer in Keras with TensorFlow backend
Reference:  Z. Liu et. al. Learning efficient convolutional networks through network slimming. ICCV2017
author: Tianzhong
09/03/2018
"""

from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import keras.backend as K
from keras import initializers
from keras import regularizers
from keras.regularizers import l1
import tensorflow as tf


class SparsityRegularization(Layer):
    def __init__(self, l1=0.01, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = -1
        else:
            self.axis = 1
        self.l1 = l1
        super(SparsityRegularization, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        self.gamma = self.add_weight(shape=(dim,),
                                     initializer=initializers.get('ones'),
                                     name='gamma',
                                     regularizer=regularizers.get(l1(l=self.l1)),
                                     trainable=True
                                     )
        self.trainable_weights = [self.gamma]
        super(SparsityRegularization, self).build(input_shape)

    def call(self, inputs, mask=None):
        return inputs * self.gamma

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'L1': self.l1
        }
        base_config = super(SparsityRegularization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
