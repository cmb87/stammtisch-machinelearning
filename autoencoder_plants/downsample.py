import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    Concatenate, Input, Add


class Downsample(tf.keras.Model):
    def __init__(self, nfilters, kernelsize, maxpoolsize, strides=(2,2), activation='relu', padding="same"):
        super(Downsample, self).__init__(name="")

        self.nfilters = nfilters
        self.kernelsize = kernelsize
        self.maxpoolsize = maxpoolsize
        self.strides = strides
        self.activation = activation
        self.padding = padding

    def build(self, input_shape):

        fin = input_shape[3]
        fconv = self.nfilters - fin
        self.conv1 = Conv2D(fconv, self.kernelsize, activation=self.activation, padding=self.padding, strides=self.strides)
        self.pool1 = MaxPooling2D(self.maxpoolsize, padding=self.padding, strides=self.strides)
        self.cat = Concatenate()


    def call(self, input_tensor, training=False):
        x1 = self.conv1(input_tensor, training=training)
        x2 = self.pool1(input_tensor)
        return self.cat([x1,x2])



