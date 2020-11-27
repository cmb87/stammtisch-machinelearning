import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    concatenate, Input, Add


class Upsample(tf.keras.Model):
    def __init__(self, nfilters, kernelsize, strides=(1,1), activation='relu', padding="same"):
        super(Upsample, self).__init__(name="")
        self.conv1 = Conv2DTranspose(nfilters, kernelsize, activation=activation, padding=padding, strides=strides)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        return x
        
