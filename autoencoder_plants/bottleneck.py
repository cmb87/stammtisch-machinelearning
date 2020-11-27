import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, \
    concatenate, Input, Add


class Nonbt1d(tf.keras.Model):
    def __init__(self, nfilters, dilation=(1,1)):
        super(Nonbt1d, self).__init__(name="")
        self.conv1 = Conv2D(nfilters, (3, 1), activation='relu', padding='same', dilation_rate=dilation)
        self.conv2 = Conv2D(nfilters, (1, 3), activation='relu', padding='same', dilation_rate=dilation)
        self.conv3 = Conv2D(nfilters, (3, 1), activation='relu', padding='same', dilation_rate=dilation)
        self.conv4 = Conv2D(nfilters, (1, 3), activation='linear', padding='same', dilation_rate=dilation)
        self.drop = Dropout(0.4)

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.conv4(x, training=training)
        x = self.drop(x, training=training)
        x += input_tensor
        return tf.nn.relu(x)

