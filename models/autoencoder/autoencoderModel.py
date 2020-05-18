# Model specification for the variational autoencoder

import tensorflow as tf
import tensorflow.keras.backend as K

# This file provides a subclass of tf.keras.Model that serves as an
#  autoencoder for the 20CRv2c fields. The model has two sub-models
#  (an encoder and a generator) and is variational (adds noise to it's
#  latent space) in training but not in prediction.

# import this file, instantiate an instance of the autoencoderModel
#  class, and then either train it and save the weights, or load
#  pre-trained weights and use it for prediction.

# Dimensionality of latent space
latent_dim = 100

# Specify and encoder model to pack the imput into a latent space
class encoderModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(encoderModel, self).__init__()
        # 4 strided convolutions
        self.conv1A = tf.keras.layers.Conv2D(5, (3, 3), padding="same")
        self.act1A = tf.keras.layers.ELU()
        self.d1A = tf.keras.layers.Dropout(0.3)
        self.conv1B = tf.keras.layers.Conv2D(
            10, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        self.d1B = tf.keras.layers.Dropout(0.3)
        self.conv1C = tf.keras.layers.Conv2D(
            30, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1C = tf.keras.layers.ELU()
        self.d1C = tf.keras.layers.Dropout(0.3)
        self.conv1D = tf.keras.layers.Conv2D(
            90, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1D = tf.keras.layers.ELU()
        self.d1D = tf.keras.layers.Dropout(0.3)
        # reshape to 1d
        self.flatten = tf.keras.layers.Reshape(target_shape=(9 * 19 * 90,))
        # reduce to latent space size
        self.pack_to_l = tf.keras.layers.Dense(latent_dim,)

    def call(self, inputs):
        x = self.conv1A(inputs)
        x = self.act1A(x)
        x = self.d1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.d1B(x)
        x = self.conv1C(x)
        x = self.act1C(x)
        x = self.d1C(x)
        x = self.conv1D(x)
        x = self.act1D(x)
        x = self.d1D(x)
        x = self.flatten(x)
        x = self.pack_to_l(x)
        # Normalise latent space to mean=0, sd=1
        x = x - K.mean(x)
        x = x / K.std(x)
        return x


# Specify a generator model to make the output from a latent vector
class generatorModel(tf.keras.Model):
    def __init__(self):
        # parent constructor
        super(generatorModel, self).__init__()
        # reshape latent space as 3d seed for deconvolution
        self.unpack_from_l = tf.keras.layers.Dense(9 * 19 * 90,)
        self.unflatten = tf.keras.layers.Reshape(target_shape=(9, 19, 90,))
        # Three transpose convolution layers to recover input shape
        self.conv1A = tf.keras.layers.Conv2DTranspose(
            90, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1A = tf.keras.layers.ELU()
        self.conv1B = tf.keras.layers.Conv2DTranspose(
            30, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1B = tf.keras.layers.ELU()
        self.conv1C = tf.keras.layers.Conv2DTranspose(
            10, (3, 3), strides=(2, 2), padding="valid"
        )
        self.act1C = tf.keras.layers.ELU()
        self.finish = tf.keras.layers.Conv2D(5, (3, 3), padding="same")

    def call(self, inputs):
        x = self.unpack_from_l(inputs)
        x = self.unflatten(x)
        x = self.conv1A(x)
        x = self.act1A(x)
        x = self.conv1B(x)
        x = self.act1B(x)
        x = self.conv1C(x)
        x = self.act1C(x)
        x = self.finish(x)
        return x


# Autoencoder model is the encoder and generator run in sequence
#  with some noise added between them in training.
class autoencoderModel(tf.keras.Model):
    def __init__(self):
        super(autoencoderModel, self).__init__()
        self.encoder = encoderModel()
        self.generator = generatorModel()
        self.noiseMean = tf.Variable(0.0, trainable=False)
        self.noiseStdDev = tf.Variable(0.5, trainable=False)

    def call(self, inputs, training=None):
        # encode to latent space
        x = self.encoder(inputs)
        if training:
            # Add noise to the latent space representation
            x += K.random_normal(
                K.shape(x),
                mean=K.get_value(self.noiseMean),
                stddev=K.get_value(self.noiseStdDev),
            )
            # Re-normalise latent space
            x = x - K.mean(x)
            x = x / K.std(x)
        # Generate real space representation from latent space
        x = self.generator(x)
        return x
