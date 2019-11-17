#!/usr/bin/env python

# Convolutional autoencoder for 20CRv2c fields.

import os
import sys
import tensorflow as tf
#tf.enable_eager_execution()
from tensorflow.data import Dataset
import pickle
import numpy
from glob import glob

# How many epochs to train for
n_epochs=26

# How big a latent space
latent_dim=100

# Target data setup
buffer_size=100
batch_size=1

# Training datasets with four variables
input_file_dir=(("%s/ML_GCM/datasets/" +
                "20CR2c/air.2m/training/") %
                   os.getenv('SCRATCH'))
t2m_files=glob("%s/*.tfd" % input_file_dir)
n_steps=len(t2m_files)
tr_tfd = tf.constant(t2m_files)

# Create TensorFlow Dataset object from the file names
tr_data = Dataset.from_tensor_slices(tr_tfd).repeat(n_epochs)

# Make a 5-variable tensor including the insolation field
def load_tensor(file_name):
    sict  = tf.read_file(file_name)
    t2m = tf.parse_tensor(sict,numpy.float32)
    t2m = tf.reshape(t2m,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'air.2m','prmsl')
    sict  = tf.read_file(file_name)
    prmsl = tf.parse_tensor(sict,numpy.float32)
    prmsl = tf.reshape(prmsl,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'prmsl','uwnd.10m')
    sict  = tf.read_file(file_name)
    uwnd  = tf.parse_tensor(sict,numpy.float32)
    uwnd  = tf.reshape(uwnd,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'uwnd.10m','vwnd.10m')
    sict  = tf.read_file(file_name)
    vwnd  = tf.parse_tensor(sict,numpy.float32)
    vwnd  = tf.reshape(vwnd,[79,159,1])
    file_name = tf.strings.regex_replace(file_name,
                                      'vwnd.10m','insolation')
    fdte = tf.strings.substr(file_name,-17,17)
    mnth = tf.strings.substr(fdte,5,2)
    dy   = tf.strings.substr(fdte,8,2)
    dy = tf.cond(tf.math.equal(mnth+dy,'0229'),
                               lambda: tf.constant('28'),lambda: dy)
    file_name=(tf.strings.substr(file_name,0,tf.strings.length(file_name)-17)+
                '1969-'+mnth+'-'+dy+tf.strings.substr(fdte,tf.strings.length(fdte)-7,7))
    sict  = tf.read_file(file_name)
    insol = tf.parse_tensor(sict,numpy.float32)
    insol = tf.reshape(insol,[79,159,1])
    ict = tf.concat([t2m,prmsl,uwnd,vwnd,insol],2) # Now [79,159,5]
    ict = tf.reshape(ict,[79,159,5])
    return ict


tr_target = tr_data.map(load_tensor)
tr_source = tr_data.map(load_tensor)

tr_data = Dataset.zip((tr_source, tr_target))
tr_data = tr_data.shuffle(buffer_size).batch(batch_size)

# Same for the test dataset
input_file_dir=(("%s/ML_GCM/datasets/" +
                "20CR2c/air.2m/test/") %
                   os.getenv('SCRATCH'))
t2m_files=glob("%s/*.tfd" % input_file_dir)
test_steps=len(t2m_files)
test_tfd = tf.constant(t2m_files)
test_data = Dataset.from_tensor_slices(test_tfd).repeat(n_epochs)
test_target = test_data.map(load_tensor)
test_source = test_data.map(load_tensor)
test_data = Dataset.zip((test_source, test_target))
test_data = test_data.batch(batch_size)

# Add noise to latent vector
def noise(encoded):
    epsilon = tf.keras.backend.random_normal(tf.keras.backend.shape(encoded),
                                             mean=0.0,stddev=0.2)
    return encoded+epsilon
# Normalise the latent vector
def stdise(encoded):
    encoded = encoded-tf.keras.backend.mean(encoded)
    encoded = encoded/tf.keras.backend.std(encoded)
    return encoded

# Input placeholder
original = tf.keras.layers.Input(shape=(79,159,5,), name='encoder_input')
# Encoding layers
x = tf.keras.layers.Conv2D(5, (3, 3), padding='same')(original)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(10, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(30, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv2D(90, (3, 3), strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Dropout(0.3)(x)
# N-dimensional latent space representation
x = tf.keras.layers.Reshape(target_shape=(9*19*90,))(x)
x = tf.keras.layers.Dense(latent_dim,)(x)
encoded  = tf.keras.layers.Lambda(stdise, output_shape=(latent_dim,))(x)

# Define an encoder model
encoder = tf.keras.models.Model(original, encoded, name='encoder')

# Add a noise layer to make the encoder variational
noise_i = tf.keras.layers.Input(shape=(latent_dim,), name='noise_input')
noise_o  = tf.keras.layers.Lambda(noise, output_shape=(latent_dim,))(noise_i)
noise_m = tf.keras.models.Model(noise_i, noise_o, name='noise_m')

# Decoding layers
encoded = tf.keras.layers.Input(shape=(latent_dim,), name='decoder_input')
x = tf.keras.layers.Dense(9*19*90,)(encoded)
x = tf.keras.layers.Reshape(target_shape=(9,19,90,))(x)
x = tf.keras.layers.Conv2DTranspose(90, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(30, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
x = tf.keras.layers.Conv2DTranspose(10, (3, 3),  strides= (2,2), padding='valid')(x)
x = tf.keras.layers.ELU()(x)
decoded = tf.keras.layers.Conv2D(5, (3, 3), padding='same')(x) # Will be 75x159x5 

# Define a generator (decoder) model
generator = tf.keras.models.Model(encoded, decoded, name='generator')

# Combine the encoder and the generator into an autoencoder
output = generator(encoder(original))
autoencoder = tf.keras.models.Model(inputs=original, outputs=output, name='autoencoder')

# Combine the encoder, the noise, and the generator, into a variational autoencoder
output_v = generator(noise_m(encoder(original)))
var_autoencoder = tf.keras.models.Model(inputs=original, outputs=output_v, name='var_autoencoder')

var_autoencoder.compile(optimizer='adadelta',loss='mean_squared_error')

# Save model and history state after every epoch
history={}
history['loss']=[]
history['val_loss']=[]
class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir=("%s/ML_GCM/autoencoder/"+
                   "Epoch_%04d") % (
                         os.getenv('SCRATCH'),epoch)
        if not os.path.isdir(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        for model in ['var_autoencoder','autoencoder','encoder','generator']:
            if not os.path.isdir(os.path.dirname("%s/%s" % (save_dir,model))):
                os.makedirs(os.path.dirname("%s/%s" % (save_dir,model)))
        tf.keras.models.save_model(var_autoencoder,"%s/var_autoencoder" % save_dir)
        tf.keras.models.save_model(autoencoder,"%s/autoencoder" % save_dir)
        tf.keras.models.save_model(encoder,"%s/encoder" % save_dir)
        tf.keras.models.save_model(generator,"%s/generator" % save_dir)
        history['loss'].append(logs['loss'])
        history['val_loss'].append(logs['val_loss'])
        history_file=("%s/ML_GCM/autoencoder/"+
                      "history_to_%04d.pkl") % (
                         os.getenv('SCRATCH'),epoch)
        pickle.dump(history, open(history_file, "wb"))
        
saver = CustomSaver()

# Train the autoencoder
history=var_autoencoder.fit(x=tr_data,
                            epochs=n_epochs,
                            steps_per_epoch=n_steps,
                            validation_data=test_data,
                            validation_steps=test_steps,
                            verbose=1,
                            callbacks=[saver])

