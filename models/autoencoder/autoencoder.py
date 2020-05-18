#!/usr/bin/env python

# Convolutional autoencoder for 20CRv2c fields.

import os
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle
import numpy
from glob import glob

# Load the model specification
from autoencoderModel import autoencoderModel

# How many epochs to train for
n_epochs = 26

# Target data setup
buffer_size = 100
batch_size = 1

# Training datasets with four variables
input_file_dir = ("%s/ML_GCM/datasets/" + "20CR2c/air.2m/training/") % os.getenv(
    "SCRATCH"
)
t2m_files = glob("%s/*.tfd" % input_file_dir)
n_steps = len(t2m_files)
tr_tfd = tf.constant(t2m_files)

# Create TensorFlow Dataset object from the file names
tr_data = tf.data.Dataset.from_tensor_slices(tr_tfd).repeat(n_epochs)

# Make a 5-variable tensor including the insolation field
def load_tensor(file_name):
    sict = tf.io.read_file(file_name)
    t2m = tf.io.parse_tensor(sict, numpy.float32)
    t2m = tf.reshape(t2m, [79, 159, 1])
    file_name = tf.strings.regex_replace(file_name, "air.2m", "prmsl")
    sict = tf.io.read_file(file_name)
    prmsl = tf.io.parse_tensor(sict, numpy.float32)
    prmsl = tf.reshape(prmsl, [79, 159, 1])
    file_name = tf.strings.regex_replace(file_name, "prmsl", "uwnd.10m")
    sict = tf.io.read_file(file_name)
    uwnd = tf.io.parse_tensor(sict, numpy.float32)
    uwnd = tf.reshape(uwnd, [79, 159, 1])
    file_name = tf.strings.regex_replace(file_name, "uwnd.10m", "vwnd.10m")
    sict = tf.io.read_file(file_name)
    vwnd = tf.io.parse_tensor(sict, numpy.float32)
    vwnd = tf.reshape(vwnd, [79, 159, 1])
    file_name = tf.strings.regex_replace(file_name, "vwnd.10m", "insolation")
    fdte = tf.strings.substr(file_name, -17, 17)
    mnth = tf.strings.substr(fdte, 5, 2)
    dy = tf.strings.substr(fdte, 8, 2)
    dy = tf.cond(
        tf.math.equal(mnth + dy, "0229"), lambda: tf.constant("28"), lambda: dy
    )
    file_name = (
        tf.strings.substr(file_name, 0, tf.strings.length(file_name) - 17)
        + "1969-"
        + mnth
        + "-"
        + dy
        + tf.strings.substr(fdte, tf.strings.length(fdte) - 7, 7)
    )
    sict = tf.io.read_file(file_name)
    insol = tf.io.parse_tensor(sict, numpy.float32)
    insol = tf.reshape(insol, [79, 159, 1])
    ict = tf.concat([t2m, prmsl, uwnd, vwnd, insol], 2)  # Now [79,159,5]
    ict = tf.reshape(ict, [79, 159, 5])
    return ict


tr_target = tr_data.map(load_tensor)
tr_source = tr_data.map(load_tensor)

tr_data = tf.data.Dataset.zip((tr_source, tr_target))
tr_data = tr_data.shuffle(buffer_size).batch(batch_size)

# Same for the test dataset
input_file_dir = ("%s/ML_GCM/datasets/" + "20CR2c/air.2m/test/") % os.getenv("SCRATCH")
t2m_files = glob("%s/*.tfd" % input_file_dir)
test_steps = len(t2m_files)
test_tfd = tf.constant(t2m_files)
test_data = tf.data.Dataset.from_tensor_slices(test_tfd).repeat(n_epochs)
test_target = test_data.map(load_tensor)
test_source = test_data.map(load_tensor)
test_data = tf.data.Dataset.zip((test_source, test_target))
test_data = test_data.batch(batch_size)

# Instantiate the model
autoencoder = autoencoderModel()

# Save the model weights and the history state after every epoch
history = {}
history["loss"] = []
history["val_loss"] = []


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        save_dir = ("%s/ML_GCM/autoencoder/" + "Epoch_%04d") % (
            os.getenv("SCRATCH"),
            epoch,
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.model.save_weights("%s/ckpt" % save_dir)
        history["loss"].append(logs["loss"])
        history["val_loss"].append(logs["val_loss"])
        history_file = "%s/history.pkl" % save_dir
        pickle.dump(history, open(history_file, "wb"))


class ReduceNoise(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        K.set_value(
            self.model.noiseStdDev, self.model.noiseStdDev * 0.9,
        )


# Train the autoencoder
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adadelta(
        learning_rate=1.0, rho=0.95, epsilon=1e-07, name="Adadelta"
    ),
    loss="mean_squared_error",
)
history = autoencoder.fit(
    x=tr_data,
    epochs=n_epochs,
    steps_per_epoch=n_steps // batch_size,
    validation_data=test_data,
    validation_steps=test_steps // batch_size,
    verbose=1,
    callbacks=[ReduceNoise(), CustomSaver()],
)
