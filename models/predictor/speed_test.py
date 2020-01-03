#!/usr/bin/env python

# How fast is the direct-forecast predictor.

import tensorflow as tf
tf.enable_eager_execution()
import numpy
import sys
import os
import datetime

import iris
import IRData.twcr as twcr


from timeit import default_timer as timer

# Start on Jan 1st, 1989
dtstart=datetime.datetime(1989,1,1,0)

# Predictor model epoch
epoch=25

sys.path.append('%s/../../lib/' % os.path.dirname(__file__))
from insolation import load_insolation
from geometry import to_analysis_grid
from normalise import normalise_insolation
from normalise import normalise_t2m
from normalise import normalise_prmsl
from normalise import normalise_wind

# Load the starting data and reshape into a state vector tensor
prmsl = twcr.load('prmsl',dtstart,version='2c')
prmsl = to_analysis_grid(prmsl.extract(iris.Constraint(member=1)))
prmsl = tf.convert_to_tensor(normalise_prmsl(prmsl.data),numpy.float32)
prmsl = tf.reshape(prmsl,[79,159,1])
t2m = twcr.load('air.2m',dtstart,version='2c')
t2m = to_analysis_grid(t2m.extract(iris.Constraint(member=1)))
t2m = tf.convert_to_tensor(normalise_t2m(t2m.data),numpy.float32)
t2m = tf.reshape(t2m,[79,159,1])
u10m = twcr.load('uwnd.10m',dtstart,version='2c')
u10m = to_analysis_grid(u10m.extract(iris.Constraint(member=1)))
u10m = tf.convert_to_tensor(normalise_wind(u10m.data),numpy.float32)
u10m = tf.reshape(u10m,[79,159,1])
v10m = twcr.load('vwnd.10m',dtstart,version='2c')
v10m = to_analysis_grid(v10m.extract(iris.Constraint(member=1)))
v10m = tf.convert_to_tensor(normalise_wind(v10m.data),numpy.float32)
v10m = tf.reshape(v10m,[79,159,1])
insol = to_analysis_grid(load_insolation(dtstart.year,dtstart.month,
                                         dtstart.day,dtstart.hour))
insol = tf.convert_to_tensor(normalise_insolation(insol.data),numpy.float32)
insol = tf.reshape(insol,[79,159,1])

state_v = tf.concat([t2m,prmsl,u10m,v10m,insol],2) # Now [79,159,5]
state_v = tf.reshape(state_v,[1,79,159,5])

# Load the predictor
model_save_file=("%s/ML_GCM/predictor/"+
                  "Epoch_%04d/predictor") % (
                      os.getenv('SCRATCH'),epoch)
predictor=tf.keras.models.load_model(model_save_file,compile=False)
# Also load the encoder (to get the latent state)
model_save_file=("%s/ML_GCM/predictor/"+
                  "Epoch_%04d/encoder") % (
                      os.getenv('SCRATCH'),epoch)
encoder=tf.keras.models.load_model(model_save_file,compile=False)

n_rep=500
elapsed = sys.maxsize
start = timer()
for _ in range(n_rep):
    state_v = predictor.predict_on_batch(state_v)
elapsed = min(timer() - start, elapsed)/n_rep

print("Time per prediction = %d ms" % int(1000*elapsed))
