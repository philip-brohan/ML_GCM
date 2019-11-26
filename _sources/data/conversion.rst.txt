Converting training data for use in TensorFlow
==============================================

To train models on the downloaded data, we need to get it out of `netCDF <https://www.unidata.ucar.edu/software/netcdf>`_ and into a file format that `TensorFlow <https://www.tensorflow.org/>`_ supports. This should probably be a `TFRecord file <https://www.tensorflow.org/tutorials/load_data/tfrecord>`_ but I was quite unable to work out how to use them, so I'm using individual files of serialised `tensors <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_.

The basic plan is:

1. Each input file will contain the data for one variable, at one point in time, appropriately arranged and normalised for use by Tensorflow . (Making the file can be slow, using it should be fast, so do all conversions at this point).
2. We want the input files to be independent, so only make one every five days or so, but arrange them to sample the annual and diurnal cycles uniformly. I make one set of files every 126 hours, through the test and training periods.
3. The structure of these files should be matched to the `model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_ that is using them. Here, this means regridding to a resolution that works well with strided convolutions, and rotation to put the important parts of the world (the UK) in the middle.
4. This makes a large batch of serialised tensor files, which can be combined into a `TensorFlow Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ for model training.

Script to make a single tensor file:

.. literalinclude:: ../../data/prepare_training_tensors/make_training_tensor.py

To run this script for every required variable and timepoint:

.. literalinclude:: ../../data/prepare_training_tensors/makeall.py

This script produces a list of commands, which can be run in serial or parallel.

The insolation data requires slightly different treatment (every 6 hours for a year, rather than every 126 hours for 40 years), but the fundamental process is the same:

.. literalinclude:: ../../data/prepare_training_tensors/make_insolation_tensor.py
.. literalinclude:: ../../data/prepare_training_tensors/makeall.insolation.py

This process provides data for the autoencoder (where input is the same as output). Other models will require additional (but similar) data - the +6hour predictor, for example, requires the same data but with a 6hour offset applied.

Library functions used:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Regridding <../lib/geometry>
   Normalisation <../lib/normalise>
