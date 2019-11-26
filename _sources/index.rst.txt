Weather Forecasting without the difficult bits
==============================================

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/363005763?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Big data: Near-surface temperature, wind, and precipitation, from the Met Office global analysis <visualisation/MO_global_analysis/index>

`Modern weather-forecast models <https://en.wikipedia.org/wiki/Unified_Model>`_ are amazing: amazingly powerful, amazingly accurate - amazingly complicated, amazingly expensive to run and to develop, amazingly difficult to use and to experiment with. Quite often, I'd rather have something less amazing, but much faster and easier to use. Modern `machine learning <https://en.wikipedia.org/wiki/Machine_learning>`_ methods offer sophisticated statistical approximators even to very complex systems like the weather, and we now have hundreds of years of reanalysis output to train them on. How good a model can we build without using any physics, dynamics, chemistry, etc. at all?

My ambition here is to build a `General Circulation Model (GCM) <https://en.wikipedia.org/wiki/General_circulation_model>`_ using *only* machine learning (ML) methods - no hand-specified physics, dynamics, chemistry etc. at all. To that end I'm going to take an existing model, and train a machine learning system to emulate it - the ML version will be less capable as a model of the atmosphere, but should make up for this by being *much* faster, and easier to use. A `previous experiment along these lines <https://github.com/philip-brohan/weather2weather>`_
had `limited success <https://vimeo.com/275778137>`_ but did suggest that good results were possible given the right model architecture.

It would be nice to use the `Met Office UM <https://en.wikipedia.org/wiki/Unified_Model>`_ (see video above) as the model to be emulated, but for a first attempt I'm going to pick an easier target:

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/369615737?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Medium-sized data: Near-surface temperature, wind, and mean-sea-level pressure, from the Twentieth Century Reanalysis version 2c <visualisation/20CRv2c_global_analysis/index>

The `Twentieth Century Reanalysis <https://www.esrl.noaa.gov/psd/data/20thC_Rean/>`_ `version 2c (20CRv2c) <https://www.esrl.noaa.gov/psd/data/gridded/data.20thC_ReanV2c.html>`_ provides data every 6-hours on a 2-degree grid, for the past 150+ years. So this is consistent data, at a manageable volume, for a long period. (I have used 20CRv2c ensemble member 1 data from 1969-2005 as training data, and 2006-on as validation).

I chose to look at 4 near-surface variables: 2-meter air temperature, mean-sea-level-presure, meridional wind and zonal wind. (Precipitation presents too many complications for a first attempt, so I did not use it here). 

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Download training data <data/download>

I'm going to use the `Tensorflow <https://www.tensorflow.org/>`_ platform to build my ML models (an arbitrary choice), so I need to convert the 20CRv2c data from `netCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ into `tensors <https://www.tensorflow.org/guide/tensor>`_.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Process training data for TensorFlow<data/conversion>

20CRv2c is a low resolution analysis (2x2 degree), and I'm using only four variables, but even so it has a state vector with 180*90*4=64,800 dimensions. Directly generating forecasts in such a high-dimensional space will be difficult to do reliably, so the first thing to do is dimension reduction. I need to build an `encoder-decoder model <https://d2l.ai/chapter_modern_recurrent-networks/encoder-decoder.html>`_, specifically, an `autoencoder <https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist>`_. This creates a compressed representation of the large state vector in terms of a small number of 'features', and I expect working in the resulting feature representation to be much easier than working in the full state space. After `some experimentation <http://brohan.org/Machine-Learning/>`_ I ended up with a :doc:`variational autoencoder with eight convolutional layers, encoding the features as a 100-dimensional latent space <models/autoencoder/index>`.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   A variational autoencoder <models/autoencoder/index>

This autoencoder was very successful in building a low-dimensional representation of the 20CRv2c data:

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/369615958?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Small data: Near-surface temperature, wind, and mean-sea-level pressure, from the Twentieth Century Reanalysis version 2c, after compression into a 100-dimensional latent space. <visualisation/20CRv2c_latent_space/index>

But the real virtue of the 100-dimensional latent space encoding is that it allows me to use the 'generator' half of the autoencoder as a generative model: it provides a method for making new, internally consistent weather states. So we can convert the autoencoder into a predictor:


.. toctree::
   :titlesonly:
   :maxdepth: 1

   A generative model <models/generator/generator>
   A +6hr predictor <models/predictor/index>

And we can then use the predictor as a GCM, by repeatedly re-using its output as its input:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   A Machine-Learning GCM using the +6hr predictor <models/GCM/index>
   
.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/371672143?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    </table>
    </center>

.. toctree::
   :titlesonly:
   :maxdepth: 1

    Simulated data: Near-surface temperature, wind, and mean-sea-level pressure, from the Machine-Learning GCM. <visualisation/GCM_video/index>

This system produces credible temperature, pressure, and wind, using only machine learning - no physics, no dynamics, no chemistry. It's not quite at the state of the art, but the model was very quick and easy to produce: the model specification is only a few dozen lines of Python, trained in 20 minutes on my laptop, and it runs at more than 100,000 times the speed of the conventional GCM it was trained on.

As a proof-of-concept this is a success: It is possible to build reasonable General Circulation Models using just conventional machine learning tools. Such models are *enormously* faster (to build and to run) than conventional physics-based GCMs; it is likely they will go on to become very widely used.

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Small Print <credits>

This document and the data associated with it are crown copyright (2019) and licensed under the terms of the `Open Government Licence <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/2/>`_. All code included is licensed under the terms of the `GNU Lesser General Public License <https://www.gnu.org/licenses/lgpl.html>`_.
