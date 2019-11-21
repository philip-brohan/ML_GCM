Generating new weather states
=============================

.. figure:: generator.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Using half the autoencoder as a generative model

.. figure:: ../../../models/autoencoder/random_ls.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Inset: the 100 random values in the latent space. Main figure: T2m, mslp, u and v winds generated from this latent space vector.

.. literalinclude:: ../../../models/autoencoder/plot_random_latent_space.py


