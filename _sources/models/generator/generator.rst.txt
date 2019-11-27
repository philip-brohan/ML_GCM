Generating new weather states
=============================

A conventional, physics-based, GCM can make new weather states, and these new states are constrained to be plausible representations of real weather by forcing them to comply with a set of physical (and empirical) rules. The :doc:`autoencoder generator model <../autoencoder/index>` can also make new weather states, and these are also constrained to be plausible representations of real weather, by constraining the latent space, during training, to contain only states which decode to plausible weather states.

.. figure:: generator.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Using half the autoencoder as a generative model. (:doc:`Source code <../autoencoder/source>` - same as :doc:`the autoencoder <../autoencoder/index>`).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for the generative model (part of the autoencoder) <../autoencoder/source>

To test the generative model, construct a latent-space state from random numbers (100 independent draws from a normal distribution, mean=0, sd=1).

.. figure:: ../../../models/autoencoder/random_ls.png
   :width: 95%
   :align: center
   :figwidth: 95%

   Inset: the 100 random values in the latent space. Main figure: T2m, mslp, u and v winds generated from this latent space vector (:doc:`Source code <source>`).

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Source code for this generative model plot <source>


