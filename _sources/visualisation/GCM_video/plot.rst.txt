Plot one frame from the ML GCM video
====================================

The video shows temperature, wind, and mean-sea-level pressure, and the 100-dimensional latent space encoded representation, as simulated by the :doc:`GCM ML model <../../models/GCM/index>`. it's the same as the :doc:`original 20CRv2c video <../20CRv2c_latent_space/plot>` except that the fields are from the :doc:`GCM ML model <../../models/GCM/index>`.

.. literalinclude:: ../../../visualisation/GCM_video/plot_gcm.py

Library and utility functions used:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Weather data normalisation <../../lib/normalise>
   Plotting utilities <../../lib/plots>
