Plot one frame from the 20CRv2c autoencoded video
=================================================

The video shows temperature, wind, and mean-sea-level pressure, it's the same as the :doc:`original 20CRv2c video <../20CRv2c_global_analysis/plot>` except that the fields are passed through :doc:`the autoencoder <../../models/autoencoder/index>` before plotting, and the 100-dimensional latent space encoded reprsentation is shown as an overlay in the bottom left. 

.. literalinclude:: ../../../visualisation/20CRv2c_compressed/plot_latent_space.py

Library and utility functions used:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Function to load insolation <../../lib/insolation>
   Specification for the data analysis grid <../../lib/geometry>
   Weather data normalisation <../../lib/normalise>
   Plotting utilities <../../lib/plots>
