Small data: 20CRv2c after compression into a 100-dimensional latent space
=========================================================================

.. raw:: html

    <center>
    <table><tr><td><center>
    <iframe src="https://player.vimeo.com/video/369615958?title=0&byline=0&portrait=0" width="795" height="448" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe></center></td></tr>
    <tr><td><center>Small data: Near-surface temperature, wind, and MSLP, from the Twentieth Century Reanalyis, after compression into a 100-dimensional latent space</center></td></tr>
    </table>
    </center>

20CRv2c is on a 2 degree grid, so this set of four surface variables has a state vector of size 180*90*4=64,800: we need that many data points, every 6-hours to make the :doc:`analysis video <../20CRv2c_global_analysis/index>`. The :doc:`autoencoder <../../models/autoencoder/index>` compresses that 64,800-dimensional state vector into a 100-dimensional latent space, and then expands it out again. This video shows the reanalysis after this compression. So it's the same as the :doc:`original 20CRv2c video <../20CRv2c_global_analysis/index>`, except that it uses only 0.15% as much data to represent the weather state.
As well as the weather state, the video shows the associated latent-space vector (the compressed data form) as the grid of 100 numbers at the bottom left.
  
.. toctree::
   :titlesonly:
   :maxdepth: 1

   Get the data <../20CRv2c_global_analysis/data>
   Train the autoencoder <../../models/autoencoder/index>
   Plot a single frame <plot>
   Plot all the frames and make a video <video>

