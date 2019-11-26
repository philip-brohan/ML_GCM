Plot one frame from the 20CRv2c video
=====================================

The video shows temperature, wind, and mean-sea-level pressure. 

* The temperature is a colour map. It is quantile normalised - that is, the temperature field is scaled so that it's distribution is flat - this means that the image shows the same amount of each colour. 
* Mean-sea-level-pressure is shown as a contour plot.
* Wind is shown as advected speckles - a random field advected along with the wind vectors - this turns the speckles into stripes in the direction of the wind. They can be made to move from frame to frame by advecting most them a little more each timestep (and resetting a few of them to zero advection). This is plotted by adding the wind field to the temperature and precipitation fields before plotting them. The field has mean zero so it has no average effect, but it perturbs the other fields in a way which shows the wind structure.

Script to make a single frame:

.. literalinclude:: ../../../visualisation/20CRv2c_global_analysis/20CRv2c_4var.py

Library and utility functions used:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   Plotting utilities <../../lib/plots>
