Plotting utilities library
==========================

This library file contains five functions:

quantile_normalise_t2m:
   Converts a cube of T2m in kelvin to a cube of approximately on the range 0-1. If the cube is on the conventional equiangular projectionj (pole at 90N), then the cube will have an approximately flat distribution (each value occurs the same number of times. This is convenient for colourmap plotting.

plot_cube:
   Generate an iris cube with the given resolution, range and pole location (data are all zero). Most useful as a common arrangement to regrid other cubes onto.

wind_field:
   Take a field of white noise, and advect it along with a wind field. This turns the speckles into stripes in the direction of the wind. They can be made to move from frame to frame by advecting most them a little more each timestep (and resetting a few of them to zero advection)

draw_lat_lon:
   Draw lines of latitude and longitude (copes with rotated poles).

get_precip_colours:
   Make a colour map for plotting precip. Uses the 'algae' colour scale from `cmocean <https://matplotlib.org/cmocean/>`_, adjusted to taper to transparency at the bottom end

.. literalinclude:: ../../lib/plots.py
