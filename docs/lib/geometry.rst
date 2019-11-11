Regridding library
==================

This library file contains one function:

to_analysis_grid:
   THe ML GCM operates on data with a single standard grid: pole at 90N, 180E, 79 latitude values and 158 longitude values. This has the UK nicely centred and a convenient grid size for strided convolutions. This function converts any iris cube onto this grid.

.. literalinclude:: ../../lib/geometry.py
