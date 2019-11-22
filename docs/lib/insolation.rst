Insolation library
==================

This library file contains one function:

load_insolation:
   We are using surface clear-sky downwelling UVb as an easily-available proxy for top-of-atmosphere shortwave. We don't care bout how it varies with the weather, only the diurnal and seasonal cycles, so only one year of data is needed (1969 is downloaded). This function loads the equuivalent 1969 data for the specified time-point.

.. literalinclude:: ../../lib/insolation.py
