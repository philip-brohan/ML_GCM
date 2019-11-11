Assemble data for the 20CRv2c video
===================================

The `Twentieth Century Reanalysis <https://www.esrl.noaa.gov/psd/data/20thC_Rean/>`_ version 2c provides data on a 2x2 degree grid back to 1851. The data are online and can be downloaded by the `IRData library <http://brohan.org/IRData/>`_

Script to collect the variables shown for one calendar year:

.. literalinclude:: ../../../visualisation/20CRv2c_global_analysis/fetch_data.py

As well as the weather data, the plot script needs a land-mask file to plot the continents. This script retrieves that.

.. literalinclude:: ../../../data/retrieve_incidentals.py

