Data for the Machine Learning GCM
=================================

Training data is taken from `20CRv2c <https://www.esrl.noaa.gov/psd/data/gridded/data.20thC_ReanV2c.html>`_, which provides full atmospheric states, every 6 hours, between 1851 and 2012. 20CR is an ensemble system, and v2c has 56 ensemble members; we can't use the ensemble mean, as it does not have consistent properties - the amplitude of the weather variability depends on how many observations are abvailable. But 20CRv2c provides `individual ensemble members <https://portal.nersc.gov/project/20C_Reanalysis/>`_ for some variables, and we can use those.

The data we are using comprises the 2m temperatures, 10m winds (u and v), and mean-sea-level pressure (MSLP). The period used is 1969 to 2005 for training, and 2006 to 2010 for validation. The period choice is arbitrary, 40 years of data is enough to demonstrate a lot of weather variability, but only runs to about 700Gb on disc, so we can experiment with it using minimal computing resources.

As well as the weather data, we want some forcing data (training data for modeling annual and diurnal cycles). I'd like the top-of-atmosphere downwelling shortwave - 20CRv2c does not have this, but it does have the surface `Clear Sky UV-B Downward Solar Flux <https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBSearch.pl?Dataset=NOAA-CIRES+20th+Century+Reanalysis+Version+2c&Variable=Clear+Sky+UV-B+Downward+Solar+Flux>`_ - only the ensemble means are readily available, but for this variable it doesn't matter (it's not very weather dependent), and for the same reason we only need one year of data. 

We access the data using the `IRData package <http://brohan.org/IRData/>`_:

.. literalinclude:: ../../data/retrieve_20CR_data.py

