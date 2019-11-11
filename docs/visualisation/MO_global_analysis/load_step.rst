Load data for the MO global analysis video
==========================================

This file contains utility functions for loading data, from one point in time, from the Met Office global analysis. It contains three functions:

load_recent_temperatures:
   Loads all the temperature fields from a period aroind the point specified (default +-5 days) and returns the mean over the period and the mean diurnal cycle component, at the given point in time, over the period (so if the given time is 7am it will return the mean difference betqween time at 7am and daily average).

load_li_precip:
   Same as the load function from `IRData <http://brohan.org/IRData/>`_ except that it takes the logarithm of precipitation before interpolating in time. (When plotting log fields, you get a more consistent result from interpolating the logs than from taking the log of the interpolated value).

load_di_icec(dte):
   Same as the load function from `IRData <http://brohan.org/IRData/>`_ except that it only loads data from file at 0Z, and otherwise interpolates. The opfc data output has 6-hourly sea-ice, but all the values within a day are the same.

.. literalinclude:: ../../../visualisation/MO_global_analysis/load_step.py
