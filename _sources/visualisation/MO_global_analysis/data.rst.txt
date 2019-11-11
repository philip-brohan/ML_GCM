Assemble data for the Met Office Global Analysis video
======================================================

*Note*: To get this data you will need access to the Met Office operational MASS archive, and such access is not generally available outside the Met Office. The same data are distributed through the `Met Office Data Portal <https://www.metoffice.gov.uk/services/data/met-office-data-for-reuse/discovery>`_ *but* only a few days data is online at any one time.

The analysis is run once every 6 hours, but we want data at a higher time resolution than that. So we collect hourly data (the highest time resolution output) - each analysis, and the first 5 hours of forecast. 

Script to collect the variables shown for all the analyses run in one calendar day:

.. literalinclude:: ../../../visualisation/MO_global_analysis/get_data_for_day.py

This script must be run repeatedly to gather data for all the days shown in the video. 

.. literalinclude:: ../../../visualisation/MO_global_analysis/fetch_data_daily.py

This produces a list of data retrieval commands - one for each day. Running these in parallel would overload the MASS system, so just run them in series - it'll take a few days to get all the data from tape.

As well as the weather data, the plot script needs a land-mask file to plot the continents. This script retrieves that.

.. literalinclude:: ../../../data/retrieve_incidentals.py

