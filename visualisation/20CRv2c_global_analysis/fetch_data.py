#!/usr/bin/env python

import datetime
import IRData.twcr as twcr

for var in ('prmsl','uwnd.10m','vwnd.10m','air.2m'):
    twcr.fetch(var,datetime.datetime(2009,3,12,6),version='2c')

