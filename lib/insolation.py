# Functions for handling insolation data

# The weather data can mostly be handled by IRData.
# But the insolation data needs special handling.

import iris
import os

# load the insolation data
def load_insolation(year, month, day, hour):
    if month == 2 and day == 29:
        day = 28
    time_constraint = iris.Constraint(
        time=iris.time.PartialDateTime(year=1969, month=month, day=day, hour=hour)
    )
    ic = iris.load_cube(
        "%s/20CR/version_2c/ensmean/cduvb.1969.nc" % os.getenv("SCRATCH"),
        iris.Constraint(name="3-hourly Clear Sky UV-B Downward Solar Flux")
        & time_constraint,
    )
    coord_s = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)
    ic.coord("latitude").coord_system = coord_s
    ic.coord("longitude").coord_system = coord_s
    return ic
