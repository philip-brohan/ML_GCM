#!/usr/bin/env python

# Atmospheric state - near-surface temperature, u-wind, v-wind, and prmsl.

import sys
import os
import IRData.opfc as opfc
import IRData.twcr as twcr
import datetime
import pickle

import iris
import numpy
import math

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from pandas import qcut

sys.path.append('%s/../../lib/' % os.path.dirname(__file__))
from plots import quantile_normalise_t2m
from plots import plot_cube
from plots import make_wind_seed
from plots import wind_field
from plots import get_precip_colours
from plots import draw_lat_lon


# Fix dask SPICE bug
import dask
dask.config.set(scheduler='single-threaded')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year",
                    type=int,required=True)
parser.add_argument("--month", help="Integer month",
                    type=int,required=True)

parser.add_argument("--day", help="Day of month",
                    type=int,required=True)
parser.add_argument("--hour", help="Time of day (0 to 23.99)",
                    type=float,required=True)
parser.add_argument("--pole_latitude", help="Latitude of projection pole",
                    default=90,type=float,required=False)
parser.add_argument("--pole_longitude", help="Longitude of projection pole",
                    default=180,type=float,required=False)
parser.add_argument("--npg_longitude", help="Longitude of view centre",
                    default=0,type=float,required=False)
parser.add_argument("--zoom", help="Scale factor for viewport (1=global)",
                    default=1,type=float,required=False)
parser.add_argument("--opdir", help="Directory for output files",
                    default="%s/images/20CRv2c_global_4var" % \
                                           os.getenv('SCRATCH'),
                    type=str,required=False)

args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)


dte=datetime.datetime(args.year,args.month,args.day,
                      int(args.hour),int(args.hour%1*60))

# Load the model data - dealing sensibly with missing fields
t2m=twcr.load('air.2m',dte,version='2c')
t2m=t2m.extract(iris.Constraint(member=1))
t2m=quantile_normalise_t2m(t2m)

u10m=twcr.load('uwnd.10m',dte,version='2c')
u10m=u10m.extract(iris.Constraint(member=1))
v10m=twcr.load('vwnd.10m',dte,version='2c')
v10m=v10m.extract(iris.Constraint(member=1))
prmsl=twcr.load('prmsl',dte,version='2c')
prmsl=prmsl.extract(iris.Constraint(member=1))

mask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % 
                                                   os.getenv('SCRATCH'))

# Define the figure (page size, background color, resolution, ...
fig=Figure(figsize=(19.2,10.8),              # Width, Height (inches)
           dpi=100,
           facecolor=(0.5,0.5,0.5,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,                # Don't draw a frame
           subplotpars=None,
           tight_layout=None)
fig.set_frameon(False) 
# Attach a canvas
canvas=FigureCanvas(fig)

# Projection for plotting
cs=iris.coord_systems.RotatedGeogCS(args.pole_latitude,
                                    args.pole_longitude,
                                    args.npg_longitude)

wind_pc=plot_cube(0.2,-180/args.zoom,180/args.zoom,
                      -90/args.zoom,90/args.zoom)   
rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
seq=(dte-datetime.datetime(2000,1,1)).total_seconds()/3600
z=make_wind_seed(resolution=0.4,seed=0)
wind_noise_field=wind_field(u10m,v10m,z,sequence=int(seq*5),epsilon=0.01)

# Define an axes to contain the plot. In this case our axes covers
#  the whole figure
ax = fig.add_axes([0,0,1,1])
ax.set_axis_off() # Don't want surrounding x and y axis

# Lat and lon range (in rotated-pole coordinates) for plot
ax.set_xlim(-180/args.zoom,180/args.zoom)
ax.set_ylim(-90/args.zoom,90/args.zoom)
ax.set_aspect('auto')

# Background
ax.add_patch(Rectangle((0,0),1,1,facecolor=(0.6,0.6,0.6,1),fill=True,zorder=1))

# Draw lines of latitude and longitude
draw_lat_lon(ax,lwd=0.75,
                pole_longitude=args.pole_longitude,
                pole_latitude=args.pole_latitude,
                npg_longitude=args.npg_longitude)

# Plot the T2M
t2m_pc=plot_cube(0.05,-180/args.zoom,180/args.zoom,
                      -90/args.zoom,90/args.zoom)   
t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
# Adjust to show the wind
wscale=200
s=wind_noise_field.data.shape
wind_noise_field.data=qcut(wind_noise_field.data.flatten(),wscale,labels=False,
                             duplicates='drop').reshape(s)-(wscale-1)/2

# Plot as a colour map
wnf=wind_noise_field.regrid(t2m,iris.analysis.Linear())
lats = t2m.coord('latitude').points
lons = t2m.coord('longitude').points
t2m_img = ax.pcolorfast(lons, lats, t2m.data*1000+wnf.data,
                        cmap='RdYlBu_r',
                        alpha=0.8,
                        zorder=100)

# PRMSL contours
prmsl_pc=plot_cube(0.25,-180/args.zoom,180/args.zoom,
                         -90/args.zoom,90/args.zoom)   
prmsl = prmsl.regrid(prmsl_pc,iris.analysis.Linear())
lats = prmsl.coord('latitude').points
lons = prmsl.coord('longitude').points
lons,lats = numpy.meshgrid(lons,lats)
CS=ax.contour(lons, lats, prmsl.data*0.01,
                           colors='black',
                           linewidths=1.0,
                           alpha=1.0,
                           levels=numpy.arange(870,1050,10),
                           zorder=200)

# Label with the date
ax.text(180/args.zoom-(360/args.zoom)*0.009,
         90/args.zoom-(180/args.zoom)*0.016,
         "%04d-%02d-%02d" % (args.year,args.month,args.day),
         horizontalalignment='right',
         verticalalignment='top',
         color='black',
         bbox=dict(facecolor=(0.6,0.6,0.6,0.5),
                   edgecolor='black',
                   boxstyle='round',
                   pad=0.5),
         size=14,
         clip_on=True,
         zorder=500)

# Render the figure as a png
fig.savefig('%s/%04d%02d%02d%02d%02d.png' % (args.opdir,args.year,
                                             args.month,args.day,
                                             int(args.hour),
                                             int(args.hour%1*60)))
