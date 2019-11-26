#!/usr/bin/env python

# Atmospheric state - near-surface temperature, wind, and precip.

import os
import sys
import IRData.opfc as opfc
import datetime
import pickle

import iris
import numpy

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from pandas import qcut

from load_step import load_recent_temperatures
from load_step import load_li_precip
from load_step import load_di_icec

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
                    default="%s/images/opfc_global_3var_meanp" % \
                                           os.getenv('SCRATCH'),
                    type=str,required=False)

args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

dte=datetime.datetime(args.year,args.month,args.day,
                      int(args.hour),int(args.hour%1*60))

# In the  temperature field, damp the diurnal cycle, and
#  boost the short-timescale variability. Load the 
#  recent data to calculate this.
(tavg,davg) = load_recent_temperatures(dte)

# Load the model data 
t2m=opfc.load('air.2m',dte,model='global')
# Remove the diurnal cycle
t2m.data -= davg.data
# Double the synoptic variability
t2m.data += (t2m.data-tavg.data)*1
# Add back a reduced diurnal cycle
t2m.data += davg.data*0.25

u10m=opfc.load('uwnd.10m',dte,model='global')
v10m=opfc.load('vwnd.10m',dte,model='global')

# We're plotting log precip - so interpolate in log space too
precip=load_li_precip(dte)

mask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % 
                                                  os.getenv('SCRATCH'))
# Icec is only daily, so interpolate manually
icec=load_di_icec(dte)

# Remap the t2m to highlight small differences
t2m=quantile_normalise_t2m(t2m)
t2m *= 1000

# Define the figure (page size, background color, resolution, ...
fig=Figure(figsize=(38.4,21.6),              # Width, Height (inches)
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

# Make the wind noise
wind_pc=plot_cube(0.2,-180/args.zoom,180/args.zoom,
                      -90/args.zoom,90/args.zoom)   
cs=iris.coord_systems.RotatedGeogCS(90.0,180.0,0.0)
rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
seq=(dte-datetime.datetime(2000,1,1)).total_seconds()/3600
z=make_wind_seed(0.4,seed=1)
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

# Plot the land mask
mask_pc=plot_cube(0.05,-180/args.zoom,180/args.zoom,
                                  -90/args.zoom,90/args.zoom)   
mask = mask.regrid(mask_pc,iris.analysis.Linear())
lats = mask.coord('latitude').points
lons = mask.coord('longitude').points
mask_img = ax.pcolorfast(lons, lats, mask.data,
                         cmap=matplotlib.colors.ListedColormap(
                                ((0.4,0.4,0.4,0),
                                 (0.4,0.4,0.4,1))),
                         vmin=0,
                         vmax=1,
                         alpha=1.0,
                         zorder=20)

# Plot the sea-ice
ice_pc=plot_cube(0.05,-180/args.zoom,180/args.zoom,
                      -90/args.zoom,90/args.zoom)   
icec = icec.regrid(ice_pc,iris.analysis.Linear())
icec_img = ax.pcolorfast(lons, lats, icec.data,
                         cmap=matplotlib.colors.ListedColormap(
                                ((0.5,0.5,0.5,0),
                                 (0.5,0.5,0.5,1))),
                         vmin=0,
                         vmax=1,
                         alpha=1.0,
                         zorder=10)

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
t2m_img = ax.pcolorfast(lons, lats, t2m.data+wnf.data,
                        cmap='RdYlBu_r',
                        vmin=-100,
                        vmax=1100,
                        alpha=0.8,
                        zorder=100)

# Plot the precip
precip_pc=plot_cube(0.25,-180/args.zoom,180/args.zoom,
                         -90/args.zoom,90/args.zoom)   
precip = precip.regrid(precip_pc,iris.analysis.Linear())
wnf=wind_noise_field.regrid(precip,iris.analysis.Linear())
precip.data += wnf.data/1000
cols=get_precip_colours()
precip_img = ax.pcolorfast(lons, lats, precip.data,
                           cmap=matplotlib.colors.ListedColormap(cols),
                           vmin=0,
                           vmax=1,
                           alpha=0.8,
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
         size=28,
         clip_on=True,
         zorder=500)

# Render the figure as a png
fig.savefig('%s/%04d%02d%02d%02d%02d.png' % (args.opdir,args.year,
                                             args.month,args.day,
                                             int(args.hour),
                                             int(args.hour%1*60)))
