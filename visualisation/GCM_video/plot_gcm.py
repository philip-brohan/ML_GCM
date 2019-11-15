#!/usr/bin/env python

# Atmospheric state - near-surface temperature, u-wind, v-wind, and prmsl.
# Show the version from the direct-forecast GCM

import os
import sys
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

sys.path.append('%s/../../lib/' % os.path.dirname(__file__))
from normalise import unnormalise_t2m
from normalise import unnormalise_prmsl
from normalise import unnormalise_wind
from plots import plot_cube
from plots import make_wind_seed
from plots import wind_field
from plots import quantile_normalise_t2m
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
parser.add_argument("--opdir", help="Directory for output files",
                    default="%s/images/ML_df_GCM_4var" % \
                                           os.getenv('SCRATCH'),
                    type=str,required=False)

args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Projection for tensors and plotting
cs=iris.coord_systems.RotatedGeogCS(90,180,0)

# Define a dummy cube to load with the compressed data
def dummy_cube():
    # Latitudes cover -90 to 90 with 79 values
    lat_values=numpy.arange(-90,91,180/78)
    latitude = iris.coords.DimCoord(lat_values,
                                    standard_name='latitude',
                                    units='degrees_north',
                                    coord_system=cs)
    # Longitudes cover -180 to 180 with 159 values
    lon_values=numpy.arange(-180,181,360/158)
    longitude = iris.coords.DimCoord(lon_values,
                                     standard_name='longitude',
                                     units='degrees_east',
                                     coord_system=cs)
    dummy_data = numpy.zeros((len(lat_values), len(lon_values)))
    dummy_cube = iris.cube.Cube(dummy_data,
                               dim_coords_and_dims=[(latitude, 0),
                                                    (longitude, 1)])
    return(dummy_cube)

dte=datetime.datetime(args.year,args.month,args.day,
                      int(args.hour),int(args.hour%1*60))

# Load the GCM representation - interpolating in time as necesary
def ls_load_at_timepoint(year,month,day,hour):
    pfile=("%s/ML_GCM/GCM_mucdf/"+
           "%04d-%02d-%02d:%02d.pkl") % (os.getenv('SCRATCH'),
            year,month,day,hour)
    res=pickle.load(open(pfile,'rb'))
    ls = res['latent_s']
    t2m=dummy_cube()
    t2m.data = res['state_v'][0,:,:,0]
    t2m.data = unnormalise_t2m(t2m.data)
    prmsl=dummy_cube()
    prmsl.data = res['state_v'][0,:,:,1]
    prmsl.data = unnormalise_prmsl(prmsl.data)
    u10m=dummy_cube()
    u10m.data = res['state_v'][0,:,:,2]
    u10m.data = unnormalise_wind(u10m.data)
    v10m=dummy_cube()
    v10m.data = res['state_v'][0,:,:,3]
    v10m.data = unnormalise_wind(v10m.data)
    return(ls,t2m,prmsl,u10m,v10m)
    
dte_past=datetime.datetime(dte.year,dte.month,dte.day,dte.hour-dte.hour%6)
dte_next=dte_past+datetime.timedelta(hours=6)
weight=(dte-dte_past).total_seconds()/(dte_next-dte_past).total_seconds()
dpast=ls_load_at_timepoint(dte_past.year,dte_past.month,dte_past.day,dte_past.hour)
dnext=ls_load_at_timepoint(dte_next.year,dte_next.month,dte_next.day,dte_next.hour)
ls=dnext[0]*weight+dpast[0]*(1-weight)
t2m=dnext[1]
t2m.data=dnext[1].data*weight+dpast[1].data*(1-weight)
prmsl=dnext[2]
prmsl.data=dnext[2].data*weight+dpast[2].data*(1-weight)
u10m=dnext[3]
u10m.data=dnext[3].data*weight+dpast[3].data*(1-weight)
v10m=dnext[4]
v10m.data=dnext[4].data*weight+dpast[4].data*(1-weight)

mask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('SCRATCH'))

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


wind_pc=plot_cube(0.2)   
rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
seq=(dte-datetime.datetime(1969,1,1)).total_seconds()/3600
z=make_wind_seed(resolution=0.4,seed=0)
wind_noise_field=wind_field(u10m,v10m,z,sequence=int(seq*5),epsilon=0.01)

# Define an axes to contain the plot. In this case our axes covers
#  the whole figure
ax = fig.add_axes([0,0,1,1])
ax.set_axis_off() # Don't want surrounding x and y axis

# Lat and lon range (in rotated-pole coordinates) for plot
ax.set_xlim(-180,180)
ax.set_ylim(-90,90)
ax.set_aspect('auto')

# Background
ax.add_patch(Rectangle((0,0),1,1,facecolor=(0.6,0.6,0.6,1),fill=True,zorder=1))

draw_lat_lon(ax)

# Plot the land mask
mask_pc=plot_cube(0.05)   
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


# Plot the T2M
t2m_pc=plot_cube(0.05)   
t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
t2m=quantile_normalise_t2m(t2m)
# Adjust to show the wind
wscale=200
s=wind_noise_field.data.shape
wind_noise_field.data=qcut(wind_noise_field.data.flatten(),wscale,labels=False,
                             duplicates='drop').reshape(s)-(wscale-1)/2

# Plot as a colour map
wnf=wind_noise_field.regrid(t2m,iris.analysis.Linear())
t2m_img = ax.pcolorfast(lons, lats, t2m.data*1000+wnf.data,
                        cmap='RdYlBu_r',
                        alpha=0.8,
                        vmin=-100,
                        vmax=1100,
                        zorder=100)

# PRMSL contours
prmsl_pc=plot_cube(0.25)   
prmsl = prmsl.regrid(prmsl_pc,iris.analysis.Linear())
lats = prmsl.coord('latitude').points
lons = prmsl.coord('longitude').points
lons,lats = numpy.meshgrid(lons,lats)
CS=ax.contour(lons, lats, prmsl.data*0.01,
                           colors='black',
                           linewidths=1.5,
                           linestyles='solid',
                           alpha=1.0,
                           levels=numpy.arange(870,1000,10),
                           zorder=200)
CS=ax.contour(lons, lats, prmsl.data*0.01,
                           colors='black',
                           linewidths=1.5,
                           linestyles='solid',
                           alpha=1.0,
                           levels=numpy.arange(1010,1080,10),
                           zorder=200)

# Label with the date
ax.text(180-(360)*0.009,
         90-(180)*0.016,
         "%04d-%02d-%02d" % (args.year,args.month,args.day),
         horizontalalignment='right',
         verticalalignment='top',
         color='black',
         bbox=dict(facecolor=(0.6,0.6,0.6,0.8),
                   edgecolor='black',
                   boxstyle='round',
                   pad=0.5),
         size=14,
         clip_on=True,
         zorder=500)

# Overlay the latent-space representation in the SE Pacific
ax2=fig.add_axes([0.025,0.05,0.15,0.15*16/9])
ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
ax2.set_axis_off() # Don't want surrounding x and y axis
x=numpy.linspace(0,10,10)
latent_img = ax2.pcolorfast(x,x,numpy.reshape(ls,(10,10)),
                           cmap='viridis',
                             alpha=1.0,
                             vmin=-3,
                             vmax=3,
                             zorder=1000)

# Render the figure as a png
fig.savefig('%s/%04d%02d%02d%02d%02d.png' % (args.opdir,args.year,
                                             args.month,args.day,
                                             int(args.hour),
                                             int(args.hour%1*60)))
