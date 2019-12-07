#!/usr/bin/env python

# Compare 4 original weather fields with 4 generated fields.

import tensorflow as tf
tf.enable_eager_execution()
import numpy

import IRData.twcr as twcr
import iris
import datetime
import argparse
import sys
import os
import math
import pickle

import Meteorographica as mg
from pandas import qcut

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys.path.append('%s/../../lib/' % os.path.dirname(__file__))
from insolation import load_insolation
from geometry import to_analysis_grid
from normalise import normalise_insolation
from normalise import normalise_t2m
from normalise import unnormalise_t2m
from normalise import normalise_prmsl
from normalise import unnormalise_prmsl
from normalise import normalise_wind
from normalise import unnormalise_wind
from plots import plot_cube
from plots import make_wind_seed
from plots import wind_field
from plots import quantile_normalise_t2m
from plots import draw_lat_lon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=False,default=25)
args = parser.parse_args()

# Define a dummy cube to load with the compressed data
def dummy_cube():
    cs=iris.coord_systems.RotatedGeogCS(90,180,0)
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
# Load the latent-space representation, and convert it back into normal space
model_save_file=("%s/ML_GCM/autoencoder/"+
                  "Epoch_%04d/generator") % (
                      os.getenv('SCRATCH'),args.epoch)
generator=tf.keras.models.load_model(model_save_file,compile=False)

# Random latent state
def random_state():
    ls=tf.convert_to_tensor(numpy.random.normal(size=100),numpy.float32)
    ls = tf.reshape(ls,[1,100])
    result=generator.predict_on_batch(ls)
    result = tf.reshape(result,[79,159,5])
    t2m=dummy_cube()
    t2m.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
    t2m.data = unnormalise_t2m(t2m.data)
    prmsl=dummy_cube()
    prmsl.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
    prmsl.data = unnormalise_prmsl(prmsl.data)
    u10m=dummy_cube()
    u10m.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
    u10m.data = unnormalise_wind(u10m.data)
    v10m=dummy_cube()
    v10m.data = tf.reshape(result.numpy()[:,:,3],[79,159]).numpy()
    v10m.data = unnormalise_wind(v10m.data)
    return(t2m,prmsl,u10m,v10m)

# Function to do the multivariate plot
lsmask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('DATADIR'))
# Random field for the wind noise
z=make_wind_seed(resolution=0.4)
def three_plot(ax,t2m,u10m,v10m,prmsl):
    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
    ax.set_aspect('auto')
    ax.set_axis_off() # Don't want surrounding x and y axis
    ax.add_patch(Rectangle((0,0),1,1,facecolor=(0.6,0.6,0.6,1),fill=True,zorder=1))
    # Draw lines of latitude and longitude
    draw_lat_lon(ax)
    # Add the continents
    mask_pc = plot_cube(0.05)   
    lsmask = iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('SCRATCH'))
    lsmask = lsmask.regrid(mask_pc,iris.analysis.Linear())
    lats = lsmask.coord('latitude').points
    lons = lsmask.coord('longitude').points
    mask_img = ax.pcolorfast(lons, lats, lsmask.data,
                             cmap=matplotlib.colors.ListedColormap(
                                    ((0.4,0.4,0.4,0),
                                     (0.4,0.4,0.4,1))),
                             vmin=0,
                             vmax=1,
                             alpha=1.0,
                             zorder=20)
    
    # Calculate the wind noise
    wind_pc=plot_cube(0.5)   
    cs=iris.coord_systems.RotatedGeogCS(90,180,0)
    rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
    u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
    v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
    wind_noise_field=wind_field(u10m,v10m,z,sequence=None,epsilon=0.01)

    # Plot the temperature
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

    # Plot the prmsl
    prmsl_pc=plot_cube(0.25)   
    prmsl = prmsl.regrid(prmsl_pc,iris.analysis.Linear())
    lats = prmsl.coord('latitude').points
    lons = prmsl.coord('longitude').points
    lons,lats = numpy.meshgrid(lons,lats)
    CS=ax.contour(lons, lats, prmsl.data*0.01,
                               colors='black',
                               linewidths=0.5,
                               alpha=1.0,
                               levels=numpy.arange(870,1050,10),
                               zorder=200)

def load_state(dt):
    prmsl=twcr.load('prmsl',dt,version='2c')
    prmsl=to_analysis_grid(prmsl.extract(iris.Constraint(member=1)))
    t2m=twcr.load('air.2m',dt,version='2c')
    t2m=to_analysis_grid(t2m.extract(iris.Constraint(member=1)))
    u10m=twcr.load('uwnd.10m',dt,version='2c')
    u10m=to_analysis_grid(u10m.extract(iris.Constraint(member=1)))
    v10m=twcr.load('vwnd.10m',dt,version='2c')
    v10m=to_analysis_grid(v10m.extract(iris.Constraint(member=1)))
    return(t2m,prmsl,u10m,v10m)

# Plot 4 fields of each, original and generated
fig=Figure(figsize=(10.8,10.8),
           dpi=300,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

def get_axis(n):
    offset=0.01
    w=(1-3*offset)/2
    h=(1-5*offset)/4
    xmin=offset
    if n%2==1: xmin=w+offset*2
    ymin=[offset,
          h+offset*2,
          2*h+offset*3,
          3*h+offset*4][(n-1)//2]
    ax=fig.add_axes([xmin,ymin,w,h])
    return(ax)

for n in (1,3,5,7):
    ax=get_axis(n)
    fields=random_state()
    three_plot(ax,fields[0],fields[2],fields[3],fields[1])
    
dts=[datetime.datetime(2009,1,2,6),
     datetime.datetime(2009,4,2,12),
     datetime.datetime(2009,7,23,18),
     datetime.datetime(2009,11,2,6)]
for n in (2,4,6,8):
    ax=get_axis(n)
    fields=load_state(dts[n//2-1])
    three_plot(ax,fields[0],fields[2],fields[3],fields[1])

# Render the figure as a png
fig.savefig("group_compare.png")

