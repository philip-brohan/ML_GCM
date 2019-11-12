#!/usr/bin/env python

# Atmospheric state - near-surface temperature, u-wind, v-wind, and prmsl.
# Show the field associated with a random latent space state.

import os
import sys
import IRData.twcr as twcr

import tensorflow as tf
tf.enable_eager_execution()

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
from normalise import unnormalise_t2m
from normalise import unnormalise_prmsl
from normalise import unnormalise_wind

from plots import plot_cube
from plots import wind_field
from plots import quantile_normalise_t2m
from plots import draw_lat_lon

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=False,default=10)

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

wind_pc=plot_cube(0.2,-180,180,-90,90)   
cs=iris.coord_systems.RotatedGeogCS(90,180,0)
rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
z=mask.regrid(u10m,iris.analysis.Linear())
(width,height)=z.data.shape
z.data=numpy.random.rand(width,height)
wind_noise_field=wind_field(u10m,v10m,z,sequence=None,epsilon=0.01)

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
mask_pc=plot_cube(0.05,-180,180,-90,90)   
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
t2m_pc=plot_cube(0.05,-180,180,
                      -90,90)   
t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
t2m=quantile_normalise_t2m(t2m)
# Adjust to show the wind
wscale=200
s=wind_noise_field.data.shape
wind_noise_field.data=qcut(wind_noise_field.data.flatten(),wscale,labels=False,
                             duplicates='drop').reshape(s)-(wscale-1)/2

# Plot as a colour map
wnf=wind_noise_field.regrid(t2m,iris.analysis.Linear())
t2m_img = ax.pcolorfast(lons, lats, t2m.data*800+wnf.data,
                        cmap='RdYlBu_r',
                        alpha=0.8,
                        zorder=100)

# PRMSL contours
prmsl_pc=plot_cube(0.25,-180,180,
                         -90,90)   
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

# Overlay the latent-space representation in the SE Pacific
ax2=fig.add_axes([0.025,0.05,0.15,0.15*16/9])
ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
ax2.set_axis_off() # Don't want surrounding x and y axis
x=numpy.linspace(0,10,10)
latent_img = ax2.pcolorfast(x,x,ls.numpy().reshape(10,10),
                           cmap='viridis',
                             alpha=1.0,
                             vmin=-3,
                             vmax=3,
                             zorder=1000)

# Render the figure as a png
fig.savefig('random_ls.png')
