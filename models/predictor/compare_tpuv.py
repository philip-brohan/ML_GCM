#!/usr/bin/env python

# Compare an original weather field with the predictor output.

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
                    type=int,required=False,default=10)
args = parser.parse_args()

dte=datetime.datetime(2010,3,12,18)

# Function to do the multivariate plot
lsmask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % 
                                                    os.getenv('SCRATCH'))
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
   
# Load the source data
prmsl=twcr.load('prmsl',dte,version='2c')
prmsl=to_analysis_grid(prmsl.extract(iris.Constraint(member=1)))
t2m=twcr.load('air.2m',dte,version='2c')
t2m=to_analysis_grid(t2m.extract(iris.Constraint(member=1)))
u10m=twcr.load('uwnd.10m',dte,version='2c')
u10m=to_analysis_grid(u10m.extract(iris.Constraint(member=1)))
v10m=twcr.load('vwnd.10m',dte,version='2c')
v10m=to_analysis_grid(v10m.extract(iris.Constraint(member=1)))
insol=to_analysis_grid(load_insolation(dte.year,dte.month,dte.day,dte.hour))

# Convert the source data into tensor format
t2m_t = tf.convert_to_tensor(normalise_t2m(t2m.data),numpy.float32)
t2m_t = tf.reshape(t2m_t,[79,159,1])
prmsl_t = tf.convert_to_tensor(normalise_prmsl(prmsl.data),numpy.float32)
prmsl_t = tf.reshape(prmsl_t,[79,159,1])
u10m_t = tf.convert_to_tensor(normalise_wind(u10m.data),numpy.float32)
u10m_t = tf.reshape(u10m_t,[79,159,1])
v10m_t = tf.convert_to_tensor(normalise_wind(v10m.data),numpy.float32)
v10m_t = tf.reshape(v10m_t,[79,159,1])
insol_t = tf.convert_to_tensor(normalise_insolation(insol.data),numpy.float32)
insol_t = tf.reshape(insol_t,[79,159,1])

# Get predicted versions of the target data
model_save_file=("%s/ML_GCM/predictor/"+
                  "Epoch_%04d/predictor") % (
                      os.getenv('SCRATCH'),args.epoch)
autoencoder=tf.keras.models.load_model(model_save_file,compile=False)
ict = tf.concat([t2m_t,prmsl_t,u10m_t,v10m_t,insol_t],2) # Now [79,159,5]
ict = tf.reshape(ict,[1,79,159,5])
result = autoencoder.predict_on_batch(ict)
result = tf.reshape(result,[79,159,5])

# Convert the predicted fields back to unnormalised cubes 
t2m_r=t2m.copy()
t2m_r.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
t2m_r.data = unnormalise_t2m(t2m_r.data)
prmsl_r=prmsl.copy()
prmsl_r.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
prmsl_r.data = unnormalise_prmsl(prmsl_r.data)
u10m_r=u10m.copy()
u10m_r.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
u10m_r.data = unnormalise_wind(u10m_r.data)
v10m_r=v10m.copy()
v10m_r.data = tf.reshape(result.numpy()[:,:,3],[79,159]).numpy()
v10m_r.data = unnormalise_wind(v10m_r.data)

# Load the actual data for mthe target time
dte2=dte+datetime.timedelta(hours=6)
prmsl=twcr.load('prmsl',dte2,version='2c')
prmsl=to_analysis_grid(prmsl.extract(iris.Constraint(member=1)))
t2m=twcr.load('air.2m',dte2,version='2c')
t2m=to_analysis_grid(t2m.extract(iris.Constraint(member=1)))
u10m=twcr.load('uwnd.10m',dte2,version='2c')
u10m=to_analysis_grid(u10m.extract(iris.Constraint(member=1)))
v10m=twcr.load('vwnd.10m',dte2,version='2c')
v10m=to_analysis_grid(v10m.extract(iris.Constraint(member=1)))

# Plot the two fields and a scatterplot for each variable
fig=Figure(figsize=(9.6*1.2,10.8),
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Two maps, original and reconstructed
ax_original=fig.add_axes([0.005,0.525,0.75,0.45])
three_plot(ax_original,t2m,u10m,v10m,prmsl)
ax_reconstructed=fig.add_axes([0.005,0.025,0.75,0.45])
three_plot(ax_reconstructed,t2m_r,u10m_r,v10m_r,prmsl_r)

# Scatterplot of encoded v original
def plot_scatter(ax,ic,pm):
    dmin=min(ic.min(),pm.min())
    dmax=max(ic.max(),pm.max())
    dmean=(dmin+dmax)/2
    dmax=dmean+(dmax-dmean)*1.02
    dmin=dmean-(dmean-dmin)*1.02
    ax.set_xlim(dmin,dmax)
    ax.set_ylim(dmin,dmax)
    ax.scatter(x=pm.flatten(),
               y=ic.flatten(),
               c='black',
               alpha=0.25,
               marker='.',
               s=2)
    ax.set(ylabel='Original', 
           xlabel='Encoded')
    ax.grid(color='black',
            alpha=0.2,
            linestyle='-', 
            linewidth=0.5)
    
ax_t2m=fig.add_axes([0.83,0.80,0.16,0.17])
plot_scatter(ax_t2m,t2m.data,t2m_r.data)
ax_prmsl=fig.add_axes([0.83,0.55,0.16,0.17])
plot_scatter(ax_prmsl,prmsl.data*0.01,prmsl_r.data*0.01)
ax_u10m=fig.add_axes([0.83,0.30,0.16,0.17])
plot_scatter(ax_u10m,u10m.data,u10m_r.data)
ax_v10m=fig.add_axes([0.83,0.05,0.16,0.17])
plot_scatter(ax_v10m,v10m.data,v10m_r.data)

# Render the figure as a png
fig.savefig("compare_tpuv.png")

