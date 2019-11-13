#!/usr/bin/env python

# Atmospheric state - near-surface temperature, u-wind, v-wind, and prmsl.
# Show the version after compression into a latant space.

import os
import sys
import IRData.twcr as twcr
import datetime
import pickle

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
from plots import wind_field
from plots import quantile_normalise_t2m
from plots import draw_lat_lon

from pandas import qcut

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
parser.add_argument("--epoch", help="Epoch",
                    type=int,required=True)
parser.add_argument("--opdir", help="Directory for output files",
                    default="%s/images/20CRv2c_ls_4var" % \
                                           os.getenv('SCRATCH'),
                    type=str,required=False)
parser.add_argument("--zfile", help="Noise pickle file name",
                    default="%s/images/20CRv2c_ls_4var/z.pkl" % \
                                           os.getenv('SCRATCH'),
                    type=str,required=False)

args = parser.parse_args()
if not os.path.isdir(args.opdir):
    os.makedirs(args.opdir)

# Function to do the multivariate plot
lsmask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('DATADIR'))
# Random field for the wind noise
z=pickle.load(open( args.zfile, "rb" ) )
def three_plot(ax,t2m,u10m,v10m,prmsl,ls):
    ax.set_xlim(-180,180)
    ax.set_ylim(-90,90)
    ax.set_aspect('auto')
    ax.set_axis_off() # Don't want surrounding x and y axis
    ax.add_patch(Rectangle((0,0),1,1,facecolor=(0.6,0.6,0.6,1),
                                               fill=True,zorder=1))
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
    wind_pc=plot_cube(0.2)   
    cs=iris.coord_systems.RotatedGeogCS(90,180,0)
    rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
    u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
    v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
    seq=(dte-datetime.datetime(2000,1,1)).total_seconds()/3600
    wind_noise_field=wind_field(u10m,v10m,z,sequence=int(seq*5),epsilon=0.01)

    # Plot the temperature
    t2m=quantile_normalise_t2m(t2m)
    t2m_pc=plot_cube(0.05)   
    t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
    # Adjust to show the wind
    wscale=200
    s=wind_noise_field.data.shape
    wind_noise_field.data=qcut(wind_noise_field.data.flatten(),wscale,
                                 labels=False,
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

    # Overlay the latent-space representation in the SE Pacific
    x=numpy.linspace(-160,-120,10)
    y=numpy.linspace(-75,-75+(40*16/18),10)
    latent_img = ax.pcolorfast(x,y,ls.reshape(10,10),
                               cmap='viridis',
                                 alpha=1.0,
                                 vmin=-3,
                                 vmax=3,
                                 zorder=1000)
   
# Get autoencoded versions of the validation data
model_save_file=("%s/ML_GCM/autoencoder/"+
                  "Epoch_%04d/autoencoder") % (
                      os.getenv('SCRATCH'),args.epoch)
autoencoder=tf.keras.models.load_model(model_save_file,compile=False)
# Also load the encoder (to get the latent state)
model_save_file=("%s/ML_GCM/autoencoder/"+
                  "/Epoch_%04d/encoder") % (
                      os.getenv('SCRATCH'),args.epoch)
encoder=tf.keras.models.load_model(model_save_file,compile=False)

# Load and compress the data - only at timepoint (i.e. hour%6==0)
def get_compressed(year,month,day,hour):
    prmsl=twcr.load('prmsl',datetime.datetime(year,month,day,hour),
                               version='2c')
    prmsl=to_analysis_grid(prmsl.extract(iris.Constraint(member=1)))
    t2m=twcr.load('air.2m',datetime.datetime(year,month,day,hour),
                               version='2c')
    t2m=to_analysis_grid(t2m.extract(iris.Constraint(member=1)))
    u10m=twcr.load('uwnd.10m',datetime.datetime(year,month,day,hour),
                               version='2c')
    u10m=to_analysis_grid(u10m.extract(iris.Constraint(member=1)))
    v10m=twcr.load('vwnd.10m',datetime.datetime(year,month,day,hour),
                               version='2c')
    v10m=to_analysis_grid(v10m.extract(iris.Constraint(member=1)))
    insol=to_analysis_grid(load_insolation(year,month,day,hour))

    # Convert the validation data into tensor format
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

    ict = tf.concat([t2m_t,prmsl_t,u10m_t,v10m_t,insol_t],2) # Now [79,159,5]
    ict = tf.reshape(ict,[1,79,159,5])
    result = autoencoder.predict_on_batch(ict)
    result = tf.reshape(result,[79,159,5])
    ls = encoder.predict_on_batch(ict)
    
    # Convert the encoded fields back to unnormalised cubes 
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
    return {'t2m':t2m_r,'prmsl':prmsl_r,'u10m':u10m_r,'v10m':v10m_r,'ls':ls}
    
# Get the compressed data at the selected time
dte=datetime.datetime(args.year,args.month,args.day,
                          int(args.hour),int(args.hour%1*60))
prevt=datetime.datetime(args.year,args.month,args.day,
                           int(args.hour)-int(args.hour)%6)
nextt=prevt+datetime.timedelta(hours=6)
s_previous=get_compressed(prevt.year,prevt.month,prevt.day,prevt.hour)
s_next=get_compressed(nextt.year,nextt.month,nextt.day,nextt.hour)
compressed={}
for var in ('t2m','prmsl','u10m','v10m'):
   cl=iris.cube.CubeList((s_previous[var],s_next[var])).merge_cube()
   compressed[var]=cl.interpolate([('time',dte)],iris.analysis.Linear())
w=(dte-prevt).total_seconds()/(nextt-prevt).total_seconds()
ls=s_previous['ls']*(1-w)+s_next['ls']*w

# Plot the two fields and a scatterplot for each variable
fig=Figure(figsize=(19.2,10.8),
           dpi=100,
           facecolor=(0.88,0.88,0.88,1),
           edgecolor=None,
           linewidth=0.0,
           frameon=False,
           subplotpars=None,
           tight_layout=None)
canvas=FigureCanvas(fig)

# Two maps, original and reconstructed
ax_compressed=fig.add_axes([0,0,1,1])
three_plot(ax_compressed,compressed['t2m'],
                         compressed['u10m'],
                         compressed['v10m'],
                         compressed['prmsl'],
                         ls)

# Render the figure as a png
fig.savefig('%s/%04d%02d%02d%02d%02d.png' % (args.opdir,args.year,
                                             args.month,args.day,
                                             int(args.hour),
                                             int(args.hour%1*60)))
