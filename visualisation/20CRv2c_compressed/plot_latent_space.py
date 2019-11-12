#!/usr/bin/env python

# Atmospheric state - near-surface temperature, u-wind, v-wind, and prmsl.
# Show the version after compression into a latant space.

import os
import IRData.opfc as opfc
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
from plots import quantile_normalise_t2m
from plots import plot_cube
from plots import wind_field
from plots import get_precip_colours
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

# Load the latent-space representation, and convert it back into normal space
model_save_file=("%s/Machine-Learning-experiments/"+
                  "convolutional_autoencoder_perturbations/"+
                  "multivariate_uk_centred/saved_models/"+
                  "Epoch_%04d/generator") % (
                      os.getenv('SCRATCH'),args.epoch)
generator=tf.keras.models.load_model(model_save_file,compile=False)

# Load the LS representation - interpolating in time as necesary
def ls_load_at_timepoint(year,month,day,hour):
    ls_file=(("%s/Machine-Learning-experiments/datasets/latent_space/"+
                  "%04d-%02d-%02d:%02d.tfd") %
                       (os.getenv('SCRATCH'),
                        year,month,day,hour))
    sict  = tf.read_file(ls_file)
    ls = tf.reshape(tf.parse_tensor(sict,numpy.float32),[1,100])
    result=generator.predict_on_batch(ls)
    result = tf.reshape(result,[79,159,4])
    t2m=dummy_cube()
    t2m.data = tf.reshape(result.numpy()[:,:,0],[79,159]).numpy()
    t2m = unnormalise_t2m(t2m)
    prmsl=dummy_cube()
    prmsl.data = tf.reshape(result.numpy()[:,:,1],[79,159]).numpy()
    prmsl = unnormalise_prmsl(prmsl)
    u10m=dummy_cube()
    u10m.data = tf.reshape(result.numpy()[:,:,2],[79,159]).numpy()
    u10m = unnormalise_wind(u10m)
    v10m=dummy_cube()
    v10m.data = tf.reshape(result.numpy()[:,:,3],[79,159]).numpy()
    v10m = unnormalise_wind(v10m)
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


mask=iris.load_cube("%s/fixed_fields/land_mask/opfc_global_2019.nc" % os.getenv('DATADIR'))

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

def plot_cube(resolution,xmin,xmax,ymin,ymax):

    lat_values=numpy.arange(ymin,ymax+resolution,resolution)
    latitude = iris.coords.DimCoord(lat_values,
                                    standard_name='latitude',
                                    units='degrees_north',
                                    coord_system=cs)
    lon_values=numpy.arange(xmin,xmax+resolution,resolution)
    longitude = iris.coords.DimCoord(lon_values,
                                     standard_name='longitude',
                                     units='degrees_east',
                                     coord_system=cs)
    dummy_data = numpy.zeros((len(lat_values), len(lon_values)))
    plot_cube = iris.cube.Cube(dummy_data,
                               dim_coords_and_dims=[(latitude, 0),
                                                    (longitude, 1)])
    return plot_cube

# Make the wind noise
def wind_field(uw,vw,zf,sequence=None,iterations=50,epsilon=0.003,sscale=1):
    # Random field as the source of the distortions
    z=pickle.load(open( zf, "rb" ) )
    z=z.regrid(uw,iris.analysis.Linear())
    (width,height)=z.data.shape
    # Each point in this field has an index location (i,j)
    #  and a real (x,y) position
    xmin=numpy.min(uw.coords()[0].points)
    xmax=numpy.max(uw.coords()[0].points)
    ymin=numpy.min(uw.coords()[1].points)
    ymax=numpy.max(uw.coords()[1].points)
    # Convert between index and real positions
    def i_to_x(i):
        return xmin + (i/width) * (xmax-xmin)
    def j_to_y(j):
        return ymin + (j/height) * (ymax-ymin)
    def x_to_i(x):
        return numpy.minimum(width-1,numpy.maximum(0, 
                numpy.floor((x-xmin)/(xmax-xmin)*(width-1)))).astype(int)
    def y_to_j(y):
        return numpy.minimum(height-1,numpy.maximum(0, 
                numpy.floor((y-ymin)/(ymax-ymin)*(height-1)))).astype(int)
    i,j=numpy.mgrid[0:width,0:height]
    x=i_to_x(i)
    y=j_to_y(j)
    # Result is a distorted version of the random field
    result=z.copy()
    # Repeatedly, move the x,y points according to the vector field
    #  and update result with the random field at their locations
    ss=uw.copy()
    ss.data=numpy.sqrt(uw.data**2+vw.data**2)
    if sequence is not None:
        startsi=numpy.arange(0,iterations,3)
        endpoints=numpy.tile(startsi,1+(width*height)//len(startsi))
        endpoints += sequence%iterations
        endpoints[endpoints>=iterations] -= iterations
        startpoints=endpoints-25
        startpoints[startpoints<0] += iterations
        endpoints=endpoints[0:(width*height)].reshape(width,height)
        startpoints=startpoints[0:(width*height)].reshape(width,height)
    else:
        endpoints=iterations+1 
        startpoints=-1       
    for k in range(iterations):
        x += epsilon*vw.data[i,j]
        x[x>xmax]=xmax
        x[x<xmin]=xmin
        y += epsilon*uw.data[i,j]
        y[y>ymax]=y[y>ymax]-ymax+ymin
        y[y<ymin]=y[y<ymin]-ymin+ymax
        i=x_to_i(x)
        j=y_to_j(y)
        update=z.data*ss.data/sscale
        update[(endpoints>startpoints) & ((k>endpoints) | (k<startpoints))]=0
        update[(startpoints>endpoints) & ((k>endpoints) & (k<startpoints))]=0
        result.data[i,j] += update
    return result

wind_pc=plot_cube(0.2,-180/args.zoom,180/args.zoom,
                      -90/args.zoom,90/args.zoom)   
rw=iris.analysis.cartography.rotate_winds(u10m,v10m,cs)
u10m = rw[0].regrid(wind_pc,iris.analysis.Linear())
v10m = rw[1].regrid(wind_pc,iris.analysis.Linear())
seq=(dte-datetime.datetime(2000,1,1)).total_seconds()/3600
wind_noise_field=wind_field(u10m,v10m,args.zfile,sequence=int(seq*5),epsilon=0.01)

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
for lat in range(-90,95,5):
    lwd=0.75
    x=[]
    y=[]
    for lon in range(-180,181,1):
        rp=iris.analysis.cartography.rotate_pole(numpy.array(lon),
                                                 numpy.array(lat),
                                                 args.pole_longitude,
                                                 args.pole_latitude)
        nx=rp[0]+args.npg_longitude
        if nx>180: nx -= 360
        ny=rp[1]
        if(len(x)==0 or (abs(nx-x[-1])<10 and abs(ny-y[-1])<10)):
            x.append(nx)
            y.append(ny)
        else:
            ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                               zorder=10))
            x=[]
            y=[]
    if(len(x)>1):        
        ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                           zorder=10))

for lon in range(-180,185,5):
    lwd=0.75
    x=[]
    y=[]
    for lat in range(-90,90,1):
        rp=iris.analysis.cartography.rotate_pole(numpy.array(lon),
                                                 numpy.array(lat),
                                                 args.pole_longitude,
                                                 args.pole_latitude)
        nx=rp[0]+args.npg_longitude
        if nx>180: nx -= 360
        ny=rp[1]
        if(len(x)==0 or (abs(nx-x[-1])<10 and abs(ny-y[-1])<10)):
            x.append(nx)
            y.append(ny)
        else:
            ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                               zorder=10))
            x=[]
            y=[]
    if(len(x)>1):        
        ax.add_line(Line2D(x, y, linewidth=lwd, color=(0.4,0.4,0.4,1),
                           zorder=10))

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


# Plot the T2M
t2m_pc=plot_cube(0.05,-180/args.zoom,180/args.zoom,
                      -90/args.zoom,90/args.zoom)   
t2m = t2m.regrid(t2m_pc,iris.analysis.Linear())
t2m=quantile_t2m(t2m)
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
latent_img = ax2.pcolorfast(x,x,ls.numpy().reshape(10,10),
                           cmap='viridis',
                             alpha=1.0,
                             vmin=-5,
                             vmax=5,
                             zorder=1000)

# Render the figure as a png
fig.savefig('%s/%04d%02d%02d%02d%02d.png' % (args.opdir,args.year,
                                             args.month,args.day,
                                             int(args.hour),
                                             int(args.hour%1*60)))
