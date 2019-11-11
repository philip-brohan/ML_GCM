#!/usr/bin/env python

# Make a fixed noise field for wind-map plots.

import os
import iris
import numpy
import pickle

import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--resolution", help="Resolution for plot grid",
                    default=0.1,type=float,required=False)
parser.add_argument("--zoom", help="Scale factor for viewport (1=global)",
                    default=1,type=float,required=False)
parser.add_argument("--opfile", help="Output (pickle) file name",
                    default="%s/images/20CRv2c_global_4var/z.pkl" % \
                                           os.getenv('SCRATCH'),
                    type=str,required=False)
args = parser.parse_args()


# Nominal projection
cs=iris.coord_systems.RotatedGeogCS(90,180,0)

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

z=plot_cube(args.resolution,-180/args.zoom,180/args.zoom,
                             -90/args.zoom,90/args.zoom)
(width,height)=z.data.shape
z.data=numpy.random.rand(width,height)-0.5

z2=plot_cube(args.resolution*2,-180/args.zoom,180/args.zoom,
                             -90/args.zoom,90/args.zoom)
(width,height)=z2.data.shape
z2.data=numpy.random.rand(width,height)-0.5
z2=z2.regrid(z,iris.analysis.Linear())
z.data=z.data+z2.data

z4=plot_cube(args.resolution*4,-180/args.zoom,180/args.zoom,
                             -90/args.zoom,90/args.zoom)
(width,height)=z4.data.shape
z4.data=numpy.random.rand(width,height)-0.5
z4=z4.regrid(z,iris.analysis.Linear())
z.data=z.data+z4.data*100


pickle.dump( z, open( args.opfile, "wb" ) )
