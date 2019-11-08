#!/usr/bin/env python

# Retrieve surface weather variables from the global analysis
#  for one day.

import os
from tempfile import NamedTemporaryFile
import subprocess

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", help="Year",
                    type=int,required=True)
parser.add_argument("--month", help="Integer month",
                    type=int,required=True)
parser.add_argument("--day", help="Day of month",
                    type=int,required=True)
parser.add_argument("--opdir", help="Directory for output files",
                    default="%s/opfc/glm" % os.getenv('SCRATCH'))
args = parser.parse_args()
opdir="%s/%04d/%02d" % (args.opdir,args.year,args.month)
if not os.path.isdir(opdir):
    os.makedirs(opdir)

print("%04d-%02d-%02d" % (args.year,args.month,args.day))

# Mass directory to use
mass_dir="moose:/opfc/atm/global/prods/%04d.pp" % args.year

# Files to retrieve from
flist=[]
for hour in (0,6,12,18):
    for fcst in (0,3,6):
        flist.append("prods_op_gl-mn_%04d%02d%02d_%02d_%03d.pp" % (
                       args.year,args.month,args.day,hour,fcst))
        flist.append("prods_op_gl-mn_%04d%02d%02d_%02d_%03d.calc.pp" % (
                       args.year,args.month,args.day,hour,fcst))
        flist.append("prods_op_gl-up_%04d%02d%02d_%02d_%03d.calc.pp" % (
                       args.year,args.month,args.day,hour,fcst))

# Create the query file
qfile=NamedTemporaryFile(mode='w+',delete=False)
qfile.write("begin_global\n")
qfile.write("   pp_file = (")
for ppfl in flist:
    qfile.write("\"%s\"" % ppfl)
    if ppfl != flist[-1]:
        qfile.write(",")
    else:
        qfile.write(")\n")
qfile.write("end_global\n")
qfile.write("begin\n")
qfile.write("    stash = (16222,24,3236,31,3225,3226)\n")
qfile.write("    lbproc = 0\n")
qfile.write("end\n")
qfile.write("begin\n")
qfile.write("    stash = (5216,5206,5205,4203,4204)\n")
qfile.write("    lbproc = 128\n")
qfile.write("end\n")
qfile.close()

# Run the query
opfile="%s/%02d.pp" % (opdir,args.day)
subprocess.call("moo select -C %s %s %s" % (qfile.name,mass_dir,opfile),
                 shell=True)

os.remove(qfile.name)
