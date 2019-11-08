# Data loading functions for the MO analysis video

import datetime
import IRData.opfc as opfc
import iris

# We want to modify the present temperature by emphasising the difference
#  with the climatological average, and reducing the diurnal variability.
# To enable this make an estimate of the climatological value (mean
#  from 'before' to 'after'), and the current diurnal variability (mean 
#  at current time of day, minus daily mean, over the same period)
# Returns two iris cubes (climatology and diurnal cycle size).
def load_recent_temperatures(dte, # current time (datetime.datetime)
                             before=5, # use this many days before
                             after=5): # use this many days after
    stt=dte-datetime.timedelta(days=before)
    ent=dte+datetime.timedelta(days=after)
    ct=stt
    tcount=0
    dcount=0
    tavg=None
    davg=None
    while ct<ent:
        try:
            ttmp=opfc.load('air.2m',ct,model='global')
            tcount += 1
            if tavg is None:
                tavg = ttmp.copy()
            else:
                tavg.data += ttmp.data
            if ct.hour==dte.hour:
                dcount += 1
                if davg is None:
                    davg = ttmp.copy()
                else:
                    davg.data += ttmp.data
        except:
            ct += datetime.timedelta(hours=1)
            continue
        ct += datetime.timedelta(hours=1)
    tavg.data /= tcount
    davg.data /= dcount
    davg.data -= tavg.data
    return (tavg,davg)
    
# We're plotting log precip - so interpolate in log space tooif dte.minute<30:
def load_li_precip(dte): # Time to plot (datetime.datetime)
    if dte.minute<30:
        prevt=dte-datetime.timedelta(minutes=30+dte.minute)
    else:
        prevt=dte-datetime.timedelta(minutes=dte.minute-30)
    nextt=prevt+datetime.timedelta(hours=1)
    prevp=opfc.load('prate_a',prevt,model='global')
    prevp.data += 1.0e-7
    prevp.data=numpy.log(prevp.data)
    nextp=opfc.load('prate_a',nextt,model='global')
    nextp.data += 1.0e-7
    nextp.data=numpy.log(nextp.data)
    w=(dte-prevt).total_seconds()/(nextt-prevt).total_seconds()
    precip=prevp.copy()
    precip.data=prevp.data*(1-w)+nextp.data*w
    precip.data += 15
    precip.data /= 11
    precip.data[precip.data<0] = 0
    precip.data[precip.data>1] = 1

# Sea-ice data is provided hourly, but only updated daily
#  so do an explicit interpolation over the day.
def load_di_icec(dte):
    dte1=datetime.datetime(dte.year,dte.month,dte.day,0)
    dte2=dte1+datetime.timedelta(days=1)
    try:
        icec=opfc.load('icec',dte1,model='global')
        tmp =opfc.load('icec',dte2,model='global')
        tmp.attributes=icec.attributes
        icec=iris.cube.CubeList((icec,tmp)).merge_cube()
        icec=icec.interpolate([('time',dte)],iris.analysis.Linear())
    except:
        icec=opfc.load('icec',dte1-datetime.timedelta(days=1),model='global')
