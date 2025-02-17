# coding: utf-8
#!/usr/bin/env python
# coding: utf-8

# Python version of radiative kernels
# V. Cooper (2023)
# - 0) Before running, make timeseries on std pressure levels
# - 1) Compute climatology
# - 2) Horizontal regridding
# - 3) Feedback calcs converted from Pendergrass public scripts

# Before running these scripts, postprocess the 3D fields into 
# timeseries that are on pressure levels (not hybrid-sigma levels as 
# in the native output). After timeseries files are prepared, they can
# be loaded here to make a climatology, regrid onto kernel grid, then compute feedbacks.

# Note: if changing the pressure levels from CMIP to something else,
# - don't run "ncl scripts/calcp.ncl"
# - run "ncl tools/calcdp_plev.ncl"
# - run "ncl tools/t_kernel_to_plev.ncl"
# - run "ncl tools/q_kernel_to_plev.ncl"

# Note: the following 2 scripts are replaced by the timeseries creation, 
# which already puts fields on pressure levels. No need to do these:
# - don't run "ncl tools/convert_base_to_plevs.ncl"
# - don't run "ncl tools/convert_change_to_plevs.ncl"


import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
# import xesmf as xe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import copy
import pandas as pd
import cmocean as cmo
from cartopy.util import add_cyclic_point
import seaborn as sns
#%matplotlib inline

import xesmf as xe

# Check if a command-line argument is provided
if len(sys.argv) < 2:
    print("Error: Please provide a case name as an argument.")
    print("Usage: python allkernel_enter.py <case_name>")
    sys.exit(1)  # Exit the script if no argument is provided

## COMPUTE CLIMATOLOGY

## these are the dummy files for format and grid matching
path = './'

## these are demodata, will be replaced later after we match
## the file formatting
basefields = xr.open_dataset(path +'demodata/'+ 'basefields.nc~')
changefields = xr.open_dataset(path +'demodata/'+ 'changefields.nc~')
PSdemo = xr.open_dataset(path + 'kernels/PS.nc')
pdiff = xr.open_dataset(path + 'dp_plev.nc')
gw = xr.open_dataset(path +'kernels/'+ 't.kernel.nc').gw

varlist = ['FSDS','FSNS','Q','T','FLNS',
           'FLNSC','FLNT','FLNTC','FSNSC',   'FSDSC',
           'FSNT','FSNTC','PS'] #'TS' will be loaded by default
## WATCH OUT YOU ACTUALLY NEED FSDSC for clear-sky linearity test

#### SELECT CASES HERE
path = '/glade/derecho/scratch/vcooper/processed/'

External_switch = False ## False for CAM, True for GFDL
strat_mask = False  ## True is default which masks stratosphere changes 
cs_linearity_test = False ## True only if doing clear-sky linearity test

## SELECT BASECASE
## Derecho era
basecase = 'f.e215.F2000climoCAM5.f19_f19.LGMRholo.003' ## CAM5
# basecase = 'f.e215.F2000climoCAM4.f19_f19.LGMRholo.003' ## CAM4

## Cheyenne era
# basecase = 'f2000climo_f19_f19_holo_3_23_23' ## CAM6

# basecase = 'f.e1221.F_2000_CAM5.f19_f19.LGMRholo.002' ## CAM5 OLD DONT USE

# basecase = 'f.e1221.F_2000.f19_f19.LGMRholo.002' ## CAM4
#basecase = 'f.e1221.F_2000.f19_f19.LGMRholo.003' ## CAM4 v3

## SELECT CHANGECASE
## pliocene version
# case = 'f.e215.F2000climo.f19_f19.tplio_vF'
# case = 'f.e215.F2000climo.f19_f19.tplio_vFpliomip'
# case = 'f.e215.F2000climo.f19_f19.annanplio_v2.1'
case = sys.argv[1] # automatically assumes string, so enter f.e215.F2000climo.f19_f19.tplio

## cam5
# case = 'f.e215.F2000climoCAM5.f19_f19.tplio_vF'
# case = 'f.e215.F2000climoCAM5.f19_f19.longrunmip.002'
# case = 'f.e215.F2000climoCAM5.f19_f19.tplio_vFpliomip'
# case = 'f.e215.F2000climoCAM5.f19_f19.tplio_vFcloud'
# case = 'f.e215.F2000climoCAM5.f19_f19.tplio_vFpliovar'
# case = 'f.e215.F2000climoCAM5.f19_f19.annanplio_v2.1'
# case = 'f.e215.F2000climoCAM5.f19_f19.tplio_vFneg05p'
# case = 'f.e215.F2000climoCAM5.f19_f19.tplio_vFpos95p'

## cam4
# case = 'f.e215.F2000climoCAM4.f19_f19.tplio_vF'
# case = 'f.e215.F2000climoCAM4.f19_f19.longrunmip.002'
# case = 'f.e215.F2000climoCAM4.f19_f19.tplio_vFpliomip'
# case = 'f.e215.F2000climoCAM4.f19_f19.annanplio_v2.1'
# case = 'f.e215.F2000climoCAM4.f19_f19.tplio_vFpliovar'
# case = 'f.e215.F2000climoCAM4.f19_f19.tplio_vFcloud'
# case = 'f.e215.F2000climoCAM4.f19_f19.tplio_vFneg05p'
# case = 'f.e215.F2000climoCAM4.f19_f19.tplio_vFpos95p'

## LGM vintage
# case = 'f2000climo_f19_f19_2xco2by05_05'
# case = 'f2000climo_f19_f19_2xc02_05'
# case = 'f2000climo_f19_f19_pattern_05'
# case = 'f2000climo_f19_f19_LGMRby05_05_fixed'

# case = 'f.e1221.F_2000_CAM5.f19_f19.longrunmip-0p5.002'
# case = 'f.e1221.F_2000_CAM5.f19_f19.longrunmip.002'
# case = 'f.e1221.F_2000_CAM5.f19_f19.LGMR-0p5.002'
# case = 'f.e1221.F_2000_CAM5.f19_f19.LGMRlgm.001'
# case = 'f.e1221.F_2000_CAM5.f19_f19.lgmDA.002'
# case = 'f.e1221.F_2000_CAM5.f19_f19.lgmDA-0p5.002'
# case = 'f.e1221.F_2000_CAM5.f19_f19.amrhein.002.1'
# case = 'f.e1221.F_2000_CAM5.f19_f19.amrhein-0p5.002'
# case = 'f.e1221.F_2000_CAM5.f19_f19.annan-seasbyamr.002.1'
# case = 'f.e1221.F_2000_CAM5.f19_f19.annan-0p5.002'

# case = 'f.e1221.F_2000.f19_f19.longrunmip-0p5.002'
# case = 'f.e1221.F_2000.f19_f19.longrunmip2x.002'
# case = 'f.e1221.F_2000.f19_f19.LGMR-0p5.002'
# case = 'f.e1221.F_2000.f19_f19.LGMRlgm.001'
# case = 'f.e1221.F_2000.f19_f19.lgmDA.002'
# case = 'f.e1221.F_2000.f19_f19.lgmDA-0p5.002'
# case = 'f.e1221.F_2000.f19_f19.amrhein.002.1'
# case = 'f.e1221.F_2000.f19_f19.amrhein-0p5.002'
# case = 'f.e1221.F_2000.f19_f19.annan-seasbyamr.002.1'
# case = 'f.e1221.F_2000.f19_f19.annan-0p5.002'

changecase = case
print('basecase', basecase)
print('changecase', changecase)

############### run script

## load in the full timeseries
fname = basecase + '.cam.h0.tseries_interp.nc'
basecam_out = xr.open_dataset(path + basecase +'/'+fname)
fname = case + '.cam.h0.tseries_interp.nc'
cam_out = xr.open_dataset(path + case +'/'+fname)

# ## TEMPORARY FSDSC FIX (this variable was not originally included)
# tpath = '/glade/scratch/derecho/vcooper/archive/' + case + '/atm/proc/tseries/month_1/'
# fname = case + '.cam.h0.FSDSC.000101-003012.nc'
# FSDSC = xr.open_dataset(tpath + fname).sel(
#     time=slice('0006-01-02','0031-01-01')).FSDSC
# cam_out['FSDSC'] = (['time','lat','lon'], FSDSC.values)

# tpath = '/glade/scratch/derecho/vcooper/archive/' + basecase + '/atm/proc/tseries/month_1/'
# fname = basecase + '.cam.h0.FSDSC.000101-003012.nc'
# baseFSDSC = xr.open_dataset(tpath + fname).sel(
#     time=slice('0006-01-02','0031-01-01')).FSDSC
# basecam_out['FSDSC'] = (['time','lat','lon'], baseFSDSC.values)

## this is stupid code to fix the months
init_bound = np.array(xr.cftime_range(start="0006", periods=1, freq="MS", calendar="noleap")[0])
dumtime = np.hstack([init_bound,cam_out.time.values])
newtime = dumtime[0:-1]
cam_out['time'] = newtime
basecam_out['time'] = newtime

cam_climo = cam_out#.groupby('time.month').mean()
basecam_climo = basecam_out#.groupby('time.month').mean()

if External_switch == False:
    cam_climo = cam_out.groupby('time.month').mean()
    basecam_climo = basecam_out.groupby('time.month').mean()
    
### Change some fields to match demodata
if External_switch == False:
    basecase = basecam_climo.rename(
        {'month':'time','T':'temp','TS':'ts','lev':'lev_p'})

    changecase = cam_climo.rename(
        {'month':'time','T':'temp','TS':'ts','lev':'lev_p'})
    
    
## OPTION TO REPLACE CAM WITH EXTERNAL MODEL DATA
if External_switch == True:
    ## GFDL
    path = '/glade/derecho/scratch/vcooper/processed/'

    ## already in climatology form
    basecase = xr.open_dataset(path + 'gfdl_holo/gfdl_holo.nc').rename(
            {'T':'temp','TS':'ts','lev':'lev_p','swdn_sfc_clr':'FSDSC'})

    #case = 'gfdl_2xCO2'
    #changecase = xr.open_dataset(path + 'gfdl_2xCO2/gfdl_2xCO2.nc').rename(
    #        {'T':'temp','TS':'ts','lev':'lev_p','swdn_sfc_clr':'FSDSC'})
    case = 'gfdl_lgm'
    changecase = xr.open_dataset(path + 'gfdl_lgm/gfdl_lgm.nc').rename(
             {'T':'temp','TS':'ts','lev':'lev_p','swdn_sfc_clr':'FSDSC'})
    
    ## shortcut to select only the kernel CMIP pressure levels
    basecase = basecase.sel(lev_p=pdiff.plev.values/100)
    changecase = changecase.sel(lev_p=pdiff.plev.values/100)
    

#### Regrid horizontal fields to match kernel 1 degree resolution

diff = changecase - basecase
## xesmf issue with date and datesec, avoiding it here:
regridvars = ['temp',
                         # 'date',
                         # 'datesec',
                         'Q',
                         'PS',
                         'ts',
                         'FSDS',
                         'FSNS',
                         'FLNS',
                         'FLNSC',
                         'FLNT',
                         'FLNTC',
                         'FSNSC',
                         'FSNT',
                         'FSNTC']

if (cs_linearity_test == True):
    regridvars = np.append(regridvars, 'FSDSC')
    
    
newgrid = basefields.ts[0].to_dataset()
newgrid = basefields

data_for_regridding = basecase.ts[0].to_dataset()
data_for_regridding = basecase

if External_switch == False:
    filename_wts = 'bilinear_2deg_to_1deg.nc'
else:
    filename_wts = 'external_regrid_wts.nc'

regridder = xe.Regridder(data_for_regridding, newgrid,
                         method='bilinear',
                         periodic=True,
                         # extrap_method='inverse_dist',extrap_num_src_pnts=8,
                         filename=filename_wts,
                         reuse_weights=True)

base_regrid = regridder(basecase[regridvars])
diff_regrid = regridder(diff[regridvars])

## avoiding xesmf issue
if External_switch == False:
    base_regrid['date'] = basecase['date']
    base_regrid['datesec'] = basecase['datesec']

    diff_regrid['date'] = diff['date']
    diff_regrid['datesec'] = diff['datesec']
    
    
## add some variables to file that seem to not change
## note: pdiffs are constant because they are diff between 
## the native kernels and the CMIP5 standard levels
base_regrid['plev'] = base_regrid.lev_p * 100
base_regrid['gw'] = ('lat', gw.values)
base_regrid['pdiff'] = (['time','lev_p','lat','lon'], pdiff.dp.values)

## note that pdiff is the same for the changefields
diff_regrid['plev'] = diff_regrid.lev_p * 100
diff_regrid['gw'] = ('lat', gw.values)
diff_regrid['pdiff'] = (['time','lev_p','lat','lon'], pdiff.dp.values)


## option to calculate only clear-sky
## accomplished by making all-sky fields equal the clear-sky fields
if (cs_linearity_test == True):
    diff_regrid['FLNT'] = diff_regrid['FLNTC']
    diff_regrid['FSNT'] = diff_regrid['FSNTC']
    diff_regrid['FLNS'] = diff_regrid['FLNSC']
    diff_regrid['FSNS'] = diff_regrid['FSNSC']
    
    base_regrid['FLNT'] = base_regrid['FLNTC']
    base_regrid['FSNT'] = base_regrid['FSNTC']
    base_regrid['FLNS'] = base_regrid['FLNSC']
    base_regrid['FSNS'] = base_regrid['FSNSC']
    
    diff_regrid['FSDS'] = diff_regrid['FSDSC']
    base_regrid['FSDS'] = base_regrid['FSDSC']
    
    
    
################################################
################################################
################################################
################ CALC FEEDBACKS ################
## Portion below is adapted from Pendergrass  ##
################################################
################################################


# File with the changes in climate: (ts, temp) (TS,T,Q)
varlist2d = ['ts','gw','FSNS', 'FSDS', 'FSNT']

# changefile=xr.open_dataset('./demodata/changefields.nc')[varlist2d]
# changefile3d=xr.open_dataset('./changefields.plev.nc')
# basefile=xr.open_dataset('./demodata/basefields.nc')[varlist2d]
# basefile3d=xr.open_dataset('./basefields.plev.nc')

changefile=diff_regrid#[varlist2d]
changefile3d=diff_regrid
basefile=base_regrid#[varlist2d]
basefile3d=base_regrid


###################

## Read air temperature kernel 
# ta_kernel_hybrid=ncread('kernels/t.kernel.nc','FLNT');
ta_kernel=xr.open_dataset('t.kernel.plev.nc')

## VTC add section to read in pressure levels
p_Pa=ta_kernel.plev
p_hPa=ta_kernel.lev_p 

## this must be generated by calcdp_plev.ncl script
pdiff=xr.open_dataset('dp_plev.nc').dp/100

# p=repmat(permute(repmat(p_hPa,[1 12]),[3 4 1 2]),[288 192 1 1]);
p = p_hPa.values[np.newaxis, :, np.newaxis, np.newaxis] * np.ones(ta_kernel.FLNT.shape)


###################

## Read in coordinate info
lat=xr.open_dataset('./kernels/PS.nc').lat
lon=xr.open_dataset('./kernels/PS.nc').lon
gw=xr.open_dataset('./kernels/t.kernel.nc').gw ## Gaussian weights for CESMgrid
# lev=ncread('kernels/t.kernel.nc','lev'); ## dont need

## Make an area weighting matrix
weight=np.tile(gw.values[:,np.newaxis], len(lon))
weight=weight/np.nansum(weight)
# print(weight.sum())

## Read surface temperature change
dts=changefile.ts

## Calculate the change in global mean surface temperature
dts_globalmean= (dts * weight).sum(dim=('lat','lon')).mean(dim='time')
print('Global mean dTS: ', dts_globalmean.values)


## Temperature feedback calculation

## Read TOA Longwave surface temperature kernel
ts_kernel=xr.open_dataset('./kernels/ts.kernel.nc').FLNT
if (cs_linearity_test == True):
    ts_kernel=xr.open_dataset('./kernels/ts.kernel.nc').FLNTC

## Multiply monthly mean TS change by the TS kernels (function of
## lat, lon, month) (units W/m2)
dLW_ts=ts_kernel * dts
#dLW_ts.mean(dim='time').plot()

## Read air temperature change [lon,lat,level,month]
dta=changefile3d.temp

## Non-pressure level version:
## Read midpoint pressure for each grid cell (lat,lon,level,month), [Pa]
## VTC adjusted this above to be pressure levels
## p=ncread('p_sigma.nc','pmid')/100; %[hPa] 

## Crude tropopause estimate: 100 hPa in the tropics, lowering with
## cosine to 300 hPa at the poles.
x=np.cos(np.deg2rad(lat))
p_tropopause_zonalmean=300-200*x
## VTC
##p_tropopause= ...
##    repmat(permute(repmat(permute(repmat(p_tropopause_zonalmean', ...
##                                         [length(lon) 1]),[1 2 3]),[1 ...
##                    1 length(lev)]),[1 2 3 4]),[1 1 1 12]);
p_tropopause = p_tropopause_zonalmean.values[
    np.newaxis, np.newaxis,:, np.newaxis] * np.ones(ta_kernel.FLNT.shape)
p_tropopause = xr.DataArray(p_tropopause,dims=changefile3d.dims, coords = changefile3d.coords)

if (strat_mask == False):
    ## True is default, but for cases with no forcing,
    ## maybe we don't want to mask these bc they are part of feedback
    ## make tropopause be at 0 hPa so that nothing is masked
    p_tropopause -= p_tropopause
    
## Set the temperature change to zero in the stratosphere (mask out stratosphere)
dta=xr.where(p>=p_tropopause, dta, np.nan)

## Convolve air temperature kernel with air temperature change
## VTC
## dLW_ta=squeeze(sum(ta_kernel.*dta,3));
dLW_ta=ta_kernel.FLNT * dta.values * pdiff.values
if (cs_linearity_test == True):
    dLW_ta=ta_kernel.FLNTC * dta.values * pdiff.values
    
## Add the surface and air temperature response; Take the annual
## average and global area average 
dLW_t_globalmean = -(
    (dLW_ta.sum(dim='plev') + dLW_ts).mean(dim='time') * weight).sum()

## Divide by the global annual mean surface warming (units: W/m2/K)
t_feedback=dLW_t_globalmean / dts_globalmean

# print('Temperature feedback: ', str(t_feedback.values), ' W m^-2 K^-1')

######### ALBEDO FEEDBACK ########
## Collect surface shortwave radiation fields to calculate albedo.
## Alternatively, you might already have the change in albedo - that
## would work too.
SW_sfc_net_1=basefile.FSNS
SW_sfc_down_1=basefile.FSDS
SW_sfc_net_2=changefile.FSNS+SW_sfc_net_1
SW_sfc_down_2=changefile.FSDS+SW_sfc_down_1

alb1= 1 - SW_sfc_net_1/SW_sfc_down_1
alb1 = xr.where(np.isnan(alb1), 0, alb1)

alb2= 1 - SW_sfc_net_2/SW_sfc_down_2
alb2 = xr.where(np.isnan(alb2), 0, alb2)

dalb=(alb2-alb1)*100

alb_kernel = xr.open_dataset('./kernels/alb.kernel.nc').FSNT
if (cs_linearity_test == True):
    alb_kernel = xr.open_dataset('./kernels/alb.kernel.nc').FSNTC
    
dSW_alb = alb_kernel * dalb

## average and global area average 
dSW_alb_globalmean = (dSW_alb.mean(dim='time') * weight).sum()
alb_feedback = dSW_alb_globalmean / dts_globalmean

# print('Surf. albedo feedback: ', str(alb_feedback.values), ' W m^-2 K^-1')


########### WATER VAPOR FEEDBACK ##########
## Calculate the change in moisture per degree warming at constant relative humidity. 
q1=basefile3d.Q
t1=basefile3d.temp


## addpath scripts/
## VTC script for
def calcsatspechum(t_in, p_in):
    ## T is temperature, P is pressure in hPa 

    ## Formulae from Buck (1981):
    es = (1.0007+(3.46e-6 * p_in)) * 6.1121 * (
        np.exp(17.502*(t_in - 273.15) / (240.97+(t_in - 273.15))))
    
    wsl = 0.622 * es / (p_in - es) ## saturation mixing ratio wrt liquid water (g/kg)
    
    es = (1.0003 + ( 4.18e-6 * p_in)) * 6.1115 * (
        np.exp(22.452 * (t_in - 273.15) / (272.55 + (t_in - 273.15))))
    
    wsi = 0.622 * es / (p - es) ## saturation mixing ratio wrt ice (g/kg)
    
    ws = wsl
    
    ws = xr.where(t_in < 273.15, wsi, ws)
    
    qs = ws / (1+ws) ## saturation specific humidity, g/kg
    return(qs)


qs1 = calcsatspechum(t1,p) #g/kg
qs2 = calcsatspechum(t1+dta,p) #g/kg
dqsdt = (qs2 - qs1) / dta
rh = q1 / qs1
dqdt = rh * dqsdt

## Read kernels
q_LW_kernel = xr.open_dataset('q.kernel.plev.nc').FLNT
q_SW_kernel = xr.open_dataset('q.kernel.plev.nc').FSNT
if (cs_linearity_test == True):
    q_LW_kernel = xr.open_dataset('q.kernel.plev.nc').FLNTC
    q_SW_kernel = xr.open_dataset('q.kernel.plev.nc').FSNTC

## Normalize kernels by the change in moisture for 1 K warming at
## constant RH
q_LW_kernel = q_LW_kernel.values / dqdt;
q_SW_kernel = q_SW_kernel.values / dqdt;

## Read the change in moisture
dq = changefile3d.Q


## Set the moisture change to zero in the stratosphere (mask out stratosphere)
dq = xr.where(p>=p_tropopause, dq, np.nan)

## Convolve moisture kernel with change in moisture
dLW_q = q_LW_kernel * dq.values * pdiff.values
dSW_q = q_SW_kernel * dq.values * pdiff.values


## Add the LW and SW responses. Note the sign convention difference
## between LW and SW!
dR_q_globalmean = (
    (-dLW_q + dSW_q).sum(dim='lev_p').mean(dim='time') * weight).sum()

## Divide by the global annual mean surface warming (units: W/m2/K)
q_feedback = dR_q_globalmean / dts_globalmean

# print('Water vapor feedback: ', str(q_feedback.values), ' W m^-2 K^-1')

## WATER VAPOR with Log of Q ###
## Calculate the change in moisture per degree warming at constant relative humidity. 
## Run the accompanying NCL script with your input files, or
## implement here.                                                             

q0 = q1 ## AP used q1 for this above #basefile3d.Q #kg/kg 

## all of these are set above
#t1 = basefile3d.temp set above
#dta = changefile3d.temp ## set above
#qs1 = calcsatspechum(t1,p); % g/kg
#qs2 = calcsatspechum(t1+dta,p); % g/kg
#dqsdt = (qs2 - qs1)./dta;

## slightly different from above
rh = 1000*q0 / qs1
dqdt = rh * dqsdt ## assume constant RH
dlogqdt= dqdt / (1000 * q0) ## convert denominator to g/kg

## Re-read kernels, 
## normalize by the change in moisture for 1 K warming at
## constant RH using log Q
q_LW_kernel = xr.open_dataset('q.kernel.plev.nc').FLNT.values
if (cs_linearity_test == True):
    q_LW_kernel = xr.open_dataset('q.kernel.plev.nc').FLNTC.values
logq_LW_kernel = q_LW_kernel / dlogqdt
q_SW_kernel = xr.open_dataset('q.kernel.plev.nc').FSNT.values
if (cs_linearity_test == True):
    q_SW_kernel = xr.open_dataset('q.kernel.plev.nc').FSNTC.values
logq_SW_kernel = q_SW_kernel / dlogqdt

## all set above
## Read the change in moisture
## dq=ncread(changefile3d,'Q');
## Mask out the stratosphere
##dq=dq.*(p>=p_tropopause);

dlogq = dq / q0

## Convolve moisture kernel with change in moisture
dLW_logq = logq_LW_kernel.values * dlogq * pdiff.values
dSW_logq = logq_SW_kernel.values * dlogq * pdiff.values

## Add the LW and SW responses. 
## Note the sign convention difference LW and SW
dR_logq_globalmean = (
    (-dLW_logq + dSW_logq).sum(dim='lev_p').mean(dim='time') * weight).sum()

## Divide by the global annual mean surface warming (units: W/m2/K)
logq_feedback = dR_logq_globalmean / dts_globalmean

## this seems to give same results, must be a bug
# print('logQ WV feedback: ', str(logq_feedback.values), ' W m^-2 K^-1')
# print('linQ WV feedback: ', str(q_feedback.values), ' W m^-2 K^-1')



###### PLANCK FEEBACK ########
## Project surface temperature change into height 
##VTC
##dts3d=repmat(permute(dts,[1 2 4 3]),[1 1 30 1]);
dts3d = dts + changefile3d.temp-changefile3d.temp

## Mask stratosphere
dt_planck = xr.where(p>=p_tropopause, dts3d, np.nan)

## Convolve air temperature kernel with 3-d surface air temp change
##VTC
##dLW_planck=squeeze(sum(ta_kernel_hybrid.*dt_planck,3));
# dLW_planck = squeeze(sum(ta_kernel * dt_planck.*pdiff,3))
dLW_planck=ta_kernel.FLNT * dt_planck.values * pdiff.values
if (cs_linearity_test==True):
    dLW_planck=ta_kernel.FLNTC * dt_planck.values * pdiff.values
    

## Take the annual average and global area average; incorporate the
## part due to surface temperature change itself 
dLW_planck_globalmean = -(
    (dLW_planck.sum(dim='plev') + dLW_ts).mean(dim='time') * weight).sum()

## Divide by the global annual mean surface warming (units: W/m2/K)
planck_feedback=dLW_planck_globalmean / dts_globalmean

## Lapse rate feedback                                                                                                                                                                 
## Calculate the departure of temperature change from the surface
## temperature change
dt_lapserate=xr.where(p>=p_tropopause, dta-dt_planck, np.nan)

## Convolve air temperature kernel with 3-d surface air temp change
## VTC
## dLW_lapserate=squeeze(sum(ta_kernel.*dt_lapserate,3));
# dLW_lapserate=squeeze(sum(ta_kernel.*dt_lapserate.*pdiff,3));
dLW_lapserate = ta_kernel.FLNT * dt_lapserate.values * pdiff.values
if (cs_linearity_test==True):
    dLW_lapserate = ta_kernel.FLNTC * dt_lapserate.values * pdiff.values
    

## Take the annual average and global area average 
dLW_lapserate_globalmean = -(
    (dLW_lapserate.sum(dim='plev')).mean(dim='time') * weight).sum()

## Divide by the global annual mean surface warming (units: W/m2/K)
lapserate_feedback = dLW_lapserate_globalmean / dts_globalmean

# print('Planck feedback: ', 
#       str(planck_feedback.values), ' W m^-2 K^-1')
# print('Lapse rate feedback: ', 
#       str(lapserate_feedback.values), ' W m^-2 K^-1')

### SANITY CHECK: Do the Planck and lapse-rate feedbacks add up to
### the total temperature feedback? (They should)

## Planck + lapse rate feedback
total_t_feedback = planck_feedback+lapserate_feedback

# print('Temperature feedback: ',
#       str(t_feedback.values), ' W m^-2 K^-1')
# print('Planck+lapse rate components: ',
#       str(total_t_feedback.values), ' W m^-2 K^-1')


##### CLOUD FEEDBACK ######

## STEP 1. Calculate total-sky and clear-sky feedbacks
lev = ta_kernel.plev.values ## this is in Pa

## Read TOA Longwave surface temperature kernel
ts_kernel_clearsky = xr.open_dataset('./kernels/ts.kernel.nc').FLNTC

## Multiply monthly mean TS change by the TS kernels 
## (function of lat, lon, month) (units W/m2)
dLW_ts_cs = ts_kernel_clearsky * dts 

## Read clear-sky air temperature kernel
ta_kernel_clearsky = xr.open_dataset('./t.kernel.plev.nc').FLNTC

## Convolve air temperature kernel with air temperature change
dLW_ta = dLW_ta.sum(dim='plev')
dLW_ta_cs = (ta_kernel_clearsky.values * dta * pdiff.values).sum(dim='lev_p')

## ALBEDO clear sky

## Read TOA albedo kernel
alb_kernel_clearsky = xr.open_dataset('./kernels/alb.kernel.nc').FSNTC

dSW_alb_cs = alb_kernel_clearsky.values * dalb

## WATER VAPOR clear sky

## read kernels
q_LW_kernel = xr.open_dataset('./q.kernel.plev.nc').FLNT
q_SW_kernel = xr.open_dataset('./q.kernel.plev.nc').FSNT
q_LW_kernel_clearsky = xr.open_dataset('./q.kernel.plev.nc').FLNTC
q_SW_kernel_clearsky = xr.open_dataset('./q.kernel.plev.nc').FSNTC

## Normalize kernels by the change in moisture for 1 K warming at
## constant RH (linear)
rh = q1 / qs1
dqdt = rh * dqsdt ## from above, reset to linear method
q_LW_kernel = q_LW_kernel.values / dqdt
q_SW_kernel = q_SW_kernel.values / dqdt
q_LW_kernel_clearsky = q_LW_kernel_clearsky.values / dqdt
q_SW_kernel_clearsky = q_SW_kernel_clearsky.values / dqdt

## Convolve moisture kernel with change in moisture
dLW_q = (q_LW_kernel.values * dq * pdiff.values).sum(dim='lev_p')
dSW_q = (q_SW_kernel.values * dq * pdiff.values).sum(dim='lev_p')
dLW_q_cs = (q_LW_kernel_clearsky.values * dq * pdiff.values).sum(dim='lev_p')
dSW_q_cs = (q_SW_kernel_clearsky.values * dq * pdiff.values).sum(dim='lev_p')


### Change in Cloud Radiative Effect (CRE) 
d_sw = changefile3d.FSNT
d_sw_cs = changefile3d.FSNTC
d_lw = changefile3d.FLNT
d_lw_cs = changefile3d.FLNTC

d_cre_sw = d_sw_cs - d_sw
d_cre_lw = d_lw_cs - d_lw


## THIS WOULD NEED TO BE ADJUSTED FOR DIFFERENT FORCINGS
### Cloud masking of radiative forcing
ghgfile = './forcing/ghg.forcing.nc'
sw = xr.open_dataset(ghgfile).FSNT
sw_cs = xr.open_dataset(ghgfile).FSNTC
lw = xr.open_dataset(ghgfile).FLNT
lw_cs = xr.open_dataset(ghgfile).FLNTC
ghg_sw = sw_cs-sw
ghg_lw = lw_cs-lw

aerosolfile = './forcing/aerosol.forcing.nc';
sw = xr.open_dataset(aerosolfile).FSNT
sw_cs = xr.open_dataset(aerosolfile).FSNTC
lw = xr.open_dataset(aerosolfile).FLNT
lw_cs = xr.open_dataset(aerosolfile).FLNTC
aerosol_sw = sw_cs - sw
aerosol_lw = lw_cs - lw

## MAKE FORCING ZERO because there is no forcing in these experiments
cloud_masking_of_forcing_sw = 0 # aerosol_sw + ghg_sw
cloud_masking_of_forcing_lw = 0 # aerosol_lw + ghg_lw


### Cloud feedback
### CRE + cloud masking of radiative forcing + corrections for each feedback

dLW_cloud = -d_cre_lw + cloud_masking_of_forcing_lw + (
    dLW_q_cs - dLW_q.values) + (dLW_ta_cs - dLW_ta.values) + (dLW_ts_cs - dLW_ts)
dSW_cloud = -d_cre_sw + cloud_masking_of_forcing_sw + (
    dSW_q_cs - dSW_q.values) + (dSW_alb_cs-dSW_alb)

## Take global and annual averages
dLW_cloud_globalmean = (-dLW_cloud.mean(dim='time') * weight).sum()
dSW_cloud_globalmean = ( dSW_cloud.mean(dim='time') * weight).sum()

## Divide by global, annual mean temperature change to get W/m2/K
lw_cloud_feedback = dLW_cloud_globalmean / dts_globalmean
sw_cloud_feedback = dSW_cloud_globalmean / dts_globalmean

# print('LW Cloud feedback: ',
#       str(lw_cloud_feedback.values), ' W m^-2 K^-1')
# print('SW Cloud feedback: ',
#       str(sw_cloud_feedback.values), ' W m^-2 K^-1')


#### ALL FEEDBACKS ####
print('Planck feedback: ', 
      str(planck_feedback.values), ' W m^-2 K^-1')
print('Lapse rate feedback: ', 
      str(lapserate_feedback.values), ' W m^-2 K^-1')
print('Water vapor feedback: ', 
      str(q_feedback.values), ' W m^-2 K^-1')
print('Surf. albedo feedback: ', 
      str(alb_feedback.values), ' W m^-2 K^-1')
print('LW Cloud feedback: ',
      str(lw_cloud_feedback.values), ' W m^-2 K^-1')
print('SW Cloud feedback: ',
      str(sw_cloud_feedback.values), ' W m^-2 K^-1')

sumall = planck_feedback + lapserate_feedback + q_feedback + alb_feedback + (
    lw_cloud_feedback + sw_cloud_feedback)
print('\n\nSum all feedbacks: ',
      str(sumall.values), ' W m^-2 K^-1')

## extras
print('\n\nTemperature feedback: ',
      str(t_feedback.values), ' W m^-2 K^-1')
print('Planck+lapse rate components: ',
      str(total_t_feedback.values), ' W m^-2 K^-1')

print('\n\nlogQ WV feedback: ', str(logq_feedback.values), ' W m^-2 K^-1')
print('linQ WV feedback: ', str(q_feedback.values), ' W m^-2 K^-1')

allmaps=xr.merge([dts, 
             -(dLW_planck.sum(dim='plev') + dLW_ts).rename('planck'),
             -dLW_lapserate.sum(dim='plev').rename('lapserate'),
              (-dLW_q + dSW_q).rename('watervapor'),
              dSW_alb.rename('albedo'),
             -dLW_cloud.rename('cloudLW'),
              dSW_cloud.rename('cloudSW')
             ])
allmaps['weights'] = (['lat','lon'], weight)
allmaps = allmaps.set_coords('weights')
temp = allmaps.mean(dim='time')

temp_gmean = (temp * temp.weights).sum() 
temp_gmean = temp_gmean/temp_gmean.ts
temp_gmean['sum'] = temp_gmean.drop('ts').to_array().sum()
temp_gmean['GMST'] = (temp['ts'] * temp.weights).sum()  # preserve order!
allmaps['dRESTOM'] =  changefile.FSNT - changefile.FLNT # store RESTOM
allmaps['dRESTOMC'] =  changefile.FSNTC - changefile.FLNTC # store RESTOM

dRESTOM = (allmaps['dRESTOM'].mean(dim='time') * allmaps['weights'].values).sum()
temp_gmean['net'] = dRESTOM / temp_gmean['GMST']
temp_gmean['res'] = temp_gmean['net'] - temp_gmean['sum']
temp_gmean['LR+WV'] = temp_gmean['lapserate'] + temp_gmean['watervapor']


#####################################################
##################### SAVE ##########################
#####################################################
print('changecase', case)

finalresult = np.array([planck_feedback,
          lapserate_feedback,
          q_feedback,
          alb_feedback,
          lw_cloud_feedback,
          sw_cloud_feedback])

savepath = './archive/'
savename = savepath + case + '.csv'
savecdf  = savepath + case + '.nc' 

if (strat_mask == True):
    savename = savepath + case + '_stratmasked.csv'
    savecdf  = savepath + case + '_stratmasked.nc'

if (cs_linearity_test == False):
    np.savetxt(savename, finalresult, delimiter=",")
    allmaps.to_netcdf(savecdf)
    print('Saved', savename)

######################################################
############   Clear-sky linearity test   ############
######################################################

if (cs_linearity_test == True):

    ## compare this value with model FSNTC-FLNTC global and annual mean
    cs_kernel = (dLW_t_globalmean + dSW_alb_globalmean + dR_q_globalmean).values
    print('Kernel CS Flux [W/m2]:', cs_kernel)
    # print((dLW_lapserate_globalmean + dLW_planck_globalmean - dLW_t_globalmean).values)

    cs_true = (
        (changefile['FSNTC'] - changefile['FLNTC']).mean(
            dim='time') * weight).sum().values
    print('Model CS Flux [W/m2]:', cs_true)
    print('Error fraction:', cs_kernel / cs_true - 1)
    
    
    savename = savepath + case + '_CS.csv'
    savecdf  = savepath + case + '_CS.nc'
    if (strat_mask == True):
        savename = savepath + case + '_stratmasked_CS.csv'
        savecdf  = savepath + case + '_stratmasked_CS.nc' 
    np.savetxt(savename, finalresult, delimiter=",")
    allmaps.to_netcdf(savecdf)
    print('Saved', savename)


######################################################
############  Add saved version for plio  ############
######################################################

spath = '/glade/derecho/scratch/vcooper/projects/plio_pattern/data/kernel_out/'
cs_fname = ''
if (cs_linearity_test == True):
    temp_gmean['dLW_t'] = dLW_t_globalmean
    temp_gmean['dSW_alb'] = dSW_alb_globalmean
    temp_gmean['dR_q'] = dR_q_globalmean
    temp_gmean['CS_kernel'] = cs_kernel
    temp_gmean['CS_true']= cs_true
    temp_gmean['error_frac'] = cs_kernel / cs_true - 1
    cs_fname = 'CS_'

temp_gmean.to_netcdf(spath + cs_fname + 'kernGmean_' + case + '.nc')
allmaps.to_netcdf(spath + cs_fname + 'kernMaps_' + case + '.nc')

    

