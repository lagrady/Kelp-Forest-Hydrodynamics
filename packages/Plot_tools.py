### Essentials
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from datetime import timedelta

#Oceanography tools
import gsw # https://teos-10.github.io/GSW-Python/gsw_flat.html
from iapws import iapws95
from physoce import tseries as ts 

#Scipy
from scipy.signal import welch 
from scipy.stats import chi2 
from scipy.special import gamma
from scipy import stats
from scipy import signal
from scipy.signal import periodogram
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from scipy import integrate
from scipy.stats.distributions import  t

#Custom function packages
import vector_tools as vt

#============================================================================================================
def overview_plot(tempData, adcpData, advData = None, startTime = None, endTime = None, resample = None, saveFig = None):
    
    tempDS = tempData.copy(deep=True)
    adcpDS = adcpData.copy(deep=True)
    if advData:
        vecDS = advData.copy(deep=True)
    
    if startTime:
        print('Subsampling')
        tempDS = tempDS.sel(time = slice(str(startTime), str(endTime)))
        adcpDS = adcpDS.sel(time = slice(str(startTime), str(endTime)))
        if advData:
            vecDS = vecDS.sel(time = slice(str(startTime), str(endTime)))

    tempDS = tempDS.Temperature

    adcpEast = adcpDS.EastDA.dropna(dim = 'time')
    adcpNorth = adcpDS.NorthDA.dropna(dim = 'time')
    adcpCDIR = vt.vec_angle(adcpEast, adcpNorth)
    if advData:
        gb = np.unique(vecDS.burst.where((vecDS.dEast < .25) & (vecDS.dNorth < .25) & (vecDS.burst.isin(vecDS.BurstNum)), drop=True))
        advEast = vecDS.East.where(vecDS.BurstNum.isin(gb)).dropna(dim = 'time')
        advNorth = vecDS.North.where(vecDS.BurstNum.isin(gb)).dropna(dim = 'time')
        advCDIR = vt.vec_angle(advEast, advNorth)

    if resample:
        print('Resampling')
        tempDS = tempDS.resample(time = str(resample)).mean()

        adcpEast = adcpEast.resample(time = str(resample)).mean().dropna(dim = 'time')
        adcpNorth = adcpNorth.resample(time = str(resample)).mean().dropna(dim = 'time')
        adcpCDIR = vt.vec_angle(adcpEast, adcpNorth)
        if advData:
            gb = np.unique(vecDS.burst.where((vecDS.dEast < .25) & (vecDS.dNorth < .25) & (vecDS.burst.isin(vecDS.BurstNum)), drop=True))
            advEast = advEast.resample(time = str(resample)).mean().dropna(dim = 'time')
            advNorth = advNorth.resample(time = str(resample)).mean().dropna(dim = 'time')
            advCDIR = vt.vec_angle(advEast, advNorth)
    

    print('Plotting')
    # TEMPERATURE
    plt.figure(figsize = (20,25))

    plt.subplot(411)
    plt.gca().set_prop_cycle(c = ['red','darkorange','yellow','green','blue','violet','black'])
    plt.plot(tempDS.time, tempDS.T, lw = 1)
    if resample:
        plt.title('Temperature within SWC Kelp Forest Mooring (' + str(resample) + ' average)', fontsize=14)
    else:
        plt.title('Temperature within SWC Kelp Forest Mooring', fontsize=14)
    plt.ylabel("Temperature (Celsius)", fontsize=12)
    plt.xticks(rotation = 15)
    plt.margins(x=.01)
    plt.legend(['2m', '4m','6m','8m','9.1m', '9.4m', '9.7m'], loc = 'upper left', fontsize=12)
    #=================================================================================================
    # Eastern Velocity
    plt.subplot(412)
    plt.plot(adcpEast.time, adcpEast,'-k', label = 'Depth average (ADCP)')
    if advData:
        plt.plot(advEast.time, advEast, '.-b', label = '1m above seafloor (ADV)')
    plt.axhline(y=0, c='black', lw=2)
    if resample:
        plt.title('Eastern velocity (' + str(resample) + ' average)', fontsize=14)
    else:
        plt.title('Eastern velocity', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.xticks(rotation = 15)
    plt.ylim(-.03,.03)
    plt.margins(x=.01)
    plt.legend(loc = 'upper right', fontsize=12)

    #=================================================================================================
    # Northern Velocity 
    plt.subplot(413)
    plt.plot(adcpNorth.time, adcpNorth,'-k', label = 'Depth average (ADCP)')
    if advData:
        plt.plot(advNorth.time, advNorth, '.-b', label = '1m above seafloor (ADV)')
    plt.axhline(y=0, c='black', lw=2)
    if resample:
        plt.title('Northern velocity (' + str(resample) + ' average)', fontsize=14)
    else:
        plt.title('Northern velocity', fontsize=14)
    plt.ylabel('Velocity (m/s)', fontsize=12)
    plt.xticks(rotation = 15)
    plt.ylim(-.03,.03)
    plt.margins(x=.01)

    #=================================================================================================
    # Current Direction 
    #plt.subplot(414)
    plt.plot(adcpCDIR.time, adcpCDIR,'-k', label = 'Depth average (ADCP)')
    if advData:
        plt.plot(advCDIR.time, advCDIR, '.-b', label = '1m above seafloor (ADV)')
    if resample:
        plt.title('Current direction (' + str(resample) + ' average)', fontsize=14)
    else:
        plt.title('Current direction', fontsize=14)
    plt.ylabel('Direction (Degrees)', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation = 15)
    plt.margins(x=.01)
    
    if saveFig:
        plt.savefig(str(saveFig))