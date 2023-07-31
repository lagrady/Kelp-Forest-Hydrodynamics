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

#=======================================================================================================================
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
    
    if saveFig:
        plt.savefig(str(saveFig))
#=======================================================================================================================
        
def temp_adcp_profile_plot(tempDS, adcpDS, velMax, velMin, binNums = [0,-1], tStart = None, tEnd = None,
                          rSamp = None, flagCutoff = None, flagPlot = False, cMap = 'viridis', supTitle = None, saveFig = None):

    binRange = adcpDS.BinDist.values[binNums[0]:binNums[1]]+adcpDS.attrs['Instrument Height(m)']
    
    if rSamp is not None:
        Temp = tempDS.Temperature.sel(time = slice(tStart,tEnd)).resample(time = rSamp).mean()
        East = adcpDS.East.isel(BinDist = np.arange(binNums[0],binNums[1],1)).sel(time = slice(str(tStart),str(tEnd))).resample(time = str(rSamp)).mean()
        North = adcpDS.North.isel(BinDist = np.arange(binNums[0],binNums[1],1)).sel(time = slice(str(tStart),str(tEnd))).resample(time = str(rSamp)).mean()
        Flag = adcpDS.Flag.isel(BinDist = np.arange(binNums[0],binNums[1],1)).sel(time = slice(str(tStart),str(tEnd))).resample(time = str(rSamp)).mean()
        Depth = (adcpDS.Depth + adcpDS.attrs['Instrument Height(m)']).sel(time = slice(str(tStart),str(tEnd))).resample(time = str(rSamp)).mean()

    else:
        Temp = tempDS.Temperature.sel(time = slice(tStart,tEnd))
        East = adcpDS.East.isel(BinDist = np.arange(binNums[0],binNums[1],1)).sel(time = slice(str(tStart),str(tEnd)))
        North = adcpDS.North.isel(BinDist = np.arange(binNums[0],binNums[1],1)).sel(time = slice(str(tStart),str(tEnd)))
        Flag = adcpDS.Flag.isel(BinDist = np.arange(binNums[0],binNums[1],1)).sel(time = slice(str(tStart),str(tEnd)))
        Depth = (adcpDS.Depth + adcpDS.attrs['Instrument Height(m)']).sel(time = slice(str(tStart),str(tEnd)))
    
    if flagCutoff is not None:
        East = East.where(Flag < flagCutoff)
        North = North.where(Flag < flagCutoff)
    
    #East = East.dropna(dim = 'time', how = 'all')
    #North = North.dropna(dim = 'time', how = 'all')
    #Depth = Depth.dropna(dim = 'time', how = 'all')
    #Flag = Flag.dropna(dim = 'time', how = 'all')

    if flagPlot == True:                                                                                         
        fig, axs = plt.subplots(4,1,constrained_layout=True, figsize = (15,12))
        cmapFlag = ListedColormap(['g', 'y', 'r'])
        normFlag = BoundaryNorm([1, 2, 3, 4], cmapFlag.N)
        flagPlot = axs[3].pcolormesh(Flag.time, binRange, Flag, cmap =  cmapFlag, norm = normFlag)
        axs[3].plot(Depth.time, Depth, '-k', label = 'Sea surface height')
        axs[3].set_title('Data QC Flag (1-2 = Good, 2-3 = Suspect, 3-4 = Fail)', size = 14)
        axs[3].set_ylabel('M.A.B [m]', size = 12)
        axs[3].set_xlabel('Date', size = 12)
        axs[3].tick_params(axis = 'x', labelrotation = 15)
        axs[3].legend(loc = 'upper left', fontsize=10)
        fig.colorbar(flagPlot, ax=axs[3], location='right', cmap = cmapFlag, ticks = [1,2,3,4]).set_label(label='Flag',size=12)
    else:
        fig, axs = plt.subplots(3,1,constrained_layout=True, figsize = (15,12))

    axs[0].set_title('Temperature', size = 14)
    axs[0].set_prop_cycle(c = ['red','darkorange','yellow','green','blue','violet','black'])
    axs[0].plot(Temp.time, Temp.T, lw = 1)
    axs[0].set_ylabel("Temperature (Celsius)", fontsize=12)
    axs[0].tick_params(axis = 'x', labelrotation = 15)
    axs[0].margins(x=.00)
    axs[0].legend(['2m', '4m','6m','8m','9.1m', '9.4m', '9.7m'], fontsize=10)

    northVel = axs[1].pcolormesh(North.time, binRange, North, vmin = velMin, vmax = velMax, cmap = str(cMap))
    axs[1].plot(Depth.time, Depth, '-k', label = 'Sea surface height')
    axs[1].set_title('Northern Velocity', size = 14)
    axs[1].set_ylabel('M.A.B [m]', size = 12)
    axs[1].tick_params(axis = 'x', labelrotation = 15)
    axs[1].legend(loc = 'upper left', fontsize=10)

    eastVel = axs[2].pcolormesh(East.time, binRange, East, vmin = velMin, vmax = velMax, cmap = str(cMap))
    axs[2].plot(Depth.time, Depth, '-k', label = 'Sea surface height')
    axs[2].set_title('Eastern Velocity', size = 14)
    axs[2].set_ylabel('M.A.B [m]', size = 12)
    axs[2].set_xlabel('Date', size = 12)
    axs[2].tick_params(axis = 'x', labelrotation = 15)
    axs[2].legend(loc = 'upper left', fontsize=10)

    fig.colorbar(northVel, ax=axs[1], location='right').set_label(label='Velocity [m/s]',size=12)
    fig.colorbar(eastVel, ax=axs[2], location='right').set_label(label='Velocity [m/s]',size=12)
    
    if supTitle is not None:
        fig.suptitle(str(supTitle), x = .475, y = 1.05, size = 20)
    
    if saveFig:
        plt.savefig(str(saveFig))
    
