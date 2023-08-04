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
def overview_plot(tempData, adcpData = None, advData = None, startTime = None, endTime = None, resample = None, supTitle = None, saveFig = None):
    
    fig, axs = plt.subplots(4,1,constrained_layout=True, figsize = (12,12))
    if supTitle:
        fig.suptitle(str(supTitle), x = .55, size = 20)
    
    # Temperature
    print('Plotting temperature')
    tempDS = tempData.copy(deep=True)
    Temp = tempDS.Temperature
    if startTime:
        Temp = Temp.sel(time = slice(str(startTime), str(endTime)))
    if resample:
        Temp = Temp.resample(time = str(resample)).mean()
    axs[0].set_title('Temperature', size=14) 
    axs[0].set_prop_cycle(c = ['red','darkorange','yellow','green','blue','violet','black'])
    axs[0].plot(Temp.time, Temp.T, lw = 1)
    axs[0].set_ylabel(r'Temperature [$^\circ$C]', fontsize=12)
    axs[0].tick_params(axis = 'x', labelrotation = 15)
    axs[0].margins(x=.01)
    axs[0].legend(['2m', '4m','6m','8m','9.1m', '9.4m', '9.7m'], loc = 'upper left', fontsize=8)
        
        
    if adcpData:
        adcpDS = adcpData.copy(deep=True)
        adcpEast = adcpDS.EastDA
        adcpNorth = adcpDS.NorthDA
        if startTime:
            adcpEast = adcpEast.sel(time = slice(str(startTime), str(endTime))).dropna(dim = 'time')
            adcpNorth = adcpNorth.sel(time = slice(str(startTime), str(endTime))).dropna(dim = 'time')
        if resample:
            adcpEast = adcpEast.resample(time = str(resample)).mean().dropna(dim = 'time')
            adcpNorth = adcpNorth.resample(time = str(resample)).mean().dropna(dim = 'time')
    if advData:
        vecDS = advData.copy(deep=True)
        if startTime:
            vecDS = vecDS.sel(time = slice(str(startTime), str(endTime)))
        gb = np.unique(vecDS.burst.where((vecDS.dEast < .25) & (vecDS.dNorth < .25) & (vecDS.burst.isin(vecDS.BurstNum)), drop=True))
        advEast = vecDS.East.where(vecDS.BurstNum.isin(gb)).dropna(dim = 'time')
        advNorth = vecDS.North.where(vecDS.BurstNum.isin(gb)).dropna(dim = 'time')
        if resample:
            advEast = advEast.resample(time = str(resample)).mean().dropna(dim = 'time')
            advNorth = advNorth.resample(time = str(resample)).mean().dropna(dim = 'time')
            
    # Northern Velocity 
    axs[1].set_title('Northern Velocity', size=14)
    if adcpData:
        axs[1].plot(adcpNorth.time, adcpNorth, '-r', lw = 2, label = 'Depth-averaged (ADCP)')
    if advData:
        axs[1].plot(advNorth.time, advNorth, '.-b', lw = 1, ms = 3, label = '1m above bottom (ADV)')
    axs[1].axhline(y=0, c = 'black', ls = '--')
    axs[1].set_ylabel(r"Velocity [$\frac{m}{s}$]", fontsize=12)
    axs[1].tick_params(axis = 'x', labelrotation = 15)
    axs[1].margins(x=.01)
    axs[1].legend(loc = 'upper left', fontsize=8)


    # Eastern Velocity    
    axs[2].set_title('Eastern Velocity', size=14)
    if adcpData:
        axs[2].plot(adcpEast.time, adcpEast, '-r', lw = 2, label = 'Depth-averaged (ADCP)')
    if advData:
        axs[2].plot(advEast.time, advEast, '.-b', lw = 1, ms = 3, label = '1m above bottom (ADV)')
    axs[2].axhline(y=0, c = 'black', ls = '--')
    axs[2].set_ylabel(r"Velocity [$\frac{m}{s}$]", fontsize=12)
    axs[2].tick_params(axis = 'x', labelrotation = 15)
    axs[2].margins(x=.01)
    axs[2].legend(loc = 'upper left', fontsize=8)
        
    # Velocity Direction
    axs[3].set_title('Current Direction', size=14)
    if adcpData:
        adcpCDIR = vt.vec_angle(adcpEast, adcpNorth)
        axs[3].plot(adcpCDIR.time, adcpCDIR, '-r', lw = 2, label = 'ADCP')
    if advData:
        advCDIR = vt.vec_angle(advEast, advNorth)
        axs[3].plot(advCDIR.time, advCDIR, '.-b', lw = 1, ms = 3, label = 'ADV')
    axs[3].set_ylim(-5,370)
    axs[3].set_ylabel("Current Heading [Degrees]", fontsize=12)
    axs[3].set_xlabel("Date", fontsize = 12)
    axs[3].tick_params(axis = 'x', labelrotation = 15)
    axs[3].margins(x=.01, y = .1)
    axs[3].legend(loc = 'upper left', fontsize=8)  
    
    if saveFig:
        plt.savefig(str(saveFig))
#=======================================================================================================================
        
def adcp_profile_plot(tempDS, adcpDS, velMax, velMin, binNums = [0,-1], tStart = None, tEnd = None,
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
    axs[0].set_ylabel("Temperature [$^\circ$C]", fontsize=12)
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

    fig.colorbar(northVel, ax=axs[1], location='right').set_label(label=r"Velocity [$\frac{m}{s}$]",size=12,rotation = -90, va = 'bottom')
    fig.colorbar(eastVel, ax=axs[2], location='right').set_label(label=r"Velocity [$\frac{m}{s}$]",size=12,rotation = -90, va = 'bottom')
    
    if supTitle is not None:
        fig.suptitle(str(supTitle), x = .475, size = 20)
    
    if saveFig:
        plt.savefig(str(saveFig))
    
