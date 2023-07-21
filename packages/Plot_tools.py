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
def overview_plot(tempDS, adcpDS, advDS, timeStart = None, timeEnd = None, saveFig = False, figName = None):
    
    if (timeStart & timeEnd):
        tempDS = tempDS.sel(time=slice(str(timeStart), str(timeEnd)))
        adcpDS = adcpDS.sel(time=slice(str(timeStart), str(timeEnd)))
        advDS = advDS.sel(time=slice(str(timeStart), str(timeEnd)))
        
    adcpDS = adcpDS.resample(
    
    temp20mRoll = tempDS.rolling(time=20).mean()
    adcp20mRoll = adcpDS.isel(BinDist=1).rolling(time=20).mean()
        
    plt.figure(figsize = (20,16))
    #=================================================================================================
    # Temperature
    plt.subplot(411)
    plt.gca().set_prop_cycle(c = ['red','darkorange','blue','green','cyan','yellow','black'])
    plt.plot(tempDS.time, tempDS.Temperature.T, lw = 1)
    plt.title('Temperature within SWC Kelp Forest', fontsize=14)
    plt.ylabel("Temperature (Celsius)", fontsize=14)
    plt.margins(x=.01)
    plt.legend(['2m', '4m','6m','8m','9.1m', '9.4m', '9.7m'], loc = 'upper left', fontsize=14)
    #=================================================================================================
    # Eastern Velocity
    plt.subplot(412)
    plt.plot(adcp_dep1.time, adcp_dep1.East.isel(BinDist=1), '.r', label = 'ADCP-Eastern')
    plt.plot(adcp_10mroll_dep1.time, adcp_10mroll_dep1.East,'-k', label = 'ADCP-Eastern (10-min rolling)')
    plt.plot(adv_10m_dep1.time, adv_10m_dep1.East, '.b', label = 'ADV-Eastern (10-min average)')
    plt.ylim(-.1,.1)
    plt.legend(loc = 'upper right')
    plt.axhline(y=0, c='black', lw=2)
    plt.margins(x=.01)
    plt.ylabel('Velocity (m/s)')
    plt.title('Eastern Velocity (1m Above Seafloor)')
    #=================================================================================================
    # Northern Velocity
    plt.subplot(413)
    plt.plot(adcp_dep1.time, adcp_dep1.North.isel(BinDist=1), '.r', label = 'ADCP-Northern')
    plt.plot(adcp_10mroll_dep1.time, adcp_10mroll_dep1.North,'-k', label = 'ADCP-Northern (10-min rolling)')
    plt.plot(adv_10m_dep1.time, adv_10m_dep1.North, '.b', label = 'ADV-Northern (10-min average)')
    plt.ylim(-.1,.1)
    plt.legend(loc = 'upper right')
    plt.axhline(y=0, c='black', lw=2)
    plt.margins(x=.01)
    plt.ylabel('Velocity (m/s)')
    plt.title('Northern Velocity (1m Above Seafloor)')
    #=================================================================================================
    # Velocity direction
    plt.subplot(414)
    plt.plot(adcp_10mroll_dep1.time, adcp_10mroll_dep1.Direction,'-k', label = 'ADCP-Velocity Direction (10-min rolling)')
    plt.plot(adv_10m_dep1.time, adv_10m_dep1.CDIR, '.b', label = 'ADV-Velocity Direction (1-min average)')
    plt.legend(loc = 'upper right')
    plt.margins(x=.01)
    plt.ylabel('Direction (Degrees)')
    plt.xlabel('Date')
    plt.title('Velocity direction (1m Above Seafloor)')