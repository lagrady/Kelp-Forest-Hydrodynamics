### Essentials
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta


# Calculate effective degrees of freedom of a time series
def edof(X,Y,lag,dt=None):
    lag = np.arange(0,lag,1)

    autoX = np.zeros((len(lag)))
    autoY = np.zeros((len(lag)))

    crossXY = np.zeros((len(lag)))
    crossYX = np.zeros((len(lag)))

    for i in lag:
        if i==0:
            autoX[i]=np.corrcoef(X,X)[0,1]
            autoY[i]=np.corrcoef(Y,Y)[0,1]
            crossXY[i] = np.corrcoef(X,Y)[0,1]
            crossYX[i] = np.corrcoef(Y,X)[0,1]
        else:
            autoX[i] = np.corrcoef(X[i:],X[:-i])[0,1]
            autoY[i] = np.corrcoef(Y[i:],Y[:-i])[0,1]
            crossXY[i] = np.corrcoef(X[i:],Y[:-i])[0,1]
            crossYX[i] = np.corrcoef(Y[i:],X[:-i])[0,1]

    corr = np.sum((autoX*autoY)+(crossXY*crossYX))
    if dt:
        dt_scale = corr * dt
    edof = np.round((len(X)/corr)-2,0)
    r = np.corrcoef(X, Y)[0,1]
    print('r='+str(r)+' p='+str(pStat.rsig(r,edof))+' edof='+str(edof))
    return autoX, autoY, dt_scale

# Calculate latitude and longitude
def find_coords(lat1, lon1, heading, dist):
    R = 6378100 # Radius of the Earth in meters
    Ad = dist/R # Angular distance of point b
    lat1 = lat1 * (np.pi/180)
    lon1 = lon1 * (np.pi/180)
    theta = heading * (np.pi/180) # The bearing towards point b in radians
    
    #lat2 = np.arcsin((np.sin(lat1) * np.cos(Ad)) + (np.cos(lat1)*np.sin(Ad)*np.cos(theta)))
    lat2 = np.arcsin(np.sin(lat1) * np.cos(Ad) + np.cos(lat1) * np.sin(Ad) * np.cos(theta)) * (180/np.pi)
    lon2 = (lon1 + np.arctan2(np.sin(theta) * np.sin(Ad) * np.cos(lat1), np.cos(Ad) - np.sin(lat1) * np.sin(lat2))) * (180/np.pi)
    return lat2, lon2