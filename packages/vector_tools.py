#Essentials
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from datetime import timedelta
import re

#Oceanography tools
import gsw # https://teos-10.github.io/GSW-Python/gsw_flat.html
from iapws import iapws95 #https://iapws.readthedocs.io/en/latest/iapws.iapws95.html
from physoce import tseries as ts #https://github.com/physoce/physoce-py

#Scipy
from scipy import integrate
from scipy.integrate import quad_vec
from scipy import stats
from scipy.stats import chi2
from scipy.stats.distributions import t
from scipy import signal
from scipy.signal import welch 
from scipy.signal import periodogram
from scipy.signal import argrelextrema
from scipy.special import gamma
from scipy.optimize import curve_fit



#===============================================================================================================================
#=============================================== DATA PROCESSING FUNCTIONS =====================================================
#===============================================================================================================================

def fastResample(ds, timeString):
    # Following sequence of code originally found on https://stackoverflow.com/questions/64282393/how-can-i-speed-up-xarray-resample-much-slower-than-pandas-resample
    df_h = ds.to_dataframe().resample(str(timeString)).mean().dropna()  # what we want (quickly), but in Pandas form
    vals = [xr.DataArray(ds=df_h[c], dims=['time'], coords={'time':df_h.index}, attrs=ds[c].attrs) for c in df_h.columns]
    dsResample = xr.Dataset(dict(zip(df_h.columns,vals)), attrs=ds.attrs)
    
    return dsResample

#===============================================================================================================================
def vec_angle(x,y):
    '''
    A function used to find the direction of a vector from 0-360 degrees given the x and y component. Used to find the current speed direction in the vector_to_ds function.
    
    INPUTS:
    x,y: the x and y components of a single vector
    
    OUTPUTS:
    A numpy.float value from 0-360, representative of the vector direction.
    
    EXAMPLE:
    import numpy as np
    import xarray as xr
    import vector_tools as vt
    
    velocity_north = 3
    velocity_east = -3
    vec_angle(velocity_east, velocity_north)
    
    '''
    cdir = xr.zeros_like(x)
    
    cdir = cdir + xr.where(((x>0.) & (y>0.)), np.arctan(y/x) * (180/np.pi), 0)
    cdir = cdir + xr.where(((x==0.) & (y>0.)), 0. * (180/np.pi), 0)
    cdir = cdir + xr.where(((x>0.) & (y<0.)), 180 - np.abs((np.arctan(y/x) * (180/np.pi))), 0)
    cdir = cdir + xr.where(((x>0.) & (y==0.)), 90., 0)
    cdir = cdir + xr.where(((x<0.) & (y<0.)), 180 + (np.arctan(y/x) * (180/np.pi)), 0)
    cdir = cdir + xr.where(((x==0.) & (y<0.)), 180., 0)
    cdir = cdir + xr.where(((x<0.) & (y>0.)), 360 - np.abs((np.arctan(y/x) * (180/np.pi))), 0)
    cdir = cdir + xr.where(((x<0.) & (y==0.)), 270., 0)
    cdir = cdir + xr.where(((x==0.) & (y==0.)), 0, 0)
    
    return cdir

#===============================================================================================================================
def vector_to_ds(datfile, vhdfile, senfile, fs):
    '''
    Take the .dat and .vhd files from the vector and generate an xarray dataset with all variables. Also conducts quality control tests as recommended by the 
    Nortek N3015-030 Comprehensive Manual for Velocimeters
    
    Additional information regarding suggested test parameters and theory behind tests can be located online at:
    https://support.nortekgroup.com/hc/en-us/articles/360029839351-The-Comprehensive-Manual-Velocimeters
    
    INPUTS:
    datfile: the .dat file imported directly from the Vector
    vhdfile: the .vhd file imported directly from the Vector
    fs: the sample frequency of the Vector during the deployment
    
    
    OUTPUTS:
    xarray dataset with "Flag" data array
    
    EXAMPLE:
    import numpy as np
    import xarray as xr
    import vector_tools as vt
    
    datfile = 'vector.dat'
    vhdfile = 'vector.vhd'
    senfile = 'vector.sen'
    fs = 32 # Vector sampled at 32Hz
    
    ds = vt.vector_to_ds(datfile, vhdfile, senfile fs)
    '''
    # Create column names for pandas dataframe
    # 'dat_cols' pertains to the default .DAT file from the vector, 'sen_cols' pertains to the default .SEN file
    print('Importing data')
    
    dat_cols = ["Burst_counter", "Ensemble_counter", "Velocity_East", "Velocity_North", "Velocity_Up", "Amplitude_B1", 
                "Amplitude_B2", "Amplitude_B3", "SNR_B1", "SNR_B2", "SNR_B3", "Correlation_B1", "Correlation_B2", 
                "Correlation_B3", "Pressure", "AnalogInput1", "AnalogInput2", "Checksum"]
    vhd_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Burst_counter", "Num_samples", "NA1", "NA2", "NA3", "NC1", "NC2", "NC3", "Dist1_st", "Dist2_st", "Dist3_st", "Distavg_st",
               "distvol_st", "Dist1_end", "Dist2_end", "Dist3_end", "Distavg_end", "distvol_end"] 
    sen_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Error_code", "Status_code", "Battery_voltage", "Soundspeed", "Heading", "Pitch", "Roll", "Temperature", 
                "Analog_input", "Checksum"]

    dat = pd.read_csv(datfile, delimiter='\s+', names = dat_cols)
    vhd = pd.read_csv(vhdfile, delimiter='\s+', names = vhd_cols)
    sen = pd.read_csv(senfile, delimiter='\s+', names = sen_cols)
    
    if fs == 32:
        t_step = '31.25L'
    elif fs == 16:
        t_step = '62.5L'
    elif fs == 8:
        t_step = '125L'
    elif fs == 4:
        t_step = '250L'
    elif fs == 2:
        t_step = '500L'
    elif fs == 1:
        t_step = '1000L'
    
    print('Creating timelines')
    
    samples = vhd.Num_samples[1]
    time_dat = np.empty(shape=(len(vhd)*samples), dtype='datetime64[ns]')
    
    for i in range(0, len(vhd)-1, 1):
        time_dat[i*samples:(i+1)*samples] = pd.date_range(start=(str(vhd.iloc[i,0])+'/'+ str(vhd.iloc[i,1])+'/'+ str(vhd.iloc[i,2])+' '+ str(vhd.iloc[i,3])+':'+ 
                                                     str(vhd.iloc[i,4])+':'+ str(vhd.iloc[i,5])), periods = samples, freq = t_step)
    time_dat = time_dat[:len(dat)]   
    #dat.insert(0, 'datetime', vhd_dates)
    #dat.datetime = pd.to_datetime(dat.datetime)
    time_dat = pd.to_datetime(time_dat, utc=True)
    
    time_start = vhd['Month'].map(str)+'/'+vhd['Day'].map(str)+'/'+vhd['Year'].map(str)+' '+vhd['Hour'].map(str)+':'+vhd['Minute'].map(str)+':'+vhd['Second'].map(str)
    time_start = pd.to_datetime(time_start,utc=True)
    
    time_sen = sen['Month'].map(str)+'/'+sen['Day'].map(str)+'/'+sen['Year'].map(str)+' '+sen['Hour'].map(str)+':'+sen['Minute'].map(str)+':'+sen['Second'].map(str)
    time_sen = pd.to_datetime(time_sen,utc=True)
    
    print('Creating xarray dataset')
    # create coords

    # put data into a dataset
    ds = xr.Dataset(
        data_vars=dict(
            BurstCounter = (["time_start"], vhd['Burst_counter']),
            NoVelSamples = (["time_start"], vhd['Num_samples']),
            ErrorCode = (["time_sen"], sen['Error_code']),
            StatusCode = (["time_sen"], sen['Status_code']),
            BatVolt = (["time_sen"], sen['Battery_voltage']),
            SoundSpeed = (["time_sen"], sen['Soundspeed']),
            Heading = (["time_sen"], sen['Heading']),
            Pitch = (["time_sen"], sen['Pitch']),
            Roll = (["time_sen"], sen['Roll']),
            Temperature = (["time_sen"], sen['Temperature']),
            ChecksumSen = (["time_sen"], sen['Checksum']),
            BurstNum = (["time"], dat['Burst_counter']),
            East = (["time"], dat['Velocity_East']),
            North = (["time"], dat['Velocity_North']),
            Up = (["time"], dat['Velocity_Up']),
            CSPD = (["time"], np.sqrt(((dat['Velocity_East'])**2) + ((dat['Velocity_North'])**2))),
            CDIR = (["time"], vec_angle(dat['Velocity_East'].to_xarray(), dat['Velocity_North'].to_xarray()).data),
            Amp1 = (["time"], dat['Amplitude_B1']),
            Amp2 = (["time"], dat['Amplitude_B2']),
            Amp3 = (["time"], dat['Amplitude_B3']),
            Snr1 = (["time"], dat['SNR_B1']),
            Snr2 = (["time"], dat['SNR_B2']),
            Snr3 = (["time"], dat['SNR_B3']),
            Corr1 = (["time"], dat['Correlation_B1']),
            Corr2 = (["time"], dat['Correlation_B2']),
            Corr3 = (["time"], dat['Correlation_B3']),
            Pressure = (["time"], dat['Pressure']),
            ChecksumDat = (["time"], dat['Checksum'])    
        ),
        coords=dict(
            time=(["time"], time_dat),
            time_sen=(["time_sen"], time_sen),
            time_start=(["time_start"], time_start)
        ),
    )
    ds['time'] = pd.to_datetime(ds.time.values)
    print('Assigning dataset attributes')
    ds['BatVolt'].attrs['units'] = 'Volts'
    ds['BatVolt'].attrs['description'] = 'Voltage of the instrument measure at 1Hz during sampling period.'
    ds['SoundSpeed'].attrs['units'] = 'm/s'
    ds['SoundSpeed'].attrs['description'] = 'Speed of sound recorded by the instrument based on recorded temperature and set salinity.'
    ds['Heading'].attrs['units'] = 'Degrees'
    ds['Heading'].attrs['units'] = 'Degrees'
    ds['Pitch'].attrs['units'] = 'Degrees'
    ds['Roll'].attrs['units'] = 'Degrees'
    ds['Temperature'].attrs['units'] = 'Celsius'
    ds['ChecksumSen'].attrs['description'] = 'A binary internal test conducted by the instrument which indicates successful or failed measurement (1 = failure). This test is conducted at 1Hz during sampling period.'
    ds['BurstCounter'].attrs['description'] = 'Sequence of burst since start of recording'
    ds['BurstNum'].attrs['description'] = 'Burst counter with a high frequency timestamp that matches the vector measurements'
    ds['East'].attrs['units'] = 'm/s'
    ds['North'].attrs['units'] = 'm/s'
    ds['Up'].attrs['units'] = 'm/s'
    ds['CSPD'].attrs['units'] = 'm/s'
    ds['CSPD'].attrs['description'] = 'Horizontal current speed, the magnitude of the Eastern and Northern velocity vectors.'
    ds['CDIR'].attrs['units'] = 'Degrees'
    ds['CDIR'].attrs['description'] = 'Direction of the horizontal current speed derived from the vec_angle function in vector_tools.py.'
    ds['Amp1'].attrs['units'] = 'Counts'
    ds['Amp1'].attrs['description'] = 'Amplitude of beam 1.'
    ds['Amp2'].attrs['units'] = 'Counts'
    ds['Amp2'].attrs['description'] = 'Amplitude of beam 2.'
    ds['Amp3'].attrs['units'] = 'Counts'
    ds['Amp3'].attrs['description'] = 'Amplitude of beam 3.'
    ds['Snr1'].attrs['units'] = 'dB'
    ds['Snr1'].attrs['description'] = 'Signal to noise ratio of beam 1.'
    ds['Snr2'].attrs['units'] = 'dB'
    ds['Snr2'].attrs['description'] = 'Signal to noise ratio of beam 2.'
    ds['Snr3'].attrs['units'] = 'dB'
    ds['Snr3'].attrs['description'] = 'Signal to noise ratio of beam 3.'
    ds['Corr1'].attrs['units'] = '%'
    ds['Corr1'].attrs['description'] = 'Beam correlation measurment from beam 1.'
    ds['Corr2'].attrs['units'] = '%'
    ds['Corr2'].attrs['description'] = 'Beam correlation measurment from beam 2.'
    ds['Corr3'].attrs['units'] = '%'
    ds['Corr3'].attrs['description'] = 'Beam correlation measurment from beam 3.'
    ds['Pressure'].attrs['units'] = 'dBar'
    ds['Pressure'].attrs['description'] = 'Ambient pressure recorded by the instrument at the same frequency as the velocity data.'
    ds['ChecksumDat'].attrs['description'] = 'A binary internal test conducted by the instrument which indicates successful or \
    failed measurement (1 = failure). This test is conducted at the same frequency as the velocity data.'
    
    return ds
#===============================================================================================================================
def vectorFlag(ds):
    
    print('Flagging data')
    dat_flag = xr.zeros_like(ds.East) # Same shape as .dat data arrays
    datFlagQartod = xr.zeros_like(dat_flag)
    
    burst_diff = np.diff(ds.BurstNum, axis = 0, prepend = 0)
    burst_diff[0] = 0
    
    min_depth = int(np.mean(ds.Pressure) - (np.std(ds.Pressure)*1.5))
                       
    testCounter = 0                   
    
    # Checksum tests 
    dat_flag = dat_flag + xr.where(ds.ChecksumDat == 0, 0, 9)
    testCounter = testCounter + 1
                       
    # Pressure test
    dat_flag = dat_flag + xr.where(ds.Pressure >= min_depth, 0, 9)
    testCounter = testCounter + 1
                       
    # SNR test
    dat_flag = dat_flag + xr.where((ds.Snr1 < 10), 9, 0) # Full failure 
    dat_flag = dat_flag + xr.where((ds.Snr2 < 10), 9, 0) 
    dat_flag = dat_flag + xr.where((ds.Snr3 < 10), 9, 0) 
    testCounter = testCounter + 3
                       
    # Beam correlation test          
    dat_flag = dat_flag + xr.where((ds.Corr1 >= 70), 0, 0) # Full pass condition
    dat_flag = dat_flag + xr.where((ds.Corr1 < 70) & (ds.Corr1 > 50), 3, 0) # Not ideal but acceptable
    dat_flag = dat_flag + xr.where((ds.Corr1 <= 50), 9, 0) # Full failure 
    
    dat_flag = dat_flag + xr.where((ds.Corr2 >= 70), 0, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr2 < 70) & (ds.Corr2 > 50), 3, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr2 <= 50), 9, 0) 
    
    dat_flag = dat_flag + xr.where((ds.Corr3 >= 70), 0, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr3 < 70) & (ds.Corr3 > 50), 3, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr3 <= 50), 9, 0) 
    testCounter = testCounter + 3
    
    # Horizontal velocity test
    # For East-West
    dat_flag = dat_flag + xr.where(np.abs(ds.East) >= 3, 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(ds.East) < 3) & (np.abs(ds.East) >= 1), 3, 0)
    dat_flag = dat_flag + xr.where(np.abs(ds.East) < 1, 0, 0)
    testCounter = testCounter + 1                 
    
    # For North-South
    dat_flag = dat_flag + xr.where(np.abs(ds.North) >= 3, 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(ds.North) < 3) & (np.abs(ds.North) >= 1), 3, 0)
    dat_flag = dat_flag + xr.where(np.abs(ds.North) < 1, 0, 0)
    testCounter = testCounter + 1 
          
    # Vertical velocity test
    dat_flag = dat_flag + xr.where(np.abs(ds.Up) >= 2, 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(ds.Up) < 2) & (np.abs(ds.Up) >= 1), 3, 0)
    dat_flag = dat_flag + xr.where(np.abs(ds.Up) < 1, 0, 0)
    testCounter = testCounter + 1 
      
    # u, v, w rate of change test
    # For East-west (u)
    du = np.diff(ds.East, axis = 0, prepend = 0) 
    du[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(du) >= 2) & (burst_diff==0), 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(du) < 2) & (np.abs(du) >= 1) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(du) < 1) & (np.abs(du) >= .25) & (burst_diff==0), 1, 0) 
    testCounter = testCounter + 1 
    
    # For North-South (v)
    dv = np.diff(ds.North, axis = 0, prepend = 0) 
    dv[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dv) >= 2) & (burst_diff==0), 4, 0) 
    dat_flag = dat_flag + xr.where((np.abs(dv) < 2) & (np.abs(dv) >= 1) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dv) < 1) & (np.abs(dv) >= .25) & (burst_diff==0), 1, 0)
    testCounter = testCounter + 1 

    # For vertical (w)
    dw = np.diff(ds.Up, axis = 0, prepend = 0) 
    dw[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dw) >= 1) & (burst_diff==0), 4, 0) # Magnitudes of vertical velocity are typically smaller than horizontal velocity, so thresholds are reduced
    dat_flag = dat_flag + xr.where((np.abs(dw) < 1) & (np.abs(dw) >= .5) & (burst_diff==0), 3, 0) 
    dat_flag = dat_flag + xr.where((np.abs(dw) < .5) & (np.abs(dw) >= .15) & (burst_diff==0), 1, 0)
    testCounter = testCounter + 1 
       
    # Current speed test
    dat_flag = dat_flag + xr.where(ds.CSPD < 4, 0, 3)
    testCounter = testCounter + 1 
   
    # Current speed and direction rate of change tests
    dCSPD = np.diff(ds.CSPD, axis = 0, prepend = 0)
    dCSPD[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) >= 4) & (burst_diff==0), 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 4) & (np.abs(dCSPD) >= 1) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 1) & (np.abs(dCSPD) >= .25) & (burst_diff==0), 1, 0)
    testCounter = testCounter + 1 

    # For current direction
    dCDIR = np.diff(ds.CDIR, axis = 0, prepend = 0)
    dCDIR[0] = dCDIR[0] - dCDIR[0]
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) >= 135) & (np.abs(dCDIR) <= 225)& (burst_diff==0), 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) < 135) & (np.abs(dCDIR) >= 30) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) <= 330) & (np.abs(dCDIR) > 225) & (burst_diff==0), 3, 0)
    testCounter = testCounter + 1 
    
    # Add the new flag data array to the existing dataset
    flagAvg = (dat_flag)/testCounter
    datFlagQartod = datFlagQartod + xr.where(flagAvg > 4, 9, 0)
    datFlagQartod = datFlagQartod + xr.where((flagAvg <= 4) & (flagAvg > 3), 4, 0)
    datFlagQartod = datFlagQartod + xr.where((flagAvg <= 3) & (flagAvg > 1), 3, 0)
    datFlagQartod = datFlagQartod + xr.where((flagAvg <= 1), 1, 0)
    ds['DataFlag'] = (["time"], datFlagQartod.values)
    ds['DataFlag'].attrs['Flag score'] = '[1, 3, 4, 9]'
    ds['DataFlag'].attrs['Grade definition'] = '1 = Pass, 3 = Suspect, 4 = Non-critical Fail, 9 = Critical Fail'
    ds['DataFlag'].attrs['Description'] = 'Flag grading system is based on QARTOD quality control parameters and tests in Nortek ADV user manual'
    
    print('Flagging sensor data')
    # SENSOR TESTS 
    # Tests for the quality of sensor parameters found on the .sen file  
    
    sen_flag = xr.zeros_like(ds.BatVolt) # Same shape as .sen data arrays
    senFlagQartod = xr.zeros_like(sen_flag)
    
    testCounter = 0
    
    # Battery voltage test
    sen_flag = sen_flag + xr.where(ds.BatVolt >= 9.6, 0, 3) # xr.where(condition, value if true, value if false)
    testCounter = testCounter + 1
    
    # Compass Heading test
    sen_flag = sen_flag + xr.where((ds.Heading >= 0) & (ds.Heading <= 360), 0, 4)
    testCounter = testCounter + 1
    
    # Soundspeed test
    sen_flag = sen_flag + xr.where((ds.SoundSpeed >= 1493) & (ds.SoundSpeed <= 1502), 0, 4)
    testCounter = testCounter + 1
    
    # Tilt test
    sen_flag = sen_flag + xr.where(np.abs(ds.Roll) < 5, 0, 4)
    sen_flag = sen_flag + xr.where(np.abs(ds.Pitch) < 5, 0, 4)
    testCounter = testCounter + 1
    
    # Checksum tests
    sen_flag = sen_flag + xr.where(ds.ChecksumSen == 0, 0, 4) 
    testCounter = testCounter + 1
    
    senFlagQartod = senFlagQartod + xr.where(sen_flag <= 1, 1, 0)
    senFlagQartod = senFlagQartod + xr.where((sen_flag > 1) & (sen_flag < 4), 3, 0)
    senFlagQartod = senFlagQartod + xr.where(sen_flag >= 4, 4, 0)
    
    ds['SenFlag'] = (["time_sen"], senFlagQartod.values)
    ds['SenFlag'].attrs['Flag score'] = '[1, 3, 4]'
    ds['SenFlag'].attrs['Grade definition'] = '1 = Pass, 3 = Suspect, 4 = Fail'
    ds['SenFlag'].attrs['description'] = 'Flag value based on internal sensor tests: battery, heading, pitch and roll, temperature, and soundspeed. Sampled at 1Hz.'
    
    return ds
#===============================================================================================================================

def vector_metadata(ds_trimmed, headerFile, metadata_list = None):
    
    ds = ds_trimmed.copy(deep=True)
    
    if metadata_list:
        for m in metadata_list:
            ds.attrs[m[0]] = m[1]
    
    headerlines = []

    #Runs through all lines in the file and splits them by [attribute name, attribute value]
    with open(headerFile) as f: 
        for i in f.readlines():
            line = i
            line = line.strip('\n')
            line = line.replace('(','')
            line = line.replace(')','')
            line = line.replace('Gyro/Accel', 'Gyro')
            line = line.replace('Hz', '')
            line = re.split(r'\s{2,}', line)


            #Keep relevant vars, drop empty spaces
            if len(line) >= 2:
                headerlines.append(line)
            else:
                continue
    for i in headerlines[:36]:
        #Use try and except statement to make some metadata sections an int data type
        #but switch it to a string if the section has letters present
        try:
            ds.attrs[str(i[0])] = int(i[1])
        except:
            ds.attrs[str(i[0])] = str(i[1])
    
    return ds

#===============================================================================================================================
#=============================================== DESPIKING AND CLEANING FUNCTIONS ==============================================
#===============================================================================================================================

def findPrincaxs(u,v):
    # THE FOLLOWING CODE IS ADAPTED FROM STEVEN CUNNINGHAM'S MASTERS THESIS (2019)

    # Rotate velocity data along the principle axes U and V
    theta, major, minor = ts.princax(u, v) # theta = angle, major = SD major axis (U), SD minor axis (V)
    U, V = ts.rot(u, v, theta)
    
    return U,V,theta

#===============================================================================================================================
def rotateVec(vector):

    #rotate vector data to the principal axis
    #vector is an xarray dataset

    #don't perform modifications in place
    v = vector.copy(deep=True)

    #initialize ENU variable for deciding what variables to rotate (assume we have ENU)
    ENU = 1

    #if in XYZ coordinates, try to convert to ENU coordinates before rotating
    #remember to propagate non-original data through
    if v.attrs['Coordinate system'] == 'XYZ':
        v['Up'] = ('time',v.W.values)
        v['UpOrig'] = ('time',v.WOrig.values)
        try:
            angle = v.attrs['X Direction (degrees)']
        except:
            ENU = 0
        else:
            rad = np.mod(450-angle,360)*2*np.pi/360
            v['East'] = ('time',np.cos(rad)*v.U.values - np.sin(rad)*v.V.values)
            v['North'] = ('time',np.sin(rad)*v.U.values + np.cos(rad)*v.V.values)
            v['EOrig'] = ('time',np.logical_and(v.UOrig.values,v.VOrig.values))
            v['NOrig'] = ('time',v.EOrig.values)


    if ENU == 1:
        # Using principle component analysis to rotate data to uncorrelated axes
        #[theta, primary, secondary] = PCrotate(v.East.values + 1j * v.North.values)
        primary,secondary,theta = findPrincaxs(v.East.values, v.North.values)
        orig = np.logical_and(v.EOrig.values,v.NOrig.values)
    else:
        # Using principle component analysis to rotate data to uncorrelated axes
        #[theta, primary, secondary] = PCrotate(v.U.values + 1j * v.V.values)
        primary,secondary,theta = findPrincaxs(v.U.values, v.V.values)
        orig = np.logical_and(v.UOrig.values,v.VOrig.values)

    # store angle of rotation as attribute and new vectors as primary and secondary
    # velocities in dataset
    v.attrs['Theta'] = theta
    v['Primary'] = ('time',primary)
    v.Primary.attrs['units'] = 'm/s'
    v['Secondary'] = ('time',secondary)
    v.Secondary.attrs['units'] = 'm/s'
    
    v['PrimaryOrig'] = ('time',orig)
    v['SecondaryOrig'] = ('time',orig)

    return v

#===============================================================================================================================
def findLimits(s1,s2,u1,u2,theta,expand=True,expSize = 0.01,expEnd = 0.95):
    #helper function for determining limits and rotating data to compare with limits.
    #s1 and s2 are unrotated / unexpanded limits, while u1 and u2 are unrotated data
    #if theta = 0, u1rot = u1 and u2rot = u2
    #if expand = False and theta = 0, a = s1, and b = s2
    #if expand = True, expSize is the fraction by which the limits are expanded for each step
    #if expand = True, expEnd is the density decrease at which the final limits are selected

    # determine actual ellipse axis lengths based on rotation angle
    a = np.sqrt((s1**2*np.cos(theta)**2-s2**2*np.sin(theta)**2)/(np.cos(theta)**4-np.sin(theta)**4))
    b = np.sqrt((s2**2-a**2*np.sin(theta)**2)/np.cos(theta)**2)

    # to test if data falls inside the ellipse, rotate ccw by theta and compare
    # to an unrotated ellipse
    Urot = (u1+1j*u2)*np.exp(-1j*theta)
    u1rot = np.real(Urot)
    u2rot = np.imag(Urot)

    #expand limits if expand = True
    if expand:

        #calculate the number of points within the ellipse
        def numIn(a,b):
            return np.sum((np.square(u1rot)/np.square(a)+np.square(u2rot)/np.square(b))<=1)

        #determine expansion step size
        diffA = a*expSize
        diffB = b*expSize

        #determine density of points in area between a,b ellipse and next smaller ellipse
        def localDensity(a,b):
            return (numIn(a,b)-numIn(a-diffA,b-diffB))/(np.pi*a*b-np.pi*(a-diffA)*(b-diffB))

        #expand cutoff by expansion step size until the local density decrease 
        #is greater than 95% of the last local density
        while (localDensity(a,b) - localDensity(a+diffA,b+diffB))/localDensity(a,b) < expEnd:
            a += diffA
            b += diffB

    return a,b,u1rot,u2rot

#===============================================================================================================================
def despike(u, repeated, expand=False,expSize = 0.01,expEnd = 0.95):
    #do one despike iteration of data u
    #if repeated = 0, only use velocity magnitude cutoff
    #if repeated > 0, use full phase space method
    #if expand = true use expanded limits method
    #if expand = false, stick with gaussian limits

    # calculate first and second derivatives of u
    n = u.size
    du = np.empty(n)*np.nan
    du2 = np.empty(n)*np.nan
    du[1:-1] = (u[2:]-u[0:-2])/2
    du[0] = u[1]-u[0]
    du[-1] = u[-1]-u[-2]
    du2[1:-1] = (du[2:]-du[0:-2])/2
    du2[0] = du[1]-du[0]
    du2[-1] = du[-1]-du[-2]

    # Determine Expected Maximum Value assuming normal random variables with zero mean
    lamda_u = (np.sqrt(2*np.log(n)))*np.nanstd(u)
    lamda_du = (np.sqrt(2*np.log(n)))*np.nanstd(du)
    lamda_du2 = (np.sqrt(2*np.log(n)))*np.nanstd(du2)

    #expand limits in u-du plane
    (u_lim1,du_lim1,_,_) = findLimits(lamda_u,lamda_du,u,du,0,expand=expand,expSize=expSize,expEnd=expEnd)

    #check for obvious spikes
    j1 = np.where(np.abs(u)>u_lim1)[0]

    if np.logical_and(j1.size > 0, repeated<=1):
        #if obvious spikes are found, replace them and then restart iteration
        spikes = j1
        du_lim2 = np.nan
        du2_lim2 = np.nan
        a_lim = np.nan
        b_lim = np.nan
        theta = np.nan

    else:
        #if obvious spikes are not found, go through full phase space spike identification
        
        #find spikes outside of u-du ellipse
        j1 = np.where((np.square(u)/np.square(u_lim1)+np.square(du)/np.square(du_lim1))>1)[0]

        #determine du-du2 ellipse and find spikes outside of it
        (du_lim2,du2_lim2,_,_) = findLimits(lamda_du,lamda_du2,du,du2,0,expand=expand,expSize=expSize,expEnd=expEnd)
        j2 = np.where((np.square(du)/np.square(du_lim2)+np.square(du2)/np.square(du2_lim2))>1)[0]

        #Determine principle axis rotation angle between u and du2
        theta = np.arctan(np.nansum(u*du2)/np.nansum(u**2))

        #rotate u-du2 plane, expand limits, and find spikes outside of corresponding ellipse 
        (a_lim,b_lim,a,b) = findLimits(lamda_u,lamda_du2,u,du2,theta,expand=expand,expSize=expSize,expEnd=expEnd)
        j3 = np.where((np.square(a)/np.square(a_lim)+np.square(b)/np.square(b_lim))>1)[0]

        #put all identified spikes together
        spikes = np.union1d(np.union1d(j1,j2),j3)

    #replace spikes
    u[spikes] = np.nan
    detected = np.isnan(u)

    #return data with spikes replaced
    return (nan_sampleHold(u),detected,[u_lim1,du_lim1,du_lim2,du2_lim2,a_lim,b_lim,theta])

#===============================================================================================================================
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    #I got this from an internet forum, but lost the link

    return np.isnan(y), lambda z: z.nonzero()[0]
    
def nan_interp(y):
    """Function to replace nans in a numpy array with interpolated values
    Input:
        - y, 1d numpy array with possible NaNs
    Ooutput:
        - ynew, a 1d numpy with NaNs replaced by interpolated values
    """
    #I got this from an internet forum, but lost the link

    y2 = np.array(y)
    nans, x = nan_helper(y2)
    if np.sum(nans) < nans.size:
        y2[nans] = np.interp(x(nans), x(~nans), y2[~nans])
    return y2
 
def nan_sampleHold(y):
    """
    function to replace nans with the last valid point
    based on 
    https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array/41191127
    """

    #find location of nans
    mask = np.isnan(y)

    #create array of indices (np.arange(mask.size)) with the locations with nans replaced by 0
    idx = np.where(~mask,np.arange(mask.size),0)

    #propagate any maximum forward (so any 0 replaced with the last non-zero index effectively!)
    np.maximum.accumulate(idx, out=idx)

    #use index to construct filled array
    out = y[idx]

    #if the first points where nan, replace them with the next valid data
    out[np.isnan(out)] = out[np.where(~np.isnan(out))[0][0]]

    return out

def nanZero(y):
    y[np.isnan(y)] = 0
    return y
    
# endregion

#===============================================================================================================================
def despike_iterate(velocities,hz=16,lp=1/20,expand=True, plot=False,verbose=False,expSize = 0.01,expEnd = 0.95):
    #perform despike algorithm on a single burst / velocity component
    #velocities is a 1-d numpy array with the velocity data
    #hz is the sampling rate of the velocity data
    #lp is the frequency at which the data is low pass filtered before despiking
    #expand = true uses an expanding cutoff, false uses a gaussian based cutoff
    #plot = true outputs a plot for the results of the despiking and returns the figure
    #verbose = true returns the limits used for the cutoff

    #initialize array for storing cutoff limits in case data can't be despiked and returns must be given
    lims = [None,None,None,None,None,None,None,None]

    #initialize array for storing what points are identified as spikes
    detectedAll = np.zeros(velocities.shape,dtype='bool')

    #check that there aren't too many nans (e.g out of water or bad correlation for practically whole burst)
    if np.sum(~np.isnan(velocities)) < 20:
        print('Too few valid points')
        if plot:
            if verbose:
                return velocities, detectedAll, None, lims
            else:
                return velocities, detectedAll, None
        else:
            if verbose:
                return velocities, detectedAll, lims
            else:
                return velocities, detectedAll

    #get rid of nans for now but save locations of nans
    nanloc = np.where(np.isnan(velocities))
    vel = nan_interp(velocities.copy())

    #find, save, and remove low pass signal
    sos = signal.butter(4,lp,'lp',fs=hz,output='sos') 
    low = signal.sosfiltfilt(sos,vel)
    v1 = vel-low

    #do initial despike pass
    (v2,detected,lims) = despike(v1, 0, expand=expand,expSize=expSize,expEnd=expEnd)

    #determine how many of the detected spikes were new
    numdiff = np.sum(np.logical_and(detected,~detectedAll))

    #store detected spikes
    detectedAll[detected] = True

    #set up while loop
    repeated = 0
    iterations = 1
    print(numdiff)

    #loop until no spikes are detected or the same # are detected 3 times in a row, or looped 100 times
    #note, the detected 3 times in a row here is used to force the despike algorigthm to use the full
    #phase space method while still detecting no spikes before exiting the loop
    #if repeated = 0, the despike algorithm will only use the velocity magnitude cutoff.
    while not(np.logical_or(repeated==3, iterations==100)):

        #reassign end of last despike as start of new despike
        v1 = v2

        #despike again
        (v2,detected,lims) = despike(v1, repeated, expand=expand,expSize=expSize,expEnd=expEnd)

        #determine how many of the detected spikes were new
        numdiff = np.sum(np.logical_and(detected,~detectedAll))

        #store detected spikes
        detectedAll[detected] = True

        #if no new spikes are detected, increase repeated number
        if numdiff == 0:
            repeated += 1
        else:
            repeated = 0

        #increase iteration count for timing out
        iterations += 1

        print(numdiff)

    #if we ended the loop on a non-full phase-space detection (only if forced out because of max iterations)
    #force a full phase-space detection
    if np.isnan(lims[3]):
        (v2,detected,lims) = despike(v1, repeated=3,expand=expand,expSize=expSize,expEnd=expEnd)
        detectedAll[detected] = True

    #add back the low pass signal and nans to the final despike result
    final = v2+low
    final[nanloc] = np.nan

    #mark nans as non-original points as well 
    detectedAll[nanloc] = True

    #plot despike results plot
    if plot:

        #remove low pass signal from original data
        u = vel-low

        #store total number of data points
        n = u.size

        #add back nans to high frequency despike result
        vplot = v2
        vplot[nanloc] = np.nan

        #determine where the spikes are
        spikes = u!=v2

        #add back nans to high frequency original data
        u[nanloc] = np.nan

        #calculate first and second derivatives of high frequency original data
        du = np.empty(n)*np.nan
        du2 = np.empty(n)*np.nan
        du[1:-1] = (u[2:]-u[0:-2])/2
        du[0] = u[1]-u[0]
        du[-1] = u[-1]-u[-2]
        du2[1:-1] = (du[2:]-du[0:-2])/2
        du2[0] = du[1]-du[0]
        du2[-1] = du[-1]-du[-2]

        #create figure
        fig = plt.figure(figsize = (15,15))
        ax1 = plt.subplot(3,3,1)
        ax2 = plt.subplot(3,3,2)
        ax3 = plt.subplot(3,3,3)
        ax4 = plt.subplot(3,1,2)
        ax5 = plt.subplot(3,1,3)

        #plot phase space scatter plots with ellipse cutoffs
        plotDespikePlane(u,du,lims[0],lims[1],0,'u','du',ax1)
        plotDespikePlane(du,du2,lims[2],lims[3],0,'du','du2',ax2)
        plotDespikePlane(u,du2,lims[4],lims[5],lims[6],'u','du2',ax3)

        #plot time series of original and despiked high frequency data
        ax4.plot(u,color='red',label='original data')
        ax4.plot(vplot,'.-',color='blue',label='despiked data')
        ax4.set_ylabel('u (m/s)')

        #plot original and final despiked data including low frequency components
        ax5.plot(vel,color='red',label='original data')
        ax5.plot(final,'.-',color='blue',label='despiked data')
        ax5.set_xlabel('sample number')
        ax5.set_ylabel('u (m/s)')
        ax5.legend(loc='upper right')

        if verbose:
            return final, detectedAll, fig, lims
        else:
            return final, detectedAll, fig
    if verbose:
        return final, detectedAll, lims 
    else:
        return final, detectedAll
    
#===============================================================================================================================
def despike_all(vector, fs, expand=False,lp=1/20,savefig=False,savePath=None,expSize = 0.01,expEnd = 0.95):
    #Despike all velocity components for all bursts in vector.
    #vector is an xarray dataset with the data collected from an adv
    #expand = true uses the expanding cutoff, while false uses gaussian cutoffs
    #lp gives the frequency at which the data is low pass filtered before despiking
    #savefig = true tells the function to save all despiking figures
    #savePath is the path to the folder in which the figures are saved

    #copy vector to not modify it in place
    v = vector.copy(deep=True)

    #pull sampling frequency of data
    hz = fs

    #pull serial number of instrument for labelling and saving figures
    #serial = vector.attrs['Hardware Serial Number'].split(' ')[-1]

    #save total number of bursts for outputing progress status
    maxOut = str(v.BurstNum.max().values)

    #check what coordinate system we are working with
    if v.attrs['Coordinate system'] == 'XYZ':

        #create new variables for storing what data is original
        v['UOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 
        v['VOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 
        v['WOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 

        #loop through all bursts
        for i in v.BurstCounter.values.astype('int'):

            #output progress status
            print(str(float(i)) + ' of ' + maxOut)

            #get indices of current burst data
            index = np.where(v.BurstNum==i)[0]

            #despike each velocity component
            Urep, detectedU = despike_iterate(v.U.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)
            Vrep, detectedV = despike_iterate(v.V.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)
            Wrep, detectedW = despike_iterate(v.W.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)

            #create and save figures
            if savefig:
                print('saving burst ' + str(i))
                saveDespikeFig(Urep,v.U[index].values,savePath+'U_Burst_{:04d}'.format(i))
                saveDespikeFig(Vrep,v.V[index].values,savePath+'V_Burst_{:04d}'.format(i))
                saveDespikeFig(Wrep,v.W[index].values,savePath+'W_Burst_{:04d}'.format(i))

            #store info on what points are original or not
            v.UOrig[index] = ~detectedU
            v.VOrig[index] = ~detectedV
            v.WOrig[index] = ~detectedW

            #update with despiked data
            v.U[index] = Urep
            v.V[index] = Vrep
            v.W[index] = Wrep

    elif v.attrs['Coordinate system'] == 'ENU':

        #create new variables for storing what data is original
        v['EOrig'] = ('time',np.ones(v.time.shape,dtype='bool'))
        v['NOrig'] = ('time',np.ones(v.time.shape,dtype='bool'))  
        v['UpOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 

        #loop through all bursts
        for i in v.BurstCounter.values.astype('int'):
            
            #output progress status
            print(str(float(i)) + ' of ' + maxOut)

            #get indices of current burst data
            index = np.where(v.BurstNum==i)[0]

            #despike each velocity component
            Erep, detectedE = despike_iterate(v.East.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd) 
            Nrep, detectedN = despike_iterate(v.North.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)
            Uprep, detectedUp = despike_iterate(v.Up.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)

            #create and save figures
            if savefig:
                print('saving burst ' + str(i))
                saveDespikeFig(Nrep,v.North[index].values,savePath+'North_Burst_{:04d}'.format(i))
                saveDespikeFig(Erep,v.East[index].values,savePath+'East_Burst_{:04d}'.format(i))
                saveDespikeFig(Uprep,v.Up[index].values,savePath+'Up_Burst_{:04d}'.format(i))

            #store info on what points are original or not
            v.NOrig[index] = ~detectedN
            v.EOrig[index] = ~detectedE
            v.UpOrig[index] = ~detectedUp

            #update with despiked data
            v.North[index] = Nrep
            v.East[index] = Erep
            v.Up[index] = Uprep

    return v
#===============================================================================================================================
def cleanVec(vector,corrCutoff=0,snrCutoff=0,angleCutoff=10000):
    #cleans data based on a correlation, snr, and tilt angle cutoff

    #copy to not modify in place
    v = vector.copy(deep=True)

    #initialize index
    index = np.zeros(np.shape(v.time.values),dtype='bool')

    #find where correlation cutoff fails
    index[np.logical_or(np.logical_or(v.Corr1.values < corrCutoff, \
                                 v.Corr2.values < corrCutoff), \
                                 v.Corr3.values < corrCutoff)] = True

    #find where snr cutoff fails
    index[np.logical_or(np.logical_or(v.Snr1.values < snrCutoff, \
                                 v.Snr2.values < snrCutoff), \
                                 v.Snr3.values < snrCutoff)] = True

    #if angle is too big, assume we don't have trustworthy tilt information
    if np.abs(angleCutoff) < 2*np.pi:

        #get pitch and roll info on correct timestep
        pitch = v.Pitch.interp(time_sen = v.time.values).values
        roll = v.Roll.interp(time_sen=v.time.values).values
        
        #convert pitch and roll to total tilt
        tilt = np.arctan(np.sqrt(np.tan(pitch*2*np.pi/360)**2+np.tan(roll*2*np.pi/360)**2))

        #find where angle cutoff fails
        index[tilt>angleCutoff] = True

    #nan out data that failed the cutoff
    if v.attrs['Coordinate system'] == 'XYZ':
        v.U[index] = np.nan
        v.V[index] = np.nan
        v.W[index] = np.nan 

    if v.attrs['Coordinate system'] == 'ENU':
        v.North[index] = np.nan
        v.East[index] = np.nan
        v.Up[index] = np.nan

    #store what cutoffs were used
    v.attrs['CorrCutoff'] = corrCutoff
    v.attrs['SnrCutoff'] = snrCutoff
    v.attrs['AngleCutoff'] = angleCutoff
    
    return v
#===============================================================================================================================
def ProcessVec(data, fs, badSections,reverse,expand = True,lp = 1/20,expSize = 0.01,expEnd = 0.95):
    #initial processing of adv data. adjusts pressure,
    #calculates depth, eliminates bad data due to
    #bad time step, out of water transitions, snr,
    #correlation, and despiking
    #
    #data is an xarray dataset
    #if reverse is true, reverses the primary and secondary velocity components (to make sure flood is positive)
    #expand, lp, expSize, and expEnd are variables passed to despike_all
    #if expand = True, an expanding cutoff is used
    #lp is the frequency cutoff for the lowpass filter used to remove the low pass signal
    #expSize is the step size for expanding the cutoff in the expanding cutoff algorithm
    #expEnd is the density change cutoff for determining when to stop expanding the phase space cutoffs

    hz = fs
    data['Depth'] = ('time', np.abs(gsw.conversions.z_from_p(data.Pressure.values,data.Lat)))

    print('cleaning vector based on correlation and snr cutoffs')

    #use correlation and snr cutoffs to clean data
    data_temp = cleanVec(data,corrCutoff=60,snrCutoff=10)


    print('despiking vector')

    plt.ioff()

    data_temp_All = despike_all(data_temp, hz, expand=expand,lp=lp,savefig=False,savePath=None,expSize=expSize,expEnd=expEnd)

    print('removing bad sections')

    for i in badSections:
        
        #pull out indices of data in bad sections
        ind = np.logical_and(data_temp_All.time>= i[0],data_temp_All.time<=i[1])

        #set data to nan
        data_temp_All.East[ind] = np.nan
        data_temp_All.North[ind] = np.nan
        data_temp_All.Up[ind] = np.nan
    
    print('rotating vectors')
    
    data_temp_All = rotateVec(data_temp_All)
    
    if reverse:
        print('reversing direction so primary is in flooding direction')
        data_temp_All.Primary[:] = -data_temp_All.Primary[:]
        data_temp_All.Secondary[:] = -data_temp_All.Secondary[:]
        data_temp_All.attrs['Theta'] = data_temp_All.Theta - np.pi
    
    print('re-adding raw velocity data')
    
    data_temp_All["EastRaw"] = (["time"], data.East)
    data_temp_All["EastRaw"].attrs['Description'] = 'Eastern velocity from original, unfiltered dataset'
    data_temp_All["NorthRaw"] = (["time"], data.North)
    data_temp_All["NorthRaw"].attrs['Description'] = 'Northern velocity from original, unfiltered dataset'
    data_temp_All["UpRaw"] = (["time"], data.Up)
    data_temp_All["UpRaw"].attrs['Description'] = 'Vertical velocity from original, unfiltered dataset'
    
    print('finding fraction of unoriginal data in each burst')
    
    #Initialize boolean vectors of unoriginal points for velocities
    EOrig = data_temp_All.EOrig
    NOrig = data_temp_All.NOrig
    UpOrig = data_temp_All.UpOrig
    PrimOrig = data_temp_All.PrimaryOrig
    SecOrig = data_temp_All.SecondaryOrig
    
    data_temp_All = badDataRatio(data_temp_All, EOrig, 'dEast')
    data_temp_All = badDataRatio(data_temp_All, NOrig, 'dNorth')
    data_temp_All = badDataRatio(data_temp_All, UpOrig, 'dUp')
    data_temp_All = badDataRatio(data_temp_All, PrimOrig, 'dPrimary')
    data_temp_All = badDataRatio(data_temp_All, SecOrig, 'dSecondary')

    data_temp_All.attrs['despike_lp_freq (hz)'] = lp
    data_temp_All.attrs['despike_cutoff_expansion_fraction'] = expSize
    data_temp_All.attrs['despike_cutoff_expansion_densityChange_end_condition'] = expEnd

    return data_temp_All
    
#===============================================================================================================================
def badDataRatio(data, velCompBool, newVarName):
    
    #Max number of samples within a single burst
    nmax = data.NoVelSamples.values[0]
    
    #Find and drop all bad datapoints within BurstNum
    goodData = data.BurstNum.where(velCompBool==True, drop=True)
    
    #Create array of remaining datapoints within each burst
    remBursts = np.unique(goodData, return_counts = True)
    
    #Bursts with 0 good points get dropped, making the array uneven with dataset dimensions
    #Dropped bursts must be manually added back to the array to make complete
    #Find difference between unaltered dataset and remBurst to see which bursts get compeletely wiped
    a = np.setdiff1d(data.BurstNum.values,remBursts[0])
    
    #Manually create an array of zeros for each of the wiped bursts and merge the two
    b = np.zeros_like(a)
    badBursts = np.array((a,b))
    
    #Append remBursts with badBursts to make an array with matching dimensions to dataset
    badData = np.append(remBursts, badBursts, axis=1).astype(int)
    
    #Sort bursts so that indices of badBursts are in temporal order (i.e burst # 1, 2, 3...95, 96)
    sort = np.argsort(badData[0])
    badData = badData[1][sort]
    
    #Calculate ratio of bad data to max number of samples per burst
    badDataRatio = (nmax-badData)/nmax #Also knowns as dSS and dCorr in Feddersen (2010)
    
    #Add badDataRatio to the dataset
    data.coords["burst"] = (["burst"], data.BurstCounter.values)
    data[str(newVarName)] = (["burst"], badDataRatio)
    data[str(newVarName)].attrs['Description'] = 'Ratio of bad data points to the total burst size for despiked velocity'
    
    return data
#===============================================================================================================================
def fullInterpVec(data):
    
    ds = data.copy(deep=True)
    
    ds['EOrig'] = missingValE = xr.where((ds.EOrig==False) | (ds.East.isnull()==True), False, True)
    ds['NOrig'] = missingValN = xr.where((ds.NOrig==False) | (ds.North.isnull()==True), False, True)
    ds['UpOrig'] = missingValUp = xr.where((ds.UpOrig==False) | (ds.Up.isnull()==True), False, True)
    ds['PrimaryOrig'] = missingValPrim = xr.where((ds.PrimaryOrig==False) | (ds.Primary.isnull()==True), False, True)
    ds['SecondaryOrig'] = missingValSec = xr.where((ds.SecondaryOrig==False) | (ds.Secondary.isnull()==True), False, True)
    
    print('Linearly interpolating dataset')
    #Perform full linear interpolation of data with no temporal limits
    dsFullInterp = ds.interpolate_na(dim="time", method="linear")
    
    print('Evaluating ratio of nans leftover in the dataset')
    #Evaluate the numer of nans in each dataset for future qc
    badDataRatio(dsFullInterp, missingValE, 'dNorth')
    badDataRatio(dsFullInterp, missingValN, 'dEast')
    badDataRatio(dsFullInterp, missingValUp, 'dUp')
    badDataRatio(dsFullInterp, missingValPrim, 'dPrimary')
    badDataRatio(dsFullInterp, missingValSec, 'dSecondary')
    
    return dsFullInterp

#===============================================================================================================================
def patchVec(data, fs):
    
    ds = data.copy(deep=True)

    ds['EOrig'] = missingValE = xr.where((ds.EOrig==False) | (ds.East.isnull()==True), False, True)
    ds['NOrig'] = missingValN = xr.where((ds.NOrig==False) | (ds.North.isnull()==True), False, True)
    ds['UpOrig'] = missingValUp = xr.where((ds.UpOrig==False) | (ds.Up.isnull()==True), False, True)
    ds['PrimaryOrig'] = missingValPrim = xr.where((ds.PrimaryOrig==False) | (ds.Primary.isnull()==True), False, True)
    ds['SecondaryOrig'] = missingValSec = xr.where((ds.SecondaryOrig==False) | (ds.Secondary.isnull()==True), False, True)
    
    print('Interpolating gaps <= 1s')
    delta = timedelta(seconds=1 + (2/fs)) #Maximum gap of 1 second (2/fs ensures that points right at the margin are taken into account)
    dsPatch = ds.interpolate_na(dim="time", method="linear", max_gap = delta) #Linearly interpolates across gaps up to a limit denoted by time delta
    
    print('Evaluating ratio of nans leftover in the dataset')
    #Evaluate the numer of nans in each dataset for future qc
    badDataRatio(dsPatch, missingValE, 'dNorth')
    badDataRatio(dsPatch, missingValN, 'dEast')
    badDataRatio(dsPatch, missingValUp, 'dUp')
    badDataRatio(dsPatch, missingValPrim, 'dPrimary')
    badDataRatio(dsPatch, missingValSec, 'dSecondary')
    
    return dsPatch
#===============================================================================================================================
def interpAvgVec(data, fs):
    
    ds = data.copy(deep=True)

    ds['EOrig'] = missingValE = xr.where((ds.EOrig==False) | (ds.East.isnull()==True), False, True)
    ds['NOrig'] = missingValN = xr.where((ds.NOrig==False) | (ds.North.isnull()==True), False, True)
    ds['UpOrig'] = missingValUp = xr.where((ds.UpOrig==False) | (ds.Up.isnull()==True), False, True)
    ds['PrimaryOrig'] = missingValPrim = xr.where((ds.PrimaryOrig==False) | (ds.Primary.isnull()==True), False, True)
    ds['SecondaryOrig'] = missingValSec = xr.where((ds.SecondaryOrig==False) | (ds.Secondary.isnull()==True), False, True)
    
    print('Interpolating gaps <= 1s')
    delta = timedelta(seconds=1 + (2/fs)) #Maximum gap of 1 second (2/fs ensures that points right at the margin are taken into account)
    dsInt = ds.interpolate_na(dim="time", method="linear", max_gap = delta) #Linearly interpolates across gaps up to a limit denoted by time delta
    
    print('Evaluating ratio of nans leftover in the dataset')
    #Evaluate the numer of nans in each dataset for future qc
    badDataRatio(dsInt, missingValE, 'dNorth')
    badDataRatio(dsInt, missingValN, 'dEast')
    badDataRatio(dsInt, missingValUp, 'dUp')
    badDataRatio(dsInt, missingValPrim, 'dPrimary')
    badDataRatio(dsInt, missingValSec, 'dSecondary')     
       
    print('Organzing remaining gaps > 1s')
    
    gapTimes = dsInt.time.where(dsInt.East.isnull() == True, drop = True).values #Generate a dataset with all leftover uninterpolated gaps
    
    #Find differences between gaps to assess which ranges are non-sequential
    tDiff = np.diff(gapTimes) 
    gapRanges = np.split(gapTimes, np.where(tDiff > np.timedelta64(int(1000000000/fs), 'ns'))[0]+1)
    
    print('Generating interpolated/averaged data')

    EAvg = np.empty(len(gapRanges))
    NAvg = np.empty(len(gapRanges))
    UpAvg = np.empty(len(gapRanges))
    
    for i in range(len(gapRanges)):
        EAvg[i] = dsInt.EastRaw.sel(time = slice(gapRanges[i][0], gapRanges[i][-1])).values.mean()
        NAvg[i] = dsInt.NorthRaw.sel(time = slice(gapRanges[i][0], gapRanges[i][-1])).values.mean()
        UpAvg[i] = dsInt.UpRaw.sel(time = slice(gapRanges[i][0], gapRanges[i][-1])).values.mean()
    
    EastAvgArr = xr.zeros_like(dsInt.East)
    NorthAvgArr = xr.zeros_like(dsInt.North)
    UpAvgArr = xr.zeros_like(dsInt.Up)
    
    for i in range(len(gapRanges)):
        EastAvgArr = EastAvgArr + xr.where(dsInt.time.isin(gapRanges[i]), EAvg[i], 0)
        NorthAvgArr = NorthAvgArr + xr.where(dsInt.time.isin(gapRanges[i]), NAvg[i], 0)
        UpAvgArr = UpAvgArr + xr.where(dsInt.time.isin(gapRanges[i]), UpAvg[i], 0)
        print(str(i)+' out of '+str(len(gapRanges)))
        
    #dsInt.coords['timeInterp'] = (["timeInterp"], dsInt.time)
    dsInt["East"] = (["time"],xr.where(EastAvgArr != 0, EastAvgArr, dsInt.East))
    dsInt["North"] = (["time"],xr.where(NorthAvgArr != 0, NorthAvgArr, dsInt.North))
    dsInt["Up"] = (["time"],xr.where(UpAvgArr != 0, UpAvgArr, dsInt.Up))
    
    dsInt = dsInt.ffill("time")
    dsInt = dsInt.bfill("time")
    
    primary, secondary, theta = rotateVec(dsInt.East, dsInt.North)
    
    dsInt["Primary"] = (["time"],primary)
    dsInt["Secondary"] = (["time"],-secondary)
    dsInt.attrs["Theta"] = theta
    
    return dsInt

#===============================================================================================================================
#=============================================== DISSIPATION ESTIMATION FUNCTIONS ==============================================
#===============================================================================================================================

def JlmIntegral(data):
    ds = data.copy(deep=True)
    burst_list = np.unique(ds.BurstNum.values)
    J_arr = np.empty((len(burst_list),3))
    
    # Initialize all variables theta, phi, and R within boundaries a to b
    phi = np.linspace(0, 2*np.pi, 1000)
    theta = np.linspace(0, np.pi, 1000)
    thetaRS = np.linspace(0, np.pi, 1000).reshape(1000,1)
    
    fTheta_11 = np.empty(len(theta)) # dims (burst, fPhi)
    fTheta_22 = np.empty(len(theta))
    fTheta_33 = np.empty(len(theta))
        
    for b in enumerate(burst_list):
        
        print('Burst #: '+str(b[1]))
        
        U = ds.Primary.where(ds.BurstNum.isin(b[1]), drop = True).values
        V = ds.Secondary.where(ds.BurstNum.isin(b[1]), drop = True).values
        W = ds.Up.where(ds.BurstNum.isin(b[1]), drop = True).values
    
        #Magnitude of current
        ubar = np.nanmean(np.sqrt(U**2))
        vbar = np.nanmean(np.sqrt(V**2))
        wbar = np.nanmean(np.sqrt(W**2))

        #Standard deviations
        usig = np.nanstd(U)
        vsig = np.nanstd(V) 
        wsig = np.nanstd(W) 
    
        #Variance
        uvar = usig**2
        vvar = vsig**2
        wvar = wsig**2

        # Find J_lm using method from Gerbi et al. (2009)
    
        R0 = ((ubar/usig) * (np.sin(thetaRS)*np.cos(phi))) + ((vbar/vsig) * (np.sin(thetaRS)*np.sin(phi)))
    
        G = np.sqrt((np.sin(thetaRS)**2) * (((np.cos(phi)/usig)**2) + ((np.sin(phi)/vsig)**2)) + ((np.cos(thetaRS)/wsig)**2))
    
        P_11 = (1/(G**2))*((((np.sin(thetaRS)**2)*(np.sin(phi)**2))/vvar)+((np.cos(thetaRS)**2)/wvar))
        P_22 = (1/(G**2))*((((np.sin(thetaRS)**2)*(np.cos(phi)**2))/uvar)+((np.cos(thetaRS)**2)/wvar))
        P_33 = ((np.sin(thetaRS)/G)**2) * (((np.cos(phi)/usig)**2) + ((np.sin(phi)/vsig)**2))
    
        fR = quad_vec(lambda R: (R**(2/3))*np.exp(-(((R0-R)**2)/2)), 0, 7)[0]
    
        fPhi_11 = (G**(-11/3))*np.sin(thetaRS)*P_11 * fR
        fPhi_22 = (G**(-11/3))*np.sin(thetaRS)*P_22 * fR
        fPhi_33 = (G**(-11/3))*np.sin(thetaRS)*P_33 * fR
    
        for i in enumerate(theta): # Iterates through theta values (the rows of the empty 2d arrays)

            fTheta_11[i[0]] = np.trapz(fPhi_11[i[0]], phi)
            fTheta_22[i[0]] = np.trapz(fPhi_22[i[0]], phi)
            fTheta_33[i[0]] = np.trapz(fPhi_33[i[0]], phi)
             
        # Evaluate the final integral of fTheta and use it to find J_lm    
        J_11 = (1/(2*((2*np.pi)**(3/2))))*(1/(usig*vsig*wsig)) * np.trapz(fTheta_11, theta)
        J_22 = (1/(2*((2*np.pi)**(3/2))))*(1/(usig*vsig*wsig)) * np.trapz(fTheta_22, theta)
        J_33 = (1/(2*((2*np.pi)**(3/2))))*(1/(usig*vsig*wsig)) * np.trapz(fTheta_33, theta)
        
        J_arr[b[0]] = [J_11,J_22,J_33]
        
    print('Creating Dataframe')
    JlmDF = pd.DataFrame(J_arr, index = burst_list, columns=['J_11', 'J_22', 'J_33'])
        
    return JlmDF

#===============================================================================================================================
# Function for estimating k from wave period and total depth using dispersion relationships
def wavedisp(wavper,h):
    """ 
    (omega,k,Cph,Cg) = wavedisp(wavper,h)
    ------------------
    Returns [omega,k,Cph,Cg]

    Inputs (can use arrays): 
            wavper - wave period [s]
            h - water depth [m]
   
    Outputs: 
    omega - angular wave frequency [radians/s]
    k - angular wave number	 [radians/m]
    Cph - phase speed [m/s]
    Cg - group velocity [m/s]
    """

    """ T Connolly 2014
    based on Matlab function wavedisp.m from S Lentz """
    # make sure inputs are arrays
    wavper=np.array(wavper)
    h=np.array(h)
    
    omega = (2*np.pi)/wavper
    g = 9.8
    c = omega**2*h/g
    
    x = np.sqrt(c)
    
    d = 100*np.ones(np.shape(wavper))
    tol = 5.*np.finfo(float).eps
    while (d>tol).any():
        f1=x*np.tanh(x)-c
        f2=x*(1/np.cosh(x))**2+np.tanh(x)
        x=x-f1/f2
        d=np.abs((c-x*np.tanh(x))/c)
    k=x/h
    Cph=omega/k
    Cg=(g*x*(1/np.cosh(x))**2+g*np.tanh(x))/(2*np.sqrt(g*k*np.tanh(x)))
    
    return (omega,k,Cph,Cg)
#===============================================================================================================================
def sppConversion(Pressure, Rho, fs, nperseg, dBarToPascal = True, ZpOffset = 0, ZvOffset = 0, radianFrequency = True):
    
    if dBarToPascal:
        pressure = Pressure * 10000 #Convert from dBar to Pascals for spectra conversion via linear wave theory
        
    Fp, Sp = welch(pressure, fs = fs, nperseg = nperseg, window='hann', detrend = 'linear') # Pressure spectra
    
    #Convert frequency to radian frequency and wavenumber
    g = 9.8 # Gravity
    z = pressure/(Rho*g) # Depth (m): the recorded pressure converted to meters of seawater
    H = np.mean(z) + ZpOffset # Sea level height (m): mean pressure detected by the pressure sensor plus the height of sensor from the bottom
    Zp = -(np.mean(z)) # Depth of pressure sensor (m)
    Zv = (-H) + ZvOffset # Depth of velocity sensor (m): Sea level height plus the height of the velocity transducers from the bottom
    T = 1/Fp
    omega,k,Cph,Cg = wavedisp(T, H)

    # Generate empty arrays for p' and w' values
    p_prime = np.empty(len(omega))
    w_prime = np.empty(len(omega))

    for j in range(len(omega)): # For loop iterates over all values of omega
        p_prime[j] = (Rho*g)*(np.cosh(k[j]*(Zp+H))/np.cosh(k[j]*H))
        w_prime[j] = (-omega[j])*(np.sinh(k[j]*(Zv+H)))/(np.sinh(k[j]*H))
    scaleFactor = w_prime**2 / p_prime**2
    
    #Calculate the equivalent Sw spectra from Sp
    if radianFrequency:
        Sw_prime = ((Sp/(2*np.pi)) * scaleFactor)
    else:
        Sw_prime = (Sp * scaleFactor)
        
    return Fp, Sw_prime

#===============================================================================================================================
def waveData(advData, Rho, fs, rSamp, nperseg, ZpOffset = 0, ZvOffset = 0):
    advDS = advData.copy(deep=True)
    bursts = advDS.BurstCounter.values
    pTS = np.empty((len(bursts), int(nperseg/2)))
    Pressure_detrend = xr.zeros_like(advDS.Pressure)
    
    print('Converting pressure spectra')
    #Convert each burst of pressure spectra to vertical velocity using linear wave theory
    for i in enumerate(bursts):
        print('Evaluating burst '+str(i[0]+1)+' of '+str(len(bursts)))
        Pressure = advDS.Pressure.where(advDS.BurstNum.isin(i[1]), drop = True)
        Pressure_detrend[i[0]*len(Pressure):(i[0] + 1)*len(Pressure)] = signal.detrend(Pressure)
        Fp, Sw_prime = vt.sppConversion(Pressure, Rho, fs, nperseg, dBarToPascal = True, ZpOffset = ZpOffset, ZvOffset = ZvOffset, radianFrequency = False) 
        pTS[i[0]] = Sw_prime[1:]
    Tp = 1/Fp[1:] #First value is infinite, which is not compatible with matplotlib pcolormesh, so it's cut from the array
    
    #Convert detrended pressure to changes in meters of seawater
    H = (Pressure_detrend * 1000)/(Rho*9.81)
    Hmean = H.resample(time = rSamp).mean()
    Hstd = H.resample(time = rSamp).std()
    
    print('Creating dataset')
    #Create dataset
    waveData = xr.Dataset(
        data_vars=dict(
            waveSpectra=(["time", "period"], pTS.data),
            Hmean=(["time"], Hmean.data),
            Hstd=(["time"], Hstd.data),
        ),
        coords=dict(
            time=(["time"],advDS.time_start.data),
            period=(["period"], Tp.data),
        ),
        attrs=dict(description="Pressure spectra over both adv deployments"),
    )
    
    waveData.attrs['Rho'] = Rho.values
    waveData.attrs['Sample Frequency'] = fs
    waveData.attrs['Average Interval'] = rSamp
    waveData.attrs['Segment Length'] = nperseg
    waveData.attrs['Pressure Sensor Offset'] = ZpOffset
    waveData.attrs['Velocity Beam Offset'] = ZvOffset
    waveData.attrs['Velocity Beam Offset'] = ZvOffset
    
    return waveData

#===============================================================================================================================
def power_law(x, a, b):
    return a*np.power(x, b)

#===============================================================================================================================
def kol_law(x,a):
    return a*np.power(x,(-5/3))

#================================================================================================================================
def EpsCalc(vecDS, tempDS, badDataRatioCutoff, selBurstNumbers = None, nperseg=None, minimumGap=1, noiseFrequency = 3.1, ZpOffset = 0, ZvOffset = 0):

    ds = vecDS.copy(deep=True)
    tempData = tempDS.copy(deep=True)
    fs = ds.attrs['Sampling rate']
    if selBurstNumbers:
        burstList = np.unique(selBurstNumbers)
    else:
        goodBursts = ds.burst.where(ds.dUp < badDataRatioCutoff, drop=True)
        burstList = np.unique(goodBursts)

    time_start = ds.time_start.where(ds.BurstCounter.isin(burstList), drop=True)

    print('Initializing arrays')
    #Initialize dimensions of frequency and spectrum using a burst from the dataset
    testBurst = ds.Up.where((ds.BurstNum.isin(burstList[0])) & (ds.Up.isnull()==False), drop = True)
    Ftest, Stest = welch(testBurst, fs=fs, nperseg= nperseg, window='hann') # Vertical velocity spectra

    #Convert frequency to radian frequency and wavenumber
    T = 1/Ftest
    H = np.mean(ds.Depth)
    omega,k,Cph,Cg = wavedisp(T, H)

    J33 = ds.J33.where(ds.burst.isin(burstList), drop=True).values # Wavenumber space integral
    dUp = ds.dUp.where(ds.burst.isin(burstList), drop=True).values # Ratio of bad datapoints within the burst

    #Calculate a minimum gap in terms of the frequency index returned by scipy.welch
    minGap = int((minimumGap*2*np.pi)/np.diff(omega)[0])

    #Convert noise floor to radian frequency
    fNoise = noiseFrequency*2*np.pi
    Fn = np.where(omega == fNoise)[0][0] #Minimum frequency before the noise floor begins

    #Initialize arrays to hold all variables generated by the dataset
    fullSw = np.empty((len(burstList),len(omega))) #Full spectrum of velocity components
    fullSu = np.empty((len(burstList),len(omega)))
    fullSv = np.empty((len(burstList),len(omega)))
    fullSp = np.empty((len(burstList),len(omega))) #Converted pressure spectrum
    isrUpper = np.empty(len(burstList)) #Upper boundary of inertial subrange (ISR)
    isrLower = np.empty(len(burstList)) #Lower boundary of ISR
    Mu = np.empty(len(burstList)) #Slope of ISR fit
    MuErr = np.empty(len(burstList)) #Error of slope
    Int = np.empty(len(burstList)) #Intercept of ISR fit
    IntErr = np.empty(len(burstList)) #Error of slope
    KolInt = np.empty(len(burstList)) #Intercept of -5/3 fit
    KolIntErr = np.empty(len(burstList)) #Error of -5/3 intercept
    FitMisfit = np.empty(len(burstList)) #Misfit between ISR slope and -5/3
    maxSw = np.empty(len(burstList)) 
    minSw = np.empty(len(burstList))
    Noise = np.empty(len(burstList)) #Magnitude of noise floor
    wavePeak = np.empty(len(burstList)) #Magnitude peak wave band frequency

    #Initialize arrays to hold all dissipation estimates and eps variables
    epsMag = np.empty(len(burstList)) #Mean of eps values over isr
    epsErr = np.empty(len(burstList)) #Error in eps estimate
    epsFitInt = np.empty(len(burstList)) # Intercept of eps estimate linear regression (LR) model
    epsFitSlope = np.empty(len(burstList)) #LR Slope
    epsFitR2val = np.empty(len(burstList)) #LR R2-value
    epsFitPval = np.empty(len(burstList)) #LR P-value
    epsFitSlopeErr = np.empty(len(burstList)) #Error of LR slope
    epsFitIntErr = np.empty(len(burstList)) #Error of LR intercept

    #Initialize array to hold Ozmidov and Komogorov length scale values once eps has been estimated
    LenOz = np.empty(len(burstList))
    LenKol = np.empty(len(burstList))

    #Begin calculating epsilon for each of the specified bursts
    for b in enumerate(burstList):
        print('Evaluating burst '+str(b[0]+1)+' of '+str(len(burstList)))
        
        #Identify burst number to be evaluated
        burstNumber = b[1]

        #Retrieve variables from the burst time period
        burstTime = ds.time.where(ds.BurstNum.isin(burstNumber), drop = True)
        burstTemp = ds.Temperature.sel(time_sen=slice(burstTime[0],burstTime[-1])) #Temperature recorded at depth of adv head

        burstUp = ds.Up.where((ds.BurstNum.isin(burstNumber)) & (ds.Up.isnull()==False), drop = True)
        burstU = ds.Primary.where((ds.BurstNum.isin(burstNumber)) & (ds.Primary.isnull()==False), drop = True)
        burstV = ds.Secondary.where((ds.BurstNum.isin(burstNumber)) & (ds.Secondary.isnull()==False), drop = True)

        # Generate vertical and horizontal velocity spectra
        Fw, Sw = welch(burstUp, fs=fs, nperseg= nperseg, window='hann', detrend = 'linear') # Vertical velocity spectra
        Fu, Su = welch(burstU, fs=fs, nperseg= nperseg, window='hann', detrend = 'linear') # Horiztonal velocity spectra
        Fv, Sv = welch(burstV, fs=fs, nperseg= nperseg, window='hann', detrend = 'linear')

        #Convert spectra to radian frequency
        SwOmega = Sw/(2*np.pi)
        SuOmega = Su/(2*np.pi)
        SvOmega = Sv/(2*np.pi)

        # Calculate and convert pressure spectra to vertical velocity to find lower cutoff frequency
        burstPressure = ds.Pressure.where(ds.BurstNum.isin(burstNumber), drop=True)
        rho = tempData.Rho.sel(time=slice(burstTime[0],burstTime[-1])).mean().values #Density at adv during the burst

        Fw_Prime, Sw_prime = sppConversion(burstPressure, rho, fs, nperseg, dBarToPascal = True, ZpOffset = ZpOffset, ZvOffset = ZvOffset, radianFrequency = True)

        #Define the lower cutoff frequency as the end of surface gravity wave band
        try:
            lfc = argrelextrema(Sw_prime, np.less, order=10)[0][0]
        except IndexError:
            #Any issue with indexes should yield a default wave cutoff of .5 Hz
            #.5 is a conservative estiamte of where the wave band ends, but still preceeds most of the potential ISR
            lfc = np.where(Fw == .5)[0][0]

        #Define the upper cutoff frequency as the beginning of the noise floor
        noiseFloor = np.mean(SwOmega[Fn:]) #Noise floor magnitude of vertical velocity spectra
        ufc = Fn

        #Generate list of every possible boundary combination within lower and upper cutoff frequencies

        #Initialize an array of all frequencies within lfc and ufc
        startRange = np.arange(lfc, ufc) 
        bounds = [] #List of all ISR boundaries
        if (ufc-lfc) > minGap:
            #Create an array that is offset by the minimum gap
            iteratorRange = np.arange(lfc+minGap, ufc) #First combination of points will be lfc : lfc + 1Hz gap

            for i in range(ufc-(lfc+minGap)):
                for j in range(len(iteratorRange)):
                    #For each combination, record the boundaries into a list
                    bounds.append((startRange[i], iteratorRange[j]))

                #Each iteration shortens iterator range by 1 to prevent repeat and backwards combinations 
                iteratorRange = iteratorRange[1:ufc] 

        # If the range is shorter than the minimum gap, boundaries become the wave cutoff frequency and the noise floor
        else:
            bounds.append((lfc, ufc))

        #Initialize arrays for curve fitting
        testMinSw = np.empty(len(bounds))
        testInt = np.empty(len(bounds))
        testIntErr = np.empty(len(bounds))
        testMu = np.empty(len(bounds))
        testMuErr = np.empty(len(bounds))
        muDiff = np.empty(len(bounds))
        testEpsMag = np.empty(len(bounds))
        testEpsErr = np.empty(len(bounds))
        testEpsFitInt = np.empty(len(bounds))
        testEpsFitSlope = np.empty(len(bounds))
        testEpsFitR2val = np.empty(len(bounds))
        testEpsFitPval = np.empty(len(bounds))
        testEpsFitSlopeErr = np.empty(len(bounds))
        testEpsFitIntErr = np.empty(len(bounds))
        testKolInt = np.empty(len(bounds))
        testKolIntErr = np.empty(len(bounds))
        testFitMisfit = np.empty(len(bounds))

        #Constants for estimating epsilon
        alpha = 1.5 # Kolomogorov constant
        Jlm = J33[b[0]]

        #Go through the list of all combinations of ISR ranges and store the results
        for i in np.arange(0,len(bounds)):
            try:
                #Generate power law fit using the next set of boundaries
                pars, cov = curve_fit(f=power_law, xdata=omega[bounds[i][0]:bounds[i][1]],
                                          ydata=SwOmega[bounds[i][0]:bounds[i][1]], p0=[0, 0], bounds=(-np.inf, np.inf), maxfev=10000)
                #Fit power curve with fixed -5/3 slope
                pars2, cov2 = curve_fit(f=kol_law, xdata=omega[bounds[i][0]:bounds[i][1]],ydata=SwOmega[bounds[i][0]:bounds[i][1]], maxfev=10000)

                muFit = pars[0] * (omega[bounds[i][0]:bounds[i][1]]**pars[1])
                kolFit = pars2[0] * (omega[bounds[i][0]:bounds[i][1]]**(-5/3))

                testMinSw[i] = SwOmega[bounds[i][0]] #Spectra at lower boundary
                testInt[i] = pars[0] #Fit intercept
                testIntErr[i] = np.sqrt(np.diag(cov))[0] #intercept error to 90% confidence level
                testMu[i] = pars[1] #Fit slope
                testMuErr[i] = np.sqrt(np.diag(cov))[1] #Slope error to 90% confidence level

                testKolInt[i] = pars2[0] #Intercept from -5/3 fit
                testKolIntErr[i] = np.sqrt(np.diag(cov2))[0] #Intercept error from -5/3 fit to 90% confidence level

                muDiff[i] = np.abs(pars[1]+(5/3)) #Misfit of the slope compared to -5/3
                testFitMisfit[i] = np.square(np.subtract(muFit, kolFit)).mean()  #Mean square error between dynamic fit and -5/3 fit

                #Estimate turbulent dissipation (Epsilon/eps)
                isrOmega = omega[bounds[i][0]:bounds[i][1]] #Radian frequency range
                S33 = SwOmega[bounds[i][0]:bounds[i][1]] #Vertical velocity spectra within ISR

                #Dissipation formula (Eq. A14 from Gerbi et al., 2009)
                eps = ((S33 * (isrOmega**(5/3)))/(alpha * Jlm))**(3/2) #Returns array of eps estimates across ISR

                #Fit a linear regression to eps estimates
                res = stats.linregress(isrOmega, eps)

                #Populate arrays
                testEpsMag[i] = np.mean(eps) #Mean value of eps for the entire burst
                testEpsErr[i] = np.sqrt(np.var(eps)/(len(eps)-1)) #Calculate error of the epsilon measurements from variance about the mean
                                                                  #Method from Feddersen (2010)
                testEpsFitInt[i] = res.intercept #Linear regression intercept
                testEpsFitSlope[i] = res.slope #Linear regression slope
                testEpsFitR2val[i] = res.rvalue**2 #R2 value of linear regression
                testEpsFitPval[i] = res.pvalue #P-value of linear regression (used for qc)
                testEpsFitSlopeErr[i] = res.stderr #Error of linear regression slope
                testEpsFitIntErr[i] = res.intercept_stderr #Error of linear regression intercept

            #If curve_fit can't fit properly, use 99999 as error values
            except:
                testMinSw[i] = 99999
                testInt[i] = 99999
                testIntErr[i] = 99999
                testMu[i] = 99999
                muDiff[i] = 99999
                testMuErr[i] = 99999
                testEpsMag[i] = 99999
                testEpsErr[i] = 99999
                testEpsFitInt[i] = 99999
                testEpsFitSlope[i] = 99999
                testEpsFitR2val[i] = 99999
                testEpsFitPval[i] = 99999
                testEpsFitSlopeErr[i] = 99999
                testEpsFitIntErr[i] = 99999
                testKolInt[i] = 99999
                testKolIntErr[i] = 99999
                testFitMisfit[i] = 99999

        #Noise floor test from Gerbi et al. (2009)
        noiseFlag = xr.where(testMinSw/2 > noiseFloor, 0, 1)

        #Curve fit intercept test from Jones and Monosmith (2008)
        intFlag = xr.where(testEpsMag > testIntErr, 0, 1)

        #Spectra fit slope test from Feddersen (2010)
        lowMu = testMu - (2*testMuErr) - .06
        highMu = testMu + (2*testMuErr) + .06
        slopeFlag = xr.where((lowMu < (-5/3)) & (highMu > (-5/3)), 0, 1)

        #Spectra slope test from Wheeler and Giddings (2023)
        normSlopeFlag = xr.where((muDiff/testMuErr) < 1.960, 0, 1)

        #Epsilon linear regression test from Feddersen (2010)
        linRegFlag = xr.where(testEpsFitPval > .01, 0, 1)

        flagSum = np.where((noiseFlag + intFlag + slopeFlag + normSlopeFlag + linRegFlag) == 0)[0] 

        #Linear Regression slope test eliminates a lot of data which may have good eps estimate
        #but a very tiny non-zero slope that is near negligible but with significant p-value
        #If a data burst passes the test, it should be prioritized over other bursts, but too many
        #bursts get rejected if the test is universally applied
        #This if statement ensures that more bursts will pass, and the lin reg test can be applied
        #in post-analysis and modified if need be
        if len(flagSum) == 0:
            finalFlag = np.where((noiseFlag + intFlag + slopeFlag + normSlopeFlag) == 0)[0]
        else:
            finalFlag = flagSum

        #If there are still no valid fits even without linReg test, burst fails and is nanned
        if len(finalFlag) == 0:
            minSw[b[0]] = np.nan
            isrLower[b[0]] = np.nan
            maxSw[b[0]] = np.nan 
            isrUpper[b[0]] = np.nan
            Int[b[0]] = np.nan 
            IntErr[b[0]] = np.nan 
            Mu[b[0]] = np.nan 
            MuErr[b[0]] = np.nan 
            KolInt[b[0]] = np.nan 
            KolIntErr[b[0]] = np.nan 
            FitMisfit[b[0]] = np.nan
            epsMag[b[0]] = np.nan
            epsErr[b[0]] = np.nan
            epsFitInt[b[0]] = np.nan 
            epsFitSlope[b[0]] = np.nan
            epsFitR2val[b[0]] = np.nan
            epsFitPval[b[0]] = np.nan
            epsFitSlopeErr[b[0]] = np.nan
            epsFitIntErr[b[0]] = np.nan
            LenOz[b[0]] = np.nan
            LenKol[b[0]] = np.nan
            print('No valid fits')
            continue

        #If the burst passes, choose the fit with the lowest misfit from -5/3 fit
        bestFit = finalFlag[testFitMisfit[finalFlag].argmin()]

        #Populate global arrays with the best fit range
        fullSw[b[0]] = SwOmega #Full spectrum of velocity components
        fullSu[b[0]] = SuOmega
        fullSv[b[0]] = SvOmega
        fullSp[b[0]] = Sw_prime
        Noise[b[0]] = ufc
        wavePeak[b[0]] = lfc
        minSw[b[0]] = SwOmega[bounds[bestFit][0]]
        isrLower[b[0]] = bounds[bestFit][0]
        maxSw[b[0]] = SwOmega[bounds[bestFit][1]] 
        isrUpper[b[0]] = bounds[bestFit][1]

        #All variables relevant from fitting power curves
        Int[b[0]] = testInt[bestFit] 
        IntErr[b[0]] = testIntErr[bestFit] 
        Mu[b[0]] = testMu[bestFit] 
        MuErr[b[0]] = testMuErr[bestFit] 
        KolInt[b[0]] = testKolInt[bestFit] 
        KolIntErr[b[0]] = testKolIntErr[bestFit] 
        FitMisfit[b[0]] = testFitMisfit[bestFit]

        #All variables pertaining to epsilon 
        epsMag[b[0]] = testEpsMag[bestFit]
        epsErr[b[0]] = testEpsErr[bestFit]
        epsFitInt[b[0]] = testEpsFitInt[bestFit] 
        epsFitSlope[b[0]] = testEpsFitSlope[bestFit]
        epsFitR2val[b[0]] = testEpsFitR2val[bestFit]
        epsFitPval[b[0]] = testEpsFitPval[bestFit] 
        epsFitSlopeErr[b[0]] = testEpsFitSlopeErr[bestFit]
        epsFitIntErr[b[0]] = testEpsFitIntErr[bestFit]

        #Ozmidov length scale
        rho1 = tempData.Rho.sel(depth=4,time=slice(burstTime[0],burstTime[-1])).mean().values #Depths 4 and 6 correspond to 9.1 and 9.7m respectively
        rho2 = tempData.Rho.sel(depth=6,time=slice(burstTime[0],burstTime[-1])).mean().values
        dRho = np.abs(rho2 - rho1)/.6 #Change in density over depth
        rhoBar = tempData.Rho.sel(time=slice(burstTime[0],burstTime[-1])).mean().values #Mean density during the burst
        g = 9.81 #Gravity
        N = np.sqrt((g/rhoBar)*dRho) #Buoyancy frequency
        LenOz[b[0]] = np.sqrt(epsMag[b[0]]/N**3)

        #Calculate the Kolmogorov length scale                       
        nuTemp = burstTemp.mean().values+273.15
        nuPress = burstPressure.mean().values/100 + 0.101325
        nu = iapws95.IAPWS95_PT(nuPress,nuTemp).nu
        LenKol[b[0]] = ((nu**3)/epsMag[b[0]])**.25
                                             
    # Create a new dataset with all relevant variables and epsilon values
    print('Creating Dataset')
    epsDS = xr.Dataset(
        data_vars=dict(
            Su = (["bNum","omega"], fullSu),
            Sv = (["bNum","omega"], fullSv),
            Sw = (["bNum","omega"], fullSw),
            Sp = (["bNum","omega"], fullSp),
            NoiseFloor = (["bNum"], Noise),
            WavePeak = (["bNum"], wavePeak),
            maxSw = (["bNum"], maxSw),
            minSw = (["bNum"], minSw),
            lowBound = (['bNum'], isrLower),
            highBound = (['bNum'], isrUpper),
            Int = (["bNum"], Int),
            IntErr = (["bNum"], IntErr),
            Mu = (["bNum"], Mu),
            MuErr = (["bNum"], MuErr),
            KolFitInt = (["bNum"], KolInt),
            KolFitIntErr = (["bNum"], KolIntErr),
            ISRMisfit = (["bNum"], FitMisfit),
            eps = (["bNum"], epsMag),
            epsErr = (["bNum"], epsErr),
            J33 = (["bNum"], J33),
            epsLRInt = (["bNum"], epsFitInt),
            epsLRSlope = (["bNum"], epsFitSlope),
            epsLRR2val = (["bNum"], epsFitR2val),
            epsLRPval = (["bNum"], epsFitPval),
            epsLRSlopeErr = (["bNum"], epsFitSlopeErr),
            epsLRIntErr = (["bNum"], epsFitIntErr),
            L_Ozmidov = (['bNum'], LenOz),
            L_Kolmogorov = (['bNum'], LenKol),
            dUp = (['bNum'], dUp),
            timeStart = (['bNum'], time_start.data)
        ),
        coords=dict(
            bNum=(["bNum"], burstList.data),
            omega=(["omega"], omega.data),
            frequency=(["frequency"], Ftest.data),
            wavenumber=(["wavenumber"], k.data)
        ),
        attrs=dict(Description="Turbulent dissipation estimates with associated variables")
    )
    #Add Metadata to dataset and variables
    epsDS.attrs['Segment length (s)'] = nperseg/fs
    epsDS.attrs['Minimum ISR gap (radian frequency)'] = minGap*np.diff(omega)[0]
    epsDS.attrs['Noise frequency (radian frequency)'] = fNoise
    epsDS.attrs['Bad data ratio cutoff'] = badDataRatioCutoff
    epsDS.attrs['Pressure sensor height offset (m)'] = ZpOffset
    epsDS.attrs['Velocity transducer height offset (m)'] = ZvOffset

    epsDS['Su'].attrs['Description'] = 'Primary velocity spectra'
    epsDS['Su'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    epsDS['Sv'].attrs['Description'] = 'Secondary velocity spectra'
    epsDS['Sv'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    epsDS['Sw'].attrs['Description'] = 'Vertical velocity spectra'
    epsDS['Sw'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    epsDS['Sp'].attrs['Description'] = 'Pressure spectra converted to vertical velocity via linear wave theory'
    epsDS['Sp'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    
    epsDS['NoiseFloor'].attrs['Description'] = 'Index number where the noise floor is located in omega array'
    epsDS['WavePeak'].attrs['Description'] = 'Index number where the peak wave spectra is detected in omega array'

    epsDS['maxSw'].attrs['Description'] = 'Vertical velocity spectra at the maximum ISR frequency'
    epsDS['maxSw'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    epsDS['minSw'].attrs['Description'] = 'Vertical velocity spectra at the minimum ISR frequency'
    epsDS['minSw'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    
    epsDS['lowBound'].attrs['Description'] = 'Index number of the lower ISR boundary in omega array'
    epsDS['highBound'].attrs['Description'] = 'Index number of the upper ISR boundary in omega array'

    epsDS['Int'].attrs['Description'] = 'Intercept of ISR power curve fit'
    epsDS['Int'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    epsDS['IntErr'].attrs['Description'] = 'Error of the power curve intercept'
    epsDS['Mu'].attrs['Description'] = 'Slope of ISR power curve fit'
    epsDS['MuErr'].attrs['Description'] = 'Error of the power curve slope'
    epsDS['KolFitInt'].attrs['Description'] = 'Intercept of -5/3 slope fit'
    epsDS['KolFitInt'].attrs['Units'] = '[m/s]^2 * [rad/s]^-1'
    epsDS['KolFitIntErr'].attrs['Description'] = 'Error of the -5/3 slope fit intercept'
    epsDS['ISRMisfit'].attrs['Description'] = 'Mean square error between ISR power curve and -5/3 slope fits'
    
    epsDS['eps'].attrs['Description'] = 'Turbulent kinetic energy dissipation rate (epsilon)'
    epsDS['eps'].attrs['Units'] = 'm^2/s^3'
    epsDS['epsErr'].attrs['Description'] = 'Error in epsilon estimate'
    epsDS['epsErr'].attrs['Units'] = 'm^2/s^3'
    
    epsDS['J33'].attrs['Description'] = 'Wavenumber integral of upward velocity component'
    
    epsDS['epsLRInt'].attrs['Description'] = 'Intercept of epsilon linear regression'
    epsDS['epsLRIntErr'].attrs['Description'] = 'Intercept error of epsilon linear regression'
    epsDS['epsLRSlope'].attrs['Description'] = 'Slope of epsilon linear regression'
    epsDS['epsLRSlopeErr'].attrs['Description'] = 'Slope error of epsilon linear regression'
    epsDS['epsLRR2val'].attrs['Description'] = 'R-squared value of epsilon linear regression'
    epsDS['epsLRPval'].attrs['Description'] = 'P-value of epsilon linear regression'
    
    epsDS['L_Ozmidov'].attrs['Description'] = 'Ozmidov length scale'
    epsDS['L_Kolmogorov'].attrs['Description'] = 'Kolmogorov length scale'
    
    epsDS['dUp'].attrs['Description'] = 'Ratio of bad data within burst'
    epsDS['timeStart'].attrs['Description'] = 'Time that each burst begins with coordinates of bNum'

    return epsDS

#===============================================================================================================================
#==================================================== PLOTTING FUNCTIONS =======================================================
#===============================================================================================================================
def vecEpsPlotter(vecDS, tempDS, epsDS, timeFrame = None, saveFig = False, filename = None, returnBnum = False):
    
    if timeFrame is not None:
        tempDep = tempDS.sel(dict(time=slice(str(timeFrame[0]), str(timeFrame[-1])))).resample(time='20Min').mean()

        vecDS = vecDS.sel(dict(time=slice(str(timeFrame[0]), str(timeFrame[-1]))))
        gb = np.unique(vecDS.burst.where((vecDS.dPrimary < .25) & (vecDS.burst.isin(vecDS.BurstNum)), drop=True))
        advDep = vecDS.Primary.where(vecDS.BurstNum.isin(gb)).resample(time='20Min').mean()
        advDep = advDep.where(advDep.isnull()==False, drop=True)

        epsDep = epsDS.sel(dict(bNum=slice(gb[0], gb[-1]))).dropna(dim="bNum", how = 'all')
        epsDep = epsDep.where(epsDep.eps.isnull()==False, drop=True)
    
    else:
        tempDep = tempDS.sel(dict(time=slice(epsDS.timeStart.values[0], epsDS.timeStart.values[-1]))).resample(time='20Min').mean()

        gb = vecDS.burst.where(vecDS.dPrimary < .25, drop=True)
        advDep = vecDS.Primary.where(vecDS.BurstNum.isin(np.unique(gb))).resample(time='20Min').mean()
        advDep = advDep.where(advDep.isnull()==False, drop=True)

        epsDep = epsDS.where(epsDS.eps.isnull()==False).dropna(dim="bNum", how="all")

    #Initialize plot
    plt.figure(figsize = (20,16))
    
    # TEMPERATURE
    plt.subplot(311)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=0), 'r-', lw = 1)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=1), 'darkorange', lw = 1)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=2), 'y-', lw = 1)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=3), 'g-', lw = 1)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=4), 'indigo', lw = 1)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=5), 'b-', lw = 1)
    plt.plot(tempDep.time, tempDep.Temperature.isel(depth=6), 'k-', lw = 1)

    plt.ylabel("Temperature (Celsius)", fontsize=14)
    plt.margins(x=.01)
    plt.title('Temperature within SWC Kelp Forest Mooring')
    plt.legend(['2m','4m','6m','8m','9.1m', '9.4m', '9.7m'], loc = 'upper left')
    
    # PRIMARY VELOCITY
    plt.subplot(312)
    plt.plot(advDep.time, advDep, '.-b', label = 'ADV-U (20-min average)')
    plt.ylim(-.05,.05)
    #plt.legend(loc = 'upper left')
    plt.axhline(y=0, c='black', lw=2)
    plt.margins(x=.01)
    plt.ylabel('Velocity (m/s)', fontsize=14)
    plt.title('Primary velocity 1m Above Seafloor')
    
    # TURBULENT DISSIPATION
    plt.subplot(313)
    plt.yscale("log")
    plt.plot(epsDep.timeStart, epsDep.eps, '.-g', ms = 6, lw=1)
    lowerCI = epsDep.eps - epsDep.epsErr
    upperCI = epsDep.eps + epsDep.epsErr
    plt.fill_between(epsDep.timeStart, lowerCI, upperCI, color='green', alpha=0.3)
    plt.margins(x=.01)
    plt.ylabel(r'$\epsilon$ $(\frac{m^{2}}{s^{3}})$', fontsize=20)
    plt.xlabel('Date', fontsize=14)
    plt.title('TKE Dissipation Rate')
    
    if saveFig:
        plt.savefig(str(filename))
    if returnBnum:
        return np.unique(epsDep.bNum)
        
#======================================================================================================================================        
#Plot a specific burst and fit
def epsSpectraPlotter(epsData, burstNumber, saveFig = False, filename = None):

    ds = epsData.copy(deep=True)
    epsDS = ds.where(ds.bNum.isin(burstNumber), drop=True)

    lb = int(epsDS.lowBound.values[0])
    ub = int(epsDS.highBound.values[0])
    wp = int(epsDS.WavePeak.values[0])

    muFit = (epsDS.Int * epsDS.omega[lb:ub]**epsDS.Mu).values[0]
    kolFit = (epsDS.KolFitInt * epsDS.omega[lb:ub]**(-5/3)).values[0]
    Sp = epsDS.Sp.values[0][:wp]
    epsPval = epsDS.epsLRPval.values

    """if epsPval < .05:
        print('Nonzero Slope')
        ub2 = ub
        while epsPval < .05:
            ub2 = ub2 - 1
            alpha = 1.5
            Jlm = epsDS.J33.values[0]
            isrOmega = epsDS.omega[lb:ub2] #Radian frequency range
            S33 = epsDS.Sw.values[0][lb:ub2] #Vertical velocity spectra within ISR
            eps = ((S33 * (isrOmega**(5/3)))/(alpha * Jlm))**(3/2) #Returns array of eps estimates across ISR
            epsErr = np.sqrt(np.var(eps)/(len(eps)-1))
            epsMag = np.mean(eps)
            res = stats.linregress(isrOmega, eps)
            epsLR = ((res.slope*isrOmega) + res.intercept)
            epsPval = res.pvalue"""

    #Epsilon constants
    alpha = 1.5
    Jlm = epsDS.J33.values[0]

    #Estimate turbulent dissipation (Epsilon/eps)
    isrOmega = epsDS.omega[lb:ub] #Radian frequency range
    S33 = epsDS.Sw.values[0][lb:ub] #Vertical velocity spectra within ISR

    #Dissipation formula (Eq. A14 from Gerbi et al., 2009)
    eps = ((S33 * (isrOmega**(5/3)))/(alpha * Jlm))**(3/2) #Returns array of eps estimates across ISR
    epsMag = epsDS.eps.values[0]
    epsErr = np.sqrt(np.var(eps)/(len(eps)-1))

    lowerCI = eps.values - epsDS.epsErr.values[0]
    upperCI = eps.values + epsDS.epsErr.values[0]

    res = stats.linregress(isrOmega, eps)
    epsLR = ((res.slope*isrOmega) + res.intercept)

    plt.figure(figsize=(20,15))

    plt.subplot(211)
    plt.title('Burst #'+str(burstNumber)+ r' ($\epsilon_\mu$= '+str(epsDS.Mu.values[0])+' +- '+
             str(epsDS.MuErr.values[0])+')')
    plt.xlabel(r'$\omega$ $[\frac{rad}{s}]$', fontsize = 15)
    plt.ylabel(r'Sww $[\frac{m^{2}}{s^{4}}]$', fontsize = 15)

    plt.loglog(epsDS.omega, epsDS.Sw.values[0], '-k', lw = 1, label = 'Sww')
    plt.loglog(epsDS.omega[:wp], Sp, '-r', lw = 1, label = 'Gravity wave spectra')
    plt.loglog(epsDS.omega[lb:ub], kolFit, '-g', label="-5/3 fit", lw = 3)
    plt.loglog(epsDS.omega[lb:ub], muFit, '--y', label="Curve fit", lw = 2)
    plt.legend()

    plt.subplot(212)
    plt.title('Burst #'+str(burstNumber)+ r' ($\epsilon$ = '+str(epsMag)+' +- '+
             str(epsDS.epsErr.values[0])+')')
    plt.xlabel(r'$\omega$ $[\frac{rad}{s}]$', fontsize = 15)
    plt.ylabel(r'$\epsilon$ $(\frac{m^{2}}{s^{3}})$', fontsize=20)

    plt.plot(isrOmega, eps, '.-k', label = r'$\epsilon$ estimates')
    plt.fill_between(isrOmega, lowerCI, upperCI, color='gray', alpha=0.8)
    plt.axhline(y = epsMag, color = 'b', ls = '--', label = r'$\epsilon$ magnitude')
    plt.plot(isrOmega, epsLR.values, '-r', label = r'$\epsilon$ linear regression')
    plt.margins(x=.01)
    plt.legend(str(epsPval))
    
    if saveFig:
        plt.savefig(str(filename))
    
