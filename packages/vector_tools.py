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
from physoce import oceans as oc

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
    
    # SENSOR TESTS 
    # Tests for the quality of sensor parameters found on the .sen file
    
    testCounter = 0
    
    # Battery voltage test
    BatVolt_flag = xr.where(ds.BatVolt >= 9.6, 1, 3) # xr.where(condition, value if true, value if false)
    ds['BatVolt_Flag'] = (["time_sen"], BatVolt_flag.values)
    ds['BatVolt_Flag'].attrs['description'] = '(Pass) Battery voltage >= 9.6 volts; (Suspect) < 9.6 volts'
    testCounter = testCounter + 1
    
    # Compass Heading test
    Heading_flag = xr.where((ds.Heading >= 0) & (ds.Heading <= 360), 1, 4)
    ds['Heading_flag'] = (["time_sen"], Heading_flag.values)
    ds['Heading_flag'].attrs['description'] = '(Pass) Heading from 0-360 degrees; (Fail) Outside of 0-360 degrees'
    testCounter = testCounter + 1
    
    # Soundspeed test
    SoundSpeed_flag = xr.where((ds.SoundSpeed >= 1493) & (ds.SoundSpeed <= 1502), 1, 4)
    ds['SoundSpeed_flag'] = (["time_sen"], SoundSpeed_flag.values)
    ds['SoundSpeed_flag'].attrs['description'] = '(Pass) SoundSpeed from 1493-1502 m/s; (Fail) Outside of range'
    testCounter = testCounter + 1
    
    # Tilt test
    #Tilt_flag = Tilt_flag + xr.where(, 1, 4)
    #Tilt_flag = Tilt_flag + xr.where( < 5, 0, 4)
    Tilt_flag = xr.where((np.abs(ds.Roll) > 5) | (np.abs(ds.Pitch) > 5), 4, 1)
    ds['Tilt_flag'] = (["time_sen"], Tilt_flag.values)
    ds['Tilt_flag'].attrs['description'] = '(Pass) Pitch or Roll <= 5 degrees; (Fail) > 5 degrees'
    testCounter = testCounter + 1
    
    # Checksum tests
    CheckSum_flag = xr.where(ds.ChecksumSen == 0, 1, 4)
    ds['CheckSumSen_flag'] = (["time_sen"], CheckSum_flag.values)
    ds['CheckSumSen_flag'].attrs['description'] = '(Pass) Checksum = 0 ; (Fail) Checksum != 0'
    testCounter = testCounter + 1
    
    #senFlagQartod = senFlagQartod + xr.where(sen_flag <= 1, 1, 0)
    #senFlagQartod = senFlagQartod + xr.where((sen_flag > 1) & (sen_flag < 4), 3, 0)
    #senFlagQartod = senFlagQartod + xr.where(sen_flag >= 4, 4, 0)
  
    print('Flagging data')
    dat_flag = xr.zeros_like(ds.East) # Same shape as .dat data arrays
    datFlagQartod = xr.zeros_like(dat_flag)
    
    burst_diff = np.diff(ds.BurstNum, axis = 0, prepend = 0)
    burst_diff[0] = 0
    
    min_depth = int(np.mean(ds.Pressure) - (np.std(ds.Pressure)*2))
                       
    testCounter = 0                   
    
    # Checksum tests 
    CheckSumDat_flag = xr.where(ds.ChecksumDat == 0, 1, 4)
    ds['CheckSumDat_flag'] = (["time"], CheckSumDat_flag.values)
    ds['CheckSumDat_flag'].attrs['description'] = '(Pass) Checksum = 0 ; (Fail) Checksum != 0'
    testCounter = testCounter + 1
                       
    # Pressure test
    Pressure_flag = xr.where(ds.Pressure >= min_depth, 1, 3)
    ds['Pressure_flag'] = (["time"], Pressure_flag.values)
    ds['Pressure_flag'].attrs['description'] = '(Pass) Pressure >= above the minimum depth ; (Fail) Pressure below the minimum depth of deployment'
    testCounter = testCounter + 1
                       
    # SNR test
    #dat_flag = dat_flag + xr.where((ds.Snr1 < 10), 9, 0) # Full failure 
    #dat_flag = dat_flag + xr.where((ds.Snr2 < 10), 9, 0) 
    #dat_flag = dat_flag + xr.where((ds.Snr3 < 10), 9, 0)
    SNR_flag = xr.where((ds.Snr1 < 10)|(ds.Snr2 < 10)|(ds.Snr3 < 10), 4, 1)
    ds['SNR_flag'] = (["time"], SNR_flag.values)
    ds['SNR_flag'].attrs['description'] = '(Pass) SNR >= 10 dB ; (Fail) SNR < 10 dB'
    testCounter = testCounter + 1
                       
    # Beam correlation test          
    #dat_flag = dat_flag + xr.where((ds.Corr1 >= 70), 0, 0) # Full pass condition
    #dat_flag = dat_flag + xr.where((ds.Corr1 < 70) & (ds.Corr1 > 50), 3, 0) # Not ideal but acceptable
    #dat_flag = dat_flag + xr.where((ds.Corr1 <= 50), 9, 0) # Full failure 
    
    #dat_flag = dat_flag + xr.where((ds.Corr2 >= 70), 0, 0) 
    #dat_flag = dat_flag + xr.where((ds.Corr2 < 70) & (ds.Corr2 > 50), 3, 0) 
    #dat_flag = dat_flag + xr.where((ds.Corr2 <= 50), 9, 0) 
    
    #dat_flag = dat_flag + xr.where((ds.Corr3 >= 70), 0, 0) 
    #dat_flag = dat_flag + xr.where((ds.Corr3 < 70) & (ds.Corr3 > 50), 3, 0) 
    #dat_flag = dat_flag + xr.where((ds.Corr3 <= 50), 9, 0)
    
    Corr_flag = xr.where((ds.Corr1 < 60)|(ds.Corr2 < 60)|(ds.Corr3 < 60), 4, 0)
    Corr_flag = Corr_flag + xr.where((((ds.Corr1 >= 60)&(ds.Corr1 < 70))|
                                       ((ds.Corr2 >= 60)&(ds.Corr2 < 70))|
                                       ((ds.Corr3 >= 60)&(ds.Corr3 < 70)))&
                                       (Corr_flag != 4), 3, 0)
    Corr_flag = Corr_flag + xr.where(((ds.Corr1 >= 70)|(ds.Corr2 >= 70)|(ds.Corr3 >= 70))&
                                       ((Corr_flag != 4) & (Corr_flag != 3)), 1, 0)
    ds['Corr_flag'] = (["time"], Corr_flag.values)
    ds['Corr_flag'].attrs['description'] = '(Pass) Corr >= 70% ; (Suspect) Corr < 70% and >= 60%; (Fail) Corr < 60%'
    
    
    testCounter = testCounter + 1
    
    # Horizontal velocity test
    # For East-West
    #dat_flag = dat_flag + xr.where(np.abs(ds.East) >= 3, 4, 0)
    #dat_flag = dat_flag + xr.where((np.abs(ds.East) < 3) & (np.abs(ds.East) >= 1), 3, 0)
    #dat_flag = dat_flag + xr.where(np.abs(ds.East) < 1, 0, 0)
    
    East_flag = xr.where(np.abs(ds.East) >= 2, 4, 0)
    East_flag = East_flag + xr.where((np.abs(ds.East) < 2) & (np.abs(ds.East) >= 1), 3, 0)
    East_flag = East_flag + xr.where(np.abs(ds.East) < 1, 1, 0)
    ds['East_flag'] = (["time"], East_flag.values)
    ds['East_flag'].attrs['description'] = '(Pass) East < 1m/s ; (Suspect) East >= 1m/s and < 2m/s; (Fail) East >= 2m/s'
    
    testCounter = testCounter + 1                 
    
    # For North-South
    #dat_flag = dat_flag + xr.where(np.abs(ds.North) >= 3, 4, 0)
    #dat_flag = dat_flag + xr.where((np.abs(ds.North) < 3) & (np.abs(ds.North) >= 1), 3, 0)
    #dat_flag = dat_flag + xr.where(np.abs(ds.North) < 1, 0, 0)
    
    North_flag = xr.where(np.abs(ds.North) >= 2, 4, 0)
    North_flag = North_flag + xr.where((np.abs(ds.North) < 2) & (np.abs(ds.North) >= 1), 3, 0)
    North_flag = North_flag + xr.where(np.abs(ds.North) < 1, 1, 0)
    ds['North_flag'] = (["time"], North_flag.values)
    ds['North_flag'].attrs['description'] = '(Pass) North < 1m/s ; (Suspect) North >= 1m/s and < 2m/s; (Fail) North >= 2m/s'
    
    testCounter = testCounter + 1 
          
    # Vertical velocity test
    #dat_flag = dat_flag + xr.where(np.abs(ds.Up) >= 2, 4, 0)
    #dat_flag = dat_flag + xr.where((np.abs(ds.Up) < 2) & (np.abs(ds.Up) >= 1), 3, 0)
    #dat_flag = dat_flag + xr.where(np.abs(ds.Up) < 1, 0, 0)
    
    Up_flag = xr.where(np.abs(ds.Up) >= 1, 4, 0)
    Up_flag = Up_flag + xr.where((np.abs(ds.Up) < 1) & (np.abs(ds.Up) >= .5), 3, 0)
    Up_flag = Up_flag + xr.where(np.abs(ds.Up) < .5, 1, 0)
    ds['Up_flag'] = (["time"], Up_flag.values)
    ds['Up_flag'].attrs['description'] = '(Pass) Up < .5m/s ; (Suspect) Up >= .5m/s and < 1m/s; (Fail) Up >= 1m/s'
    
    testCounter = testCounter + 1 
      
    # u, v, w rate of change test
    # For East-west (u)
    du = np.diff(ds.East, axis = 0, prepend = 0) 
    du[0] = 0
    EastAcceleration_flag = xr.where((np.abs(du) >= 2) & (burst_diff==0), 4, 0)
    EastAcceleration_flag = EastAcceleration_flag + xr.where((np.abs(du) < 2) & (np.abs(du) >= 1) & (burst_diff==0), 3, 0)
    EastAcceleration_flag = EastAcceleration_flag + xr.where((np.abs(du) < 1) & (burst_diff==0), 1, 0) 
    ds['EastAcceleration_flag'] = (["time"], EastAcceleration_flag)
    ds['EastAcceleration_flag'].attrs['description'] = '(Pass) Eastern acceleration < 1m/s^2; (Suspect) >= 1m/s^2 and < 2m/s; (Fail) >= 2m/s'
    testCounter = testCounter + 1 
    
    # For North-South (v)
    dv = np.diff(ds.North, axis = 0, prepend = 0) 
    dv[0] = 0
    #dat_flag = dat_flag + xr.where((np.abs(dv) >= 2) & (burst_diff==0), 4, 0) 
    #dat_flag = dat_flag + xr.where((np.abs(dv) < 2) & (np.abs(dv) >= 1) & (burst_diff==0), 3, 0)
    #dat_flag = dat_flag + xr.where((np.abs(dv) < 1) & (np.abs(dv) >= .25) & (burst_diff==0), 1, 0)
    
    NorthAcceleration_flag = xr.where((np.abs(du) >= 2) & (burst_diff==0), 4, 0)
    NorthAcceleration_flag = NorthAcceleration_flag + xr.where((np.abs(du) < 2) & (np.abs(du) >= 1) & (burst_diff==0), 3, 0)
    NorthAcceleration_flag = NorthAcceleration_flag + xr.where((np.abs(du) < 1) & (burst_diff==0), 1, 0) 
    ds['NorthAcceleration_flag'] = (["time"], NorthAcceleration_flag)
    ds['NorthAcceleration_flag'].attrs['description'] = '(Pass) Northern Acceleration < 1m/s^2; (Suspect) >= 1m/s^2 and < 2m/s^2; (Fail) >= 2m/s^2'
    testCounter = testCounter + 1 

    # For vertical (w)
    dw = np.diff(ds.Up, axis = 0, prepend = 0) 
    dw[0] = 0
    #dat_flag = dat_flag + xr.where((np.abs(dw) >= 1) & (burst_diff==0), 4, 0) # Magnitudes of vertical velocity are typically smaller than horizontal velocity, so thresholds are reduced
    #dat_flag = dat_flag + xr.where((np.abs(dw) < 1) & (np.abs(dw) >= .5) & (burst_diff==0), 3, 0) 
    #dat_flag = dat_flag + xr.where((np.abs(dw) < .5) & (np.abs(dw) >= .15) & (burst_diff==0), 1, 0)
    
    UpAcceleration_flag = xr.where((np.abs(du) >= 2) & (burst_diff==0), 4, 0)
    UpAcceleration_flag = UpAcceleration_flag + xr.where((np.abs(du) < 2) & (np.abs(du) >= 1) & (burst_diff==0), 3, 0)
    UpAcceleration_flag = UpAcceleration_flag + xr.where((np.abs(du) < 1) & (burst_diff==0), 1, 0) 
    ds['UpAcceleration_flag'] = (["time"], UpAcceleration_flag)
    ds['UpAcceleration_flag'].attrs['description'] = '(Pass) Upward Acceleration < 1m/s^2; (Suspect) >= 1m/s^2 and < 2m/s^2; (Fail) >= 2m/s^2'
    
    testCounter = testCounter + 1 
       
    # Current speed test
    CSPD_flag = xr.where(ds.CSPD < 3, 1, 3)
    ds['CSPD_flag'] = (["time"], CSPD_flag.values)
    ds['CSPD_flag'].attrs['description'] = '(Pass) Horizontal current speed < 3m/s ; (Suspect) >= 3m/s'
    testCounter = testCounter + 1 
   
    # Current speed and direction rate of change tests
    dCSPD = np.diff(ds.CSPD, axis = 0, prepend = 0)
    dCSPD[0] = 0
    #dat_flag = dat_flag + xr.where((np.abs(dCSPD) >= 4) & (burst_diff==0), 4, 0)
    #dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 4) & (np.abs(dCSPD) >= 1) & (burst_diff==0), 3, 0)
    #dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 1) & (np.abs(dCSPD) >= .25) & (burst_diff==0), 1, 0)
    
    CSPDAcceleration_flag = xr.where((np.abs(dCSPD) >= 3) & (burst_diff==0), 4, 0)
    CSPDAcceleration_flag = CSPDAcceleration_flag + xr.where((np.abs(dCSPD) < 3) & (np.abs(dCSPD) >= 1) & (burst_diff==0), 3, 0)
    CSPDAcceleration_flag = CSPDAcceleration_flag + xr.where((np.abs(dCSPD) < 1) & (burst_diff==0), 1, 0)
    ds['CSPDAcceleration_flag'] = (["time"], CSPDAcceleration_flag)
    ds['CSPDAcceleration_flag'].attrs['description'] = '(Pass) Current speed acceleration < 1m/s^2; (Suspect) >= 1m/s^2 and < 3m/s^2; (Fail) >= 3m/s^2'
    
    testCounter = testCounter + 1 

    # For current direction
    dCDIR = np.diff(ds.CDIR, axis = 0, prepend = 0)
    dCDIR[0] = dCDIR[0] - dCDIR[0]
    CDIR_flag = xr.where((np.abs(dCDIR) >= 135) & (np.abs(dCDIR) <= 225)& (burst_diff==0), 4, 0)
    CDIR_flag = CDIR_flag + xr.where((np.abs(dCDIR) < 135) & (np.abs(dCDIR) >= 30) & (burst_diff==0), 3, 0)
    CDIR_flag = CDIR_flag + xr.where((np.abs(dCDIR) <= 330) & (np.abs(dCDIR) > 225) & (burst_diff==0), 3, 0)
    CDIR_flag = CDIR_flag + xr.where((CDIR_flag != 4)&(CDIR_flag != 3), 1, 0)
    ds['CDIRdelta_flag'] = (["time"], CDIR_flag)
    ds['CDIRdelta_flag'].attrs['description'] = '(Pass) Current speed direction change < 30 degrees; (Suspect) >= 30 and < 135 degrees; (Fail) >= 135 degrees'
    
    
    testCounter = testCounter + 1 
    
    # Add the new flag data array to the existing dataset
    #flagAvg = (dat_flag)/testCounter
    #datFlagQartod = datFlagQartod + xr.where(flagAvg > 4, 9, 0)
    #datFlagQartod = datFlagQartod + xr.where((flagAvg <= 4) & (flagAvg > 3), 4, 0)
    ##datFlagQartod = datFlagQartod + xr.where((flagAvg <= 3) & (flagAvg > 1), 3, 0)
    #datFlagQartod = datFlagQartod + xr.where((flagAvg <= 1), 1, 0)
    #ds['DataFlag'] = (["time"], datFlagQartod.values)
    #ds['DataFlag'].attrs['Flag score'] = '[1, 3, 4, 9]'
    #ds['DataFlag'].attrs['Grade definition'] = '1 = Pass, 3 = Suspect, 4 = Non-critical Fail, 9 = Critical Fail'
    ds.attrs['Flag score'] = '[1, 3, 4]'
    ds.attrs['Grade definition'] = 'Pass = 1; Suspect = 3; Fail = 4'
    ds.attrs['Flag description'] = 'Flag grading system is based on QARTOD quality control parameters and tests in Nortek ADV user manual'
    
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
        theta, major, minor = ts.princax(v.East, v.North) # theta = angle, major = SD major axis (U), SD minor axis (V)
        secondary, primary = ts.rot(v.East, v.North, -theta-90) #Theta is set so that primary is on y-axis and secondary is x-axis
        orig = np.logical_and(v.EOrig.values,v.NOrig.values)
    else:
        # Using principle component analysis to rotate data to uncorrelated axes
        theta, major, minor = ts.princax(v.U, v.V) # theta = angle, major = SD major axis (U), SD minor axis (V)
        secondary, primary = ts.rot(v.U, v.V, -theta-90) #Theta is set so that primary is on y-axis and secondary is x-axis
        orig = np.logical_and(v.UOrig,v.VOrig)

    # store angle of rotation as attribute and new vectors as primary and secondary
    # velocities in dataset
    v.attrs['Theta'] = -theta-90
    v['Primary'] = ('time',primary)
    v.Primary.attrs['Std.'] = major
    v.Primary.attrs['units'] = 'm/s'
    v['Secondary'] = ('time',secondary)
    v.Secondary.attrs['Std.'] = minor
    v.Secondary.attrs['units'] = 'm/s'
    
    v['PrimaryOrig'] = ('time',orig)
    v['PrimaryOrig'].attrs['Description'] = 'Original data mask for primary velocity component that combines EOrig and NOrig'
    v['PrimaryOrig'].attrs['Conditions'] = 'True = unaltered data; False = Despiked or removed data due to failing qc tests'
    v['SecondaryOrig'] = ('time',orig)
    v['SecondaryOrig'].attrs['Description'] = 'Original data mask for secondary velocity componentthat combines EOrig and NOrig'
    v['SecondaryOrig'].attrs['Conditions'] = 'True = unaltered data; False = Despiked or removed data due to failing qc tests'

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
        v['EOrig'].attrs['Description'] = 'True = original data point; False = despiked data point (unoriginal)'
        v['NOrig'] = ('time',np.ones(v.time.shape,dtype='bool'))
        v['NOrig'].attrs['Description'] = 'True = original data point; False = despiked data point (unoriginal)'
        v['UpOrig'] = ('time',np.ones(v.time.shape,dtype='bool'))
        v['UpOrig'].attrs['Description'] = 'True = original data point; False = despiked data point (unoriginal)'

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
    
    corrMask = np.where((v.Corr1 < corrCutoff) |
                        (v.Corr2 < corrCutoff) |
                        (v.Corr3 < corrCutoff), False, True)

    #find where snr cutoff fails
    index[np.logical_or(np.logical_or(v.Snr1.values < snrCutoff, \
                                 v.Snr2.values < snrCutoff), \
                                 v.Snr3.values < snrCutoff)] = True
    
    snrMask = np.where((v.Snr1 < snrCutoff) |
                       (v.Snr2 < snrCutoff) |
                       (v.Snr3 < snrCutoff), False, True)

    #if angle is too big, assume we don't have trustworthy tilt information
    if np.abs(angleCutoff) < 2*np.pi:

        #get pitch and roll info on correct timestep
        pitch = v.Pitch.interp(time_sen = v.time.values).values
        roll = v.Roll.interp(time_sen=v.time.values).values
        
        #convert pitch and roll to total tilt
        tilt = np.arctan(np.sqrt(np.tan(pitch*2*np.pi/360)**2+np.tan(roll*2*np.pi/360)**2))

        #find where angle cutoff fails
        index[tilt>angleCutoff] = True
        
        tiltMask = np.where(tilt > angleCutoff, False, True)
        
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
    
    #True = passes all qc tests; False = fails one or more qc tests
    qcMaskInitial = np.logical_and(corrMask, snrMask)
    qcMaskFinal = np.logical_and(qcMaskInitial, tiltMask)
    
    #v['qcMask'] = (['time'], qcMaskFinal)
    #v['qcMask'].attrs['Description'] = 'True = data that passes correlation/snr/tilt tests; False = data that fails'
    
    return v, qcMaskFinal
#===============================================================================================================================
def ProcessVec(data, badSections, corrCutoff, snrCutoff, tiltCutoff, reverse=False,expand = True,lp = 1/20,expSize = 0.01,expEnd = 0.95):
    #initial processing of adv data. eliminates bad data due to
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

    hz = data.attrs['Sampling rate']

    print('cleaning vector based on correlation and snr cutoffs')

    #use correlation and snr cutoffs to clean data
    data_temp, qcMask = cleanVec(data,corrCutoff=corrCutoff,snrCutoff=snrCutoff,angleCutoff=tiltCutoff)
    

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
    
    #Combine Despike masks with qc masks
    data_temp_All['EOrig'] = (['time'], np.logical_and(data_temp_All.EOrig.values, qcMask))
    data_temp_All['EOrig'].attrs['Description'] = 'Original data mask for eastern velocity'
    data_temp_All['EOrig'].attrs['Conditions'] = 'True = unaltered data; False = Despiked or removed data due to failing qc tests'
    data_temp_All['NOrig'] = (['time'], np.logical_and(data_temp_All.NOrig.values, qcMask))
    data_temp_All['NOrig'].attrs['Description'] = 'Original data mask for northern velocity'
    data_temp_All['NOrig'].attrs['Conditions'] = 'True = unaltered data; False = Despiked or removed data due to failing qc tests'
    data_temp_All['UpOrig'] = (['time'], np.logical_and(data_temp_All.UpOrig.values, qcMask))
    data_temp_All['UpOrig'].attrs['Description'] = 'Original data mask for vertical velocity'
    data_temp_All['UpOrig'].attrs['Conditions'] = 'True = unaltered data; False = Despiked or removed data due to failing qc tests'
    
    print('rotating vectors')
    
    data_temp_All = rotateVec(data_temp_All)
    
    if reverse:
        print('reversing direction so primary is in flooding direction')
        data_temp_All.Primary[:] = -data_temp_All.Primary[:]
        data_temp_All.Secondary[:] = -data_temp_All.Secondary[:]
    
    print('Adding bad data ratios to dataset')
    #Add bad data ratios to the dataset
    n = data_temp_All.attrs['Samples per burst'] #Total number of samples per burst
    EbadPoints = data_temp_All.BurstNum.where(data_temp_All.EOrig==False).dropna(dim='time', how = 'all')
    NbadPoints = data_temp_All.BurstNum.where(data_temp_All.NOrig==False).dropna(dim='time', how = 'all')
    UpbadPoints = data_temp_All.BurstNum.where(data_temp_All.UpOrig==False).dropna(dim='time', how = 'all')
    PbadPoints = data_temp_All.BurstNum.where(data_temp_All.PrimaryOrig==False).dropna(dim='time', how = 'all')
    SbadPoints = data_temp_All.BurstNum.where(data_temp_All.SecondaryOrig==False).dropna(dim='time', how = 'all')
    
    data_temp_All.coords["burst"] = (["burst"], data_temp_All.BurstCounter.values)
    data_temp_All["EOrigRatio"] = (['burst'], np.unique(EbadPoints, return_counts=True)[1]/n)
    data_temp_All["EOrigRatio"].attrs['Description'] = 'Ratio of bad data points out of max number of samples'
    data_temp_All["NOrigRatio"] = (['burst'], np.unique(NbadPoints, return_counts=True)[1]/n)
    data_temp_All["NOrigRatio"].attrs['Description'] = 'Ratio of bad data points out of max number of samples'
    data_temp_All["UpOrigRatio"] = (['burst'], np.unique(UpbadPoints, return_counts=True)[1]/n)
    data_temp_All["UpOrigRatio"].attrs['Description'] = 'Ratio of bad data points out of max number of samples'
    data_temp_All["PrimOrigRatio"] = (['burst'], np.unique(PbadPoints, return_counts=True)[1]/n)
    data_temp_All["PrimOrigRatio"].attrs['Description'] = 'Ratio of bad data points out of max number of samples'
    data_temp_All["SecOrigRatio"] = (['burst'], np.unique(SbadPoints, return_counts=True)[1]/n)
    data_temp_All["SecOrigRatio"].attrs['Description'] = 'Ratio of bad data points out of max number of samples'
    
    print('Adding raw velocities to dataset')
    #Re-add back the original, unaltered velocities to the dataset for future gap repairs
    data_temp_All['EastRaw'] = data.East
    data_temp_All['EastRaw'].attrs['Description'] = 'The unaltered eastern velocity'
    data_temp_All['EastRaw'].attrs['Units'] = 'm/s'

    data_temp_All['NorthRaw'] = data.North
    data_temp_All['NorthRaw'].attrs['Description'] = 'The unaltered northern velocity'
    data_temp_All['NorthRaw'].attrs['Units'] = 'm/s'

    data_temp_All['UpRaw'] = data.Up
    data_temp_All['UpRaw'].attrs['Description'] = 'The unaltered vertical velocity'
    data_temp_All['UpRaw'].attrs['Units'] = 'm/s'
    
    #Add the final metadata to the dataset
    data_temp_All.attrs['Description'] = 'Despiked data with low correlation/snr and high tilt points removed' 
    data_temp_All.attrs['despike_lp_freq (hz)'] = lp
    data_temp_All.attrs['despike_cutoff_expansion_fraction'] = expSize
    data_temp_All.attrs['despike_cutoff_expansion_densityChange_end_condition'] = expEnd

    return data_temp_All
    
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
def InterpAvg_ADV(ADVdata, interpTimeDelta = 1, reversePrincVel = False):
    
    ds = ADVdata.copy(deep=True)
    fs = ds.attrs['Sampling rate']

    #First interpolate all gaps <= 1s
    print('Interpolating gaps <= 1s')
    delta = timedelta(seconds=interpTimeDelta + (2/fs)) #Maximum gap of 1 second (2/fs ensures that points right at the margin are taken into account)

    #Linearly interpolate across gaps up to a limit denoted by time delta
    ds['East'] = ds.East.interpolate_na(dim="time", method="linear", use_coordinate=True, max_gap = delta)
    ds['North'] = ds.North.interpolate_na(dim="time", method="linear", use_coordinate=True, max_gap = delta)
    ds['Up'] = ds.Up.interpolate_na(dim="time", method="linear", use_coordinate=True, max_gap = delta)

    print('Averaging gaps > 1s')
    #Average the rest of the longer gaps
    EgapTimes = ds.time.where(ds.East.isnull()==True).dropna(dim='time',how='all')
    EgapIDX = np.where(ds.time.isin(EgapTimes))[0]
    EgapDiff = np.diff(EgapIDX, prepend = EgapIDX[0])
    EgapRanges = np.split(EgapIDX, np.where(EgapDiff > 1)[0])
    for i in np.arange(len(EgapRanges)):
        ds.East[EgapRanges[i][0]:EgapRanges[i][-1]+1] = ds.EastRaw[EgapRanges[i][0]:EgapRanges[i][-1]+1].mean()
    
    NgapTimes = ds.time.where(ds.North.isnull()==True).dropna(dim='time',how='all')
    NgapIDX = np.where(ds.time.isin(NgapTimes))[0]
    NgapDiff = np.diff(NgapIDX, prepend = NgapIDX[0])
    NgapRanges = np.split(NgapIDX, np.where(NgapDiff > 1)[0])
    for i in np.arange(len(NgapRanges)):
        ds.North[NgapRanges[i][0]:NgapRanges[i][-1]+1] = ds.NorthRaw[NgapRanges[i][0]:NgapRanges[i][-1]+1].mean()
    
    UpgapTimes = ds.time.where(ds.Up.isnull()==True).dropna(dim='time',how='all')
    UpgapIDX = np.where(ds.time.isin(UpgapTimes))[0]
    UpgapDiff = np.diff(UpgapIDX, prepend = UpgapIDX[0])
    UpgapRanges = np.split(UpgapIDX, np.where(UpgapDiff > 1)[0])
    for i in np.arange(len(UpgapRanges)):
        ds.Up[UpgapRanges[i][0]:UpgapRanges[i][-1]+1] = ds.UpRaw[UpgapRanges[i][0]:UpgapRanges[i][-1]+1].mean()
    
    #Recalculating variables that are effected by altered velocities
    print('Recalculating CSPD and CDIR')
    ds['CSPD'] = (["time"], np.sqrt((ds.East.values**2) + (ds.North.values**2)))
    ds['CDIR'] = (["time"], vec_angle(ds.East, ds.North).data)
    
    print('Recalculating principle velocities')
    dsFinal = rotateVec(ds)
    
    if reversePrincVel:
        print('reversing direction so primary is in flooding direction')
        dsFinal.Primary[:] = -dsFinal.Primary[:]
        dsFinal.Secondary[:] = -dsFinal.Secondary[:]
        
    return dsFinal

#===============================================================================================================================
#=============================================== DISSIPATION ESTIMATION FUNCTIONS ==============================================
#===============================================================================================================================

def JlmIntegral(ADVdata, n):
    
    #Make copy of dataset
    ds = ADVdata.copy(deep=True)
    
    #Retrieve list of burst numbers of dataset
    burst_list = np.unique(ds.BurstNum.values)
    
    #Initialize Jlm arrays for each dimension and burst
    J11_arr = np.empty(len(burst_list))
    J22_arr = np.empty(len(burst_list))
    J33_arr = np.empty(len(burst_list))

    #Initialize all variables theta, phi, and R within boundaries a to b
    #as specified by the Gerbi et al., 2009 equation
    phi = np.linspace(0, 2*np.pi, n)
    theta = np.linspace(0, np.pi, n)
    
    #Reshap theta array to form a 'vertical' array
    thetaRS = np.linspace(0, np.pi, n).reshape(n,1)
    
    #Initialize empty arrays to hold the final integral with theta
    fTheta_11 = np.empty(len(theta))
    fTheta_22 = np.empty(len(theta))
    fTheta_33 = np.empty(len(theta))
    
    #Cycle through each burst in the dataset
    for b in enumerate(burst_list):

        print('Burst #: '+str(b[1]))
    
        #Call East, North, Vertical velocities from dataset
        E = ds.East.where(ds.BurstNum.isin(b[1]), drop = True).values
        N = ds.North.where(ds.BurstNum.isin(b[1]), drop = True).values
        W = ds.Up.where(ds.BurstNum.isin(b[1]), drop = True).values
        
        #Rotate the individual burst along principle axis
        #Ensures that the assumptions of the equation are being followed
        theta_rot, major, minor = ts.princax(E, N) # theta = angle, major = SD major axis (U), SD minor axis (V)
        V, U = ts.rot(E, N, -theta_rot-90)

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

        #Create 2D arrays of R0, G, and Plm over all values of phi and theta
        R0 = ((ubar/usig) * (np.sin(thetaRS)*np.cos(phi))) + ((vbar/vsig) * (np.sin(thetaRS)*np.sin(phi)))

        G = np.sqrt((np.sin(thetaRS)**2) * (((np.cos(phi)/usig)**2) + ((np.sin(phi)/vsig)**2)) + ((np.cos(thetaRS)/wsig)**2))

        P_11 = (1/(G**2))*((((np.sin(thetaRS)**2)*(np.sin(phi)**2))/vvar)+((np.cos(thetaRS)**2)/wvar))
        P_22 = (1/(G**2))*((((np.sin(thetaRS)**2)*(np.cos(phi)**2))/uvar)+((np.cos(thetaRS)**2)/wvar))
        P_33 = ((np.sin(thetaRS)/G)**2) * (((np.cos(phi)/usig)**2) + ((np.sin(phi)/vsig)**2))
        
        #Use scipy.quad_vec to integrate R over two dimensions for all values of phi and theta
        #R itself is integrated from 0 - 10, which is where the curve becomes asymptotic
        fR = quad_vec(lambda R: (R**(2/3))*np.exp(-(((R0-R)**2)/2)), 0, 10)[0]
        #Creates 2D array of R (a function of theta and phi) where each column is
        #a value of phi and each row pertains to a value of theta
        
        #Caculate second integral of the function (phi)
        fPhi_11 = (G**(-11/3))*np.sin(thetaRS)*P_11 * fR
        fPhi_22 = (G**(-11/3))*np.sin(thetaRS)*P_22 * fR
        fPhi_33 = (G**(-11/3))*np.sin(thetaRS)*P_33 * fR
        
        #Integrate the array through all values of theta
        for i in enumerate(theta): 

            fTheta_11[i[0]] = np.trapz(fPhi_11[i[0]], phi)
            fTheta_22[i[0]] = np.trapz(fPhi_22[i[0]], phi)
            fTheta_33[i[0]] = np.trapz(fPhi_33[i[0]], phi)

        # Evaluate the final integral of fTheta and use it to find J_lm    
        J_11 = (1/(2*((2*np.pi)**(3/2))))*(1/(usig*vsig*wsig)) * np.trapz(fTheta_11, theta)
        J_22 = (1/(2*((2*np.pi)**(3/2))))*(1/(usig*vsig*wsig)) * np.trapz(fTheta_22, theta)
        J_33 = (1/(2*((2*np.pi)**(3/2))))*(1/(usig*vsig*wsig)) * np.trapz(fTheta_33, theta)
        
        #Add values to arrays
        J11_arr[b[0]] = J_11
        J22_arr[b[0]] = J_22
        J33_arr[b[0]] = J_33
        
        
    print('Adding wavespace integral to dataset')
    ds['J11'] = (['burst'],J11_arr)
    ds['J11'].attrs['Description'] = 'The wavespace integral used in equation A13 of Gerbi et al. (2009)'
    ds['J11'].attrs['Direction'] = 'Primary'
    
    ds['J22'] = (['burst'],J22_arr)
    ds['J22'].attrs['Description'] = 'The wavespace integral used in equation A13 of Gerbi et al. (2009)'
    ds['J22'].attrs['Direction'] = 'Secondary' 
    
    ds['J33'] = (['burst'],J33_arr)
    ds['J33'].attrs['Description'] = 'The wavespace integral used in equation A13 of Gerbi et al. (2009)'
    ds['J33'].attrs['Direction'] = 'Vertical'
    
    return ds

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
def power_law(x, a, b):
    return a*np.power(x, b)

#===============================================================================================================================
def kol_law(x,a):
    return a*np.power(x,(-5/3))

#================================================================================================================================
def ADV_spectraDS(ADVdata, TEMPdata, segLength=60, window='hann', pNoisefloorFreq=.4, vNoisefloorFreq=3.1,
                  wtNoisefloorFactor=12,wtPeakwaveFactor=1.1, wtSlope = -4, minimumGap = 1):
    '''
    A function that takes the fully quality controlled and repaired ADV xarray dataset and an associated density time series, and uses
    both to form a comprehensive dataset of wave parameters and turbulent dissipation estimates.
    
    Methods for calculating wave parameters and estimating turbulent dissipation from ADV velocity and pressure spectra are
    used from Feddersen (2010), Gerbi et al. (2009), Jones and Monosmith (2007, 2008), Trowbrige and Elgar (2001), Wheeler and Giddings (2023)
    
    INPUTS:
    ADVdata: Fully despiked, filtered, and repaired ADV dataset generated by ADV_dat.ipynb
    TEMPdata: A time series of density that spans the length of the ADV dataset (must have 'time' dimension)
    segLength: The duration (in seconds) of data used in each segment of the FFT computed 
                by scipy.welch (this is multiplied by sample frequency)
    window: The desired filter window for the spectra (defaults to hann)
    pNoisefloorFreq: The frequency where the pressure spectra noise floor begins in Hz (defaults to .4)
    vNoisefloorFreq: The frequency where the vertical velocity noise floor begins in Hz (defaults to 3.1)
    wtNoisefloorFactor: This factor controls how much higher the wave band cutoff frequency must be from 
                        the pressure noise floor magnitude (defaults to 12 as specified in Jones and Monosmith (2007))
    wtPeakwaveFactor: This factor controls how much lower the wave band cutoff frequency must be from
                        the peak wave frequency (defaults to 1.1 as specified in Jones and Monosmith (2007))
    wtSlope: The slope of the model wavetail after the surface wave cutoff frequency (defaults to -4)
    minimumGap: The minimum range of the ISR that the function uses to estimate turbulent dissipation9
    
    
    OUTPUTS:
    xarray dataset with wave, inertial subrange, and turbulent dissipation parameters
    '''
    
    print('Initializing dataset information and arrays')
    #Make a copy of the datasets to prevent accidental modification of original data
    ADVds = ADVdata.copy(deep=True)
    TEMPds = TEMPdata.copy(deep=True)

    #Instrument data
    fs = ADVds.attrs['Sampling rate']                    #The sampling rate of the instrument (32 Hz in this case)
    ZpOffset = ADVds.attrs['Pressure sensor height (m)'] #Height of ADV pressure sensor off the seafloor (m)
    ZvOffset = ADVds.attrs['Velocity sample height (m)'] #Height of velocity measurement off the seafloor (m)
    n = ADVds.attrs['Samples per burst']                 #Number of data points within the burst (must be consistent for all bursts)
    nperseg = segLength * fs                             #The segment length of the FFT ensemble average

    #The frequency at which the noise floor dominates the signal
    #Estimated through observations made in the raw spectral data
    pressureFn = pNoisefloorFreq #Frequency of pressure sensor noise floor cutoff in Hz
    velocityFn = vNoisefloorFreq #Frequency of velocity transducer noise floor cutoff in Hz

    #Wave band cutoff frequency factors used in Jones and Monosmith (2007)
    fnFactor = wtNoisefloorFactor
    fpFactor = wtPeakwaveFactor
    
    #List of burst numbers to categorize adv samples
    burstList = np.unique(ADVds.burst)
    
    #Initialize arrays to hold all variables generated by the function
    SpModel_arr = np.empty((len(burstList),int(nperseg/2)+1))  #Raw pressure spectra with omega^-4 tail
    Sn_arr = np.empty((len(burstList),int(nperseg/2)+1))       #Sea level spectra
    SwPrime_arr = np.empty((len(burstList),int(nperseg/2)+1))  #Vertical velocity from pressure spectra
    Rho_arr = np.empty(len(burstList))                         #Mean density for each burst
    Z_arr = np.empty(len(burstList))                           #Seafloor depth for each burst
    Fc_arr = np.empty(len(burstList))                          #Position of gravity wave cutoff in the frequency array
    wtCutoff_arr = np.empty(len(burstList))                    #Position of omega^-4 wavetail cutoff in frequency array
    pNoisefloor_arr = np.empty(len(burstList))                 #Magnitude of pressure noisefloor
    vNoisefloor_arr = np.empty(len(burstList))                 #Magnitude of vertical velocity noisefloor
    Hrms_arr = np.empty(len(burstList))                        #Root mean square wave height
    Hs_arr = np.empty(len(burstList))                          #Significant wave height
    Tavg_arr = np.empty(len(burstList))                        #Average wave period
    Tpeak_arr = np.empty(len(burstList))                       #Peak wave period
    waveOrbital_arr = np.empty(len(burstList))                 #Estimated wave orbital velocity
    CSPD_arr = np.empty(len(burstList))                        #Mean current velocity for each burst
    CSPDstd_arr = np.empty(len(burstList))                     #Standard deviation of current velocity per burst

    #Initialize arrays for variables inertial subrange (ISR)
    isrUpper = np.empty(len(burstList))  #Upper boundary of ISR based on the best fit
    isrLower = np.empty(len(burstList))  #Lower boundary of ISR based on the best fit
    Mu = np.empty(len(burstList))        #Slope of ISR fit
    MuErr = np.empty(len(burstList))     #Error of slope
    Int = np.empty(len(burstList))       #Intercept of ISR fit
    IntErr = np.empty(len(burstList))    #Error of slope
    KolInt = np.empty(len(burstList))    #Intercept of -5/3 fit
    KolIntErr = np.empty(len(burstList)) #Error of -5/3 intercept
    FitMisfit = np.empty(len(burstList)) #Misfit between ISR slope and -5/3
    maxSw = np.empty(len(burstList))     #The highest magnitude of vertical velocity spectra in the ISR
    minSw = np.empty(len(burstList))     #Lowest magnitude in the ISR

    #Initialize arrays to hold all dissipation estimates and eps variables
    epsMag = np.empty(len(burstList))         #Mean of eps values over isr
    epsErr = np.empty(len(burstList))         #Error in eps estimate
    epsFitInt = np.empty(len(burstList))      #Intercept of eps estimate linear regression (LR) model
    epsFitSlope = np.empty(len(burstList))    #LR Slope
    epsFitR2val = np.empty(len(burstList))    #LR R2-value
    epsFitPval = np.empty(len(burstList))     #LR P-value
    epsFitSlopeErr = np.empty(len(burstList)) #Error of LR slope
    epsFitIntErr = np.empty(len(burstList))   #Error of LR intercept
    validFits = np.empty(len(burstList))      #Number of ISR fits that pass qc tests
    validLB = np.empty(len(burstList))        #The lowest valid frequency of the ISR
    validUB = np.empty(len(burstList))        #The highest valid frequency of the ISR
    epsKDE = np.empty(len(burstList))         #Eps estimate from Gaussian KDE fit
    R_ratio = np.empty(len(burstList))
    
    #Constants for estimating epsilon
    alpha = 1.5 # Kolomogorov constant
    
    #Jlm wavenumber integrals from Gerbi et al. (2009)
    J11 = ADVds.J11.where(ADVds.burst.isin(burstList), drop=True).values
    J22 = ADVds.J22.where(ADVds.burst.isin(burstList), drop=True).values
    J33 = ADVds.J33.where(ADVds.burst.isin(burstList), drop=True).values
    
    #Ratios of unoriginal data to total samples for future quality control
    PrimOrigRatio = ADVds.PrimOrigRatio.where(ADVds.BurstCounter.isin(burstList), drop=True).values
    SecOrigRatio = ADVds.SecOrigRatio.where(ADVds.BurstCounter.isin(burstList), drop=True).values
    UpOrigRatio = ADVds.UpOrigRatio.where(ADVds.BurstCounter.isin(burstList), drop=True).values
    
    #List of start times to be used as a dimension in the final dataset
    TimeStart = ADVds.time_start.where(ADVds.BurstCounter.isin(burstList), drop=True).values
    burstTime = ADVds.time.values.reshape(len(burstList),n) #Array of all segmented times
    
    #A list of burst number indexed by start time
    BurstCounter = ADVds.BurstCounter.where(ADVds.time_start.isin(TimeStart), drop=True).values

    #All velocity and pressure components are converted to 2D arrays for easy retrieval by the function
    #Dimensions (# of bursts, total samples)
    burstCSPD = ADVds.CSPD.values.reshape(len(burstList),n)     #Array of current speed
    burstU = ADVds.Primary.values.reshape(len(burstList),n)     #Horizontal velocities
    burstV = ADVds.Secondary.values.reshape(len(burstList),n) 
    burstW = ADVds.Up.values.reshape(len(burstList),n)          #Vertical velocity
    burstCSPD = ADVds.CSPD.values.reshape(len(burstList),n)     #Current speed
    burstPRaw = ADVds.Pressure.values.reshape(len(burstList),n) #Raw adv head pressure in dbar
    burstPPascal = burstPRaw * 10000                            #Adv head pressure conbverted from dBar to Pascals

    #Calculate PSD of both pressure time series, segment averaging the spectra via a specified segment length and filter window
    #All time series of pressure are linearly detrended to account for tidal effects
    FpRaw, SpRaw = welch(burstPRaw, fs = fs, nperseg = nperseg, window=window, detrend = 'linear') #Raw pressure spectra
    FpPa, SpPa = welch(burstPPascal, fs = fs, nperseg = nperseg, window=window, detrend = 'linear') #Raw pressure spectra (pascals)
    Fw, Sw = welch(burstW, fs = fs, nperseg = nperseg, window=window, detrend = 'linear') #Vertical velocity spectra
    Fu, Su = welch(burstU, fs = fs, nperseg = nperseg, window=window, detrend = 'linear') #Horizontal velocity spectra
    Fv, Sv = welch(burstV, fs = fs, nperseg = nperseg, window=window, detrend = 'linear')

    #Convert to spectra to radian frequency
    SpRawOmega = SpRaw / (2*np.pi)
    SpPaOmega = SpPa / (2*np.pi)
    SwOmega = Sw / (2*np.pi)
    SuOmega = Su / (2*np.pi)
    SvOmega = Sv / (2*np.pi)
    
    #Create seperate array of period spectrum from the frequency arrays generated by scipy.welch
    T = 1/FpRaw 

    #Define the minimum gap of the ISR in terms of the frequency array
    #I.E. a 'minGap' of 60 means that every 60 positions in the frequency array
    #pertains to 1Hz, if 1Hz is the desired minimum gap specific in the function parameters
    minGap = int((minimumGap*2*np.pi)/np.diff(Fw*(2*np.pi))[0])
    
    print('Calculating spectra and estimating dissipation for...')
    #Begin the wave parameter and dissipation estimation loop
    for b in enumerate(burstList):
        
        #Each time b[0] is used, that determines the burst being evaluated numerically
        #b[1] is the actual burst value in terms of the ADVdata index
        #I.E. If bursts 3,4,6,10,12 were being evaluated:
        #b[0] = 0,1,2,3,4
        #b[1] = 3,4,6,10,12
        
        print('Burst #'+str(b[0]+1)+' of '+str(len(burstList)))

        #Calculate mean current speed and its standard deviation for the burst
        CSPD_arr[b[0]] = burstCSPD[b[0]].mean()
        CSPDstd_arr[b[0]] = burstCSPD[b[0]].std()

        #Generate aspectrum of radian frequency (omega) and wavenumber (k)
        Rho = TEMPdata.Rho.sel(time = slice(burstTime[0][0],burstTime[0][-1])).mean() #Mean density during the burst
        if Rho.isnull()==True:
            try:
                Rho = Rho_arr[b[0]-1]
                Rho_arr[b[0]] = Rho
            except:
                Rho = 1025
                Rho_arr[b[0]] = Rho
        else:
            Rho_arr[b[0]] = Rho.values

        #wavedisp function uses period (from frequency) and water depth (h) to calculate omega, k, and phase speed of waves
        #Convert frequency to radian frequency and wavenumber
        g = -9.8 # Gravity
        z = (burstPPascal[b[0]]/(Rho.values*g)) - ZpOffset #Depth (m): the recorded pressure converted to meters of seawater
        Z_arr[b[0]] = z.mean() #Mean z is recorded in a seperate array
        H = np.mean(-z) #Sea level height (m): mean pressure detected by the pressure sensor plus the height of sensor from the bottom
        Zp = np.mean(z + ZpOffset) #Depth of pressure sensor (m)
        Zv = np.mean(z + ZvOffset) #Depth of velocity sensor (m): Sea level height plus the height of the velocity transducers from the bottom

        #Calculate radian frequency and wavenumber using wavedisp function
        omega,k,Cph,Cg = wavedisp(T, H)

        #Use linear wave theory to convert pressure spectra to vertical velocity and sea level spectra
        #Sea level spectra is calculated using methods from Jones and Monosmith (2007)

        #Locate the noisefloor of the pressure spectra
        fnIDX = np.where(FpRaw <=pNoisefloorFreq)[0][-1] #The array indices where the noise floor begins in the RAW pressure spectrum
        pNoisefloor = np.mean(SpRawOmega[b[0]][fnIDX:]) #Average magnitude of noise floor

        #Record the pressure noisefloor magnitude in seperate array
        pNoisefloor_arr[b[0]] = pNoisefloor

        #Now find the gravity wave cutoff frequency as specific by Jones and Monosmith (2007)
        fp = omega[np.where(SpRawOmega[b[0]] == np.nanmax(SpRawOmega[b[0]][:fnIDX]))[0][0]] #The peak frequency of the raw pressure spectrum
        fcIDX = np.where((omega[:fnIDX] > (fp * fpFactor)) & (SpRawOmega[b[0]][:fnIDX] > (pNoisefloor*fnFactor)))[0][-1] #The array indices of the final cutoff frequency
        Fc_arr[b[0]] = int(fcIDX)

        #Converting the pressure spectrum using linear wave theory
        Kp = np.empty(len(k))
        p_prime = np.empty(len(omega))
        w_prime = np.empty(len(omega))
        
        #Using wavenumber from the wavedisp function, the wave dispersion equations can be used
        #to determine scaling factors for converting the pressure spectra at depth
        for i in enumerate(k):
            Kp[i[0]] = (np.cosh(i[1]*(Zp+H))/np.cosh(i[1]*H))**2
            p_prime[i[0]] = (Rho*(-g))*(np.cosh(i[1]*(Zp+H))/np.cosh(i[1]*H))
            w_prime[i[0]] = (-omega[i[0]])*(np.sinh(i[1]*(Zv+H)))/(np.sinh(i[1]*H))

        #Scale factor for converting pressure spectra (pascals) to vertical velocity spectra
        scaleFactor = w_prime**2 / p_prime**2
        SwPrime = SpPaOmega[b[0]] * scaleFactor
        
        #Use conversion equation to estimate sea level spectrum
        Sn = (SpRawOmega[b[0]]/Kp) 

        #Since pressure spectra hits a noise floor early in the spectrum, a model
        #tail can be formed using an omega^-4 slope for a more complete spectrum
        
        #Calculate the omega^-4 model tail for the sea level spectra
        SnWavetail = (Sn[fcIDX]/(omega[fcIDX]**(wtSlope)))*(omega**(wtSlope))

        #Convert the wavetail to the raw pressure spectrum
        SpWavetail = SnWavetail*Kp

        #Convert wavetail to Sw'w' spectra as well
        SwPrimeWavetail = ((SnWavetail*Kp)*1e8) * scaleFactor

        #Determine where the model wavetail should end by using the noise floor of the vertical velocity spectra
        velFnIDX = np.where(omega == (vNoisefloorFreq*(2*np.pi)))[0][0]  #Frequency of where the velocity noise floor starts in the omega array
        vNoisefloor = np.mean(SwOmega[b[0]][velFnIDX:]) #Magnitude of vertical velocity spectra noisefloor

        #Record the vertical velocity noisefloor magnitude in seperate array
        vNoisefloor_arr[b[0]] = vNoisefloor

        #Define the wavetail cutoff as the frequency where the wavetail goes below the vertical velocity noise floor
        try:
            wavetailCutoff = np.where((SwPrimeWavetail <= vNoisefloor)&(omega>fp))[0][1] #Where the tail should end
        except:
            wavetailCutoff = np.where(omega==np.pi)[0][0] #If wavetail runs into error, just use conservative
                                                          #cutoff of pi rad/s (.5 Hz)

        #Record the omega array indices for the wavetail cutoff
        wtCutoff_arr[b[0]] = int(wavetailCutoff)

        #Make seperate modelled spectra that include the wavetail
        SpModel = np.empty(len(SpRawOmega[b[0]]))*np.nan #Initialize empty "model" array for pressure spectra
        SpModel[:fcIDX+1] = SpRawOmega[b[0]][:fcIDX+1] #Fill model up to the surface wave cutoff frequency with the normal spectra
        SpModel[fcIDX:wavetailCutoff+1] = SpWavetail[fcIDX:wavetailCutoff+1] #Fill the model with the wavetail after the wave cutoff      frequency and go until the wavetail cutoff frequency
        #Record the model pressure spectra in seperate array 
        SpModel_arr[b[0]] = SpModel

        #Do the same process for the sea level spectra (Snn)
        SnModel = np.empty(len(Sn))*np.nan
        SnModel[:fcIDX+1] = Sn[:fcIDX+1]
        SnModel[fcIDX:wavetailCutoff+1] = SnWavetail[fcIDX:wavetailCutoff+1]
        Sn_arr[b[0]] = SnModel

        #And vertical velocity from pressure spectra (Sw'w')
        SwPrimeModel = np.empty(len(SwPrime))*np.nan
        SwPrimeModel[:fcIDX+1] = SwPrime[:fcIDX+1]
        SwPrimeModel[fcIDX:wavetailCutoff+1] = SwPrimeWavetail[fcIDX:wavetailCutoff+1]
        SwPrime_arr[b[0]] = SwPrimeModel

        #Calculate wave height and period using moment integrals from Jones and Monosmith (2007)
        m0 = np.trapz((omega[1:wavetailCutoff+1]**0)*SnModel[1:wavetailCutoff+1], omega[1:wavetailCutoff+1])
        m2 = np.trapz((omega[1:wavetailCutoff+1]**2)*SnModel[1:wavetailCutoff+1], omega[1:wavetailCutoff+1])

        Hrms_arr[b[0]] = 2*np.sqrt(2*m0) #Root mean square wave height
        Hs_arr[b[0]] = 4 * np.sqrt(m0)   #Significant wave height
        Tavg_arr[b[0]] = (2*np.pi) * np.sqrt(m0/m2) #Average wave period
        Tpeak_arr[b[0]] = 1/(omega[np.where(SnModel==np.nanmax(SnModel))[0][0]]/(2*np.pi)) #Peak wave period
        waveOrbital_arr[b[0]] = oc.ubwave(Hs_arr[b[0]], Tavg_arr[b[0]], H) #Estimate wave orbital velocity at depth "H"
        
        #Using some of the wave parameters and the modelled spectra, turbulent dissipation can be estimated
        #First the inertial subrange (ISR) must be identified
        
        #Starting boundaries are the end of the surface wave band and the beginning of the noise floor
        lfc = int(wavetailCutoff)-2 #Generally where the gravity wave band begins to diverge from vertical velocity
        ufc = velFnIDX #Where the velocity noise floor starts

        #Initialize an array of all frequency combinations within lfc and ufc
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
        testHighBound = np.empty(len(bounds))
        testMinFit = np.empty(len(bounds))
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
        epsList = list()

        #Estimate Turbulent Kinetic energy dissipation rate (epsilon)

        #Test all combinations of ISR ranges and store the results
        for i in np.arange(0,len(bounds)):
            try:
                #Generate power law fit using the current set of bounds
                #Scipy.curve_fit finds the best fit parameters using least squares error
                pars, cov = curve_fit(f=power_law, xdata=omega[bounds[i][0]:bounds[i][1]],
                                          ydata=SwOmega[b[0]][bounds[i][0]:bounds[i][1]], p0=[0, 0], bounds=(-np.inf, np.inf), maxfev=10000)

                #Using the same concept, use the boundaries to fit a power law with fixed -5/3 slope
                #to represent Kolmogorov's law
                pars2, cov2 = curve_fit(f=kol_law, xdata=omega[bounds[i][0]:bounds[i][1]],
                                        ydata=SwOmega[b[0]][bounds[i][0]:bounds[i][1]], maxfev=10000)

                #Calculate the misfit between the dynamic fit and Kolmogorov's law using the mean square error
                muFit = pars[0] * (omega[bounds[i][0]:bounds[i][1]]**pars[1])
                testMinFit[i] = muFit[-1] #The magnitude of the fit at the high boundary
                kolFit = pars2[0] * (omega[bounds[i][0]:bounds[i][1]]**(-5/3))
                muDiff[i] = np.abs(pars[1]+(5/3)) #Misfit of the slope compared to -5/3

                testMinSw[i] = SwOmega[b[0]][bounds[i][0]] #Spectra at lower boundary

                #Record all of the relevant variables for ISR and epsilon quality control 
                testInt[i] = pars[0] #Fit intercept
                testIntErr[i] = np.sqrt(np.diag(cov))[0] #intercept error to 90% confidence level
                testMu[i] = pars[1] #Fit slope
                testMuErr[i] = np.sqrt(np.diag(cov))[1] #Slope error to 90% confidence level

                testKolInt[i] = pars2[0] #Intercept from -5/3 fit
                testKolIntErr[i] = np.sqrt(np.diag(cov2))[0] #Intercept error from -5/3 fit to 90% confidence level
                
                ### INITIAL ESTIMATE ###
                #Estimate turbulent dissipation (Epsilon/eps)
                highBound = bounds[i][1]
                testHighBound[i] = bounds[i][1]
                isrOmega = omega[bounds[i][0]:highBound]    #Frequency range of the proposed ISR
                S33 = SwOmega[b[0]][bounds[i][0]:highBound] #Vertical velocity spectra within ISR

                #Dissipation formula (Eq. A14 from Gerbi et al., 2009)
                eps = ((S33 * (isrOmega**(5/3)))/(alpha * J33[b[0]]))**(3/2) #Returns array of eps estimates across ISR
                
                testFitMisfit[i] = np.sum((kolFit-S33)**2)/(len(S33)-1)  #Mean SE between spectrum and -5/3 fit

                #Fit a linear regression to eps estimates
                res = stats.linregress(isrOmega, eps)
                
                testEpsMag[i] = np.mean(eps) #Mean value of eps for the entire burst
                testEpsErr[i] = np.sqrt(np.var(eps)/(len(eps)-1)) #Calculate error of the epsilon measurements from variance about the mean
                                                                  #Method from Feddersen (2010)
                testEpsFitInt[i] = res.intercept #Linear regression intercept
                testEpsFitSlope[i] = res.slope #Linear regression slope
                testEpsFitR2val[i] = res.rvalue**2 #R2 value of linear regression
                testEpsFitPval[i] = res.pvalue #P-value of linear regression (used for qc)
                testEpsFitSlopeErr[i] = res.stderr #Error of linear regression slope
                testEpsFitIntErr[i] = res.intercept_stderr #Error of linear regression intercept
                epsList.append(eps)

            #If curve_fit can't find a proper best fit line, populate test arrays with 99999 as error values
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

        #Using all ISR and epsilon estimates from this burst, run them all through a series of
        #quality control tests to determine which frequency ranges and epsilon values are potentially valid

        #Noise floor test from Gerbi et al. (2009) requires that Sww/2 at 2pi is > the noisefloor
        #Gerbi fits to a fixed range of 2pi-10pi, which can go below the noise floor, whereas this study uses a dynamic range
        #To try an mimic Gerbi's noise floor test, we reject any fits that dip below the noise floor
        noiseFlag = xr.where((testMinFit > vNoisefloor), 0, 1)
        noiseFlagSum = np.where(noiseFlag==0)[0]
        if len(noiseFlagSum)==0:
            minSw[b[0]] = np.nan;isrLower[b[0]] = np.nan;maxSw[b[0]] = np.nan; isrUpper[b[0]] = np.nan;
            Int[b[0]] = np.nan;IntErr[b[0]] = np.nan; Mu[b[0]] = np.nan; MuErr[b[0]] = np.nan;
            KolInt[b[0]] = np.nan; KolIntErr[b[0]] = np.nan; FitMisfit[b[0]] = np.nan;epsMag[b[0]] = np.nan;
            epsErr[b[0]] = np.nan;epsFitInt[b[0]] = np.nan; epsFitSlope[b[0]] = np.nan;epsFitR2val[b[0]] = np.nan;
            epsFitPval[b[0]] = np.nan;epsFitSlopeErr[b[0]] = np.nan;epsFitIntErr[b[0]] = np.nan;epsKDE[b[0]] = np.nan;
            validFits[b[0]] = 0;validLB[b[0]] = np.nan;validUB[b[0]] = np.nan
            print('Failed noise floor test')
            continue

        #Curve fit intercept test from Jones and Monosmith (2008)
        #Magnitude of dissipation must be greater than the error of the -5/3 fit intercept
        intFlag = xr.where(testEpsMag > testKolIntErr, 0, 1)
        intFlagSum = np.where(intFlag==0)[0]
        if len(intFlagSum)==0:
            minSw[b[0]] = np.nan;isrLower[b[0]] = np.nan;maxSw[b[0]] = np.nan; isrUpper[b[0]] = np.nan;
            Int[b[0]] = np.nan;IntErr[b[0]] = np.nan; Mu[b[0]] = np.nan; MuErr[b[0]] = np.nan;
            KolInt[b[0]] = np.nan; KolIntErr[b[0]] = np.nan; FitMisfit[b[0]] = np.nan;epsMag[b[0]] = np.nan;
            epsErr[b[0]] = np.nan;epsFitInt[b[0]] = np.nan; epsFitSlope[b[0]] = np.nan;epsFitR2val[b[0]] = np.nan;
            epsFitPval[b[0]] = np.nan;epsFitSlopeErr[b[0]] = np.nan;epsFitIntErr[b[0]] = np.nan;epsKDE[b[0]] = np.nan;
            validFits[b[0]] = 0;validLB[b[0]] = np.nan;validUB[b[0]] = np.nan
            print('Failed intercept test')
            continue

        #Spectra fit slope test from Feddersen (2010)
        #The dynamic slope fit must be within an acceptable range of -5/3
        lowMu = testMu - (2*testMuErr) - .06
        highMu = testMu + (2*testMuErr) + .06
        slopeFlag = xr.where((lowMu < (-5/3)) & (highMu > (-5/3)), 0, 1)
        slopeFlagSum = np.where(slopeFlag==0)[0]
        if len(slopeFlagSum)==0:
            minSw[b[0]] = np.nan;isrLower[b[0]] = np.nan;maxSw[b[0]] = np.nan; isrUpper[b[0]] = np.nan;
            Int[b[0]] = np.nan;IntErr[b[0]] = np.nan; Mu[b[0]] = np.nan; MuErr[b[0]] = np.nan;
            KolInt[b[0]] = np.nan; KolIntErr[b[0]] = np.nan; FitMisfit[b[0]] = np.nan;epsMag[b[0]] = np.nan;
            epsErr[b[0]] = np.nan;epsFitInt[b[0]] = np.nan; epsFitSlope[b[0]] = np.nan;epsFitR2val[b[0]] = np.nan;
            epsFitPval[b[0]] = np.nan;epsFitSlopeErr[b[0]] = np.nan;epsFitIntErr[b[0]] = np.nan;epsKDE[b[0]] = np.nan;
            validFits[b[0]] = 0;validLB[b[0]] = np.nan;validUB[b[0]] = np.nan
            print('Failed Feddersen slope test')
            continue

        #Spectra slope test from Wheeler and Giddings (2023)
        #Uses the error of the fitted slope and difference between -5/3 to check  
        #if it's within the 97.5th percentile of a standard normal distribution
        normSlopeFlag = xr.where((muDiff/testMuErr) < 1.960, 0, 1)
        normSlopeFlagSum = np.where(normSlopeFlag==0)[0]
        if len(normSlopeFlagSum)==0:
            minSw[b[0]] = np.nan;isrLower[b[0]] = np.nan;maxSw[b[0]] = np.nan; isrUpper[b[0]] = np.nan;
            Int[b[0]] = np.nan;IntErr[b[0]] = np.nan; Mu[b[0]] = np.nan; MuErr[b[0]] = np.nan;
            KolInt[b[0]] = np.nan; KolIntErr[b[0]] = np.nan; FitMisfit[b[0]] = np.nan;epsMag[b[0]] = np.nan;
            epsErr[b[0]] = np.nan;epsFitInt[b[0]] = np.nan; epsFitSlope[b[0]] = np.nan;epsFitR2val[b[0]] = np.nan;
            epsFitPval[b[0]] = np.nan;epsFitSlopeErr[b[0]] = np.nan;epsFitIntErr[b[0]] = np.nan;epsKDE[b[0]] = np.nan;
            validFits[b[0]] = 0;validLB[b[0]] = np.nan;validUB[b[0]] = np.nan
            print('Failed W&G slope test')
            continue

        #Epsilon linear regression test from Feddersen (2010)
        #Uses a linear regression between omega and epsilon to check 
        #that the slope is indistinguishable from 0
        linRegFlag = xr.where(testEpsFitPval >= .05, 0, 1)

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
            epsKDE[b[0]] = np.nan
            validFits[b[0]] = 0
            validLB[b[0]] = np.nan
            validUB[b[0]] = np.nan
            #LenOz[b[0]] = np.nan
            #LenKol[b[0]] = np.nan
            print('No valid fits')
            continue

        #Using all of the ISR and epsilon estimates that pass the tests
        #the best fit ISR is chosen by the smallest least squares error 
        #of the -5/3 fit from the spectrum itself (Jones and Monosmith, 2008)
        bestFit = finalFlag[testFitMisfit[finalFlag].argmin()] #Returns the index value of the best fit line

        #Populate final arrays with each variable pertaining to the "bestFit" index
        minSw[b[0]] = SwOmega[b[0]][bounds[bestFit][0]]
        isrLower[b[0]] = bounds[bestFit][0]
        maxSw[b[0]] = SwOmega[b[0]][int(testHighBound[bestFit])]
        isrUpper[b[0]] = int(testHighBound[bestFit])

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

        #Gaussian kde epsilon estimate
        finalEpsList = np.array(())
        for ind in finalFlag:
            finalEpsList = np.append(finalEpsList,epsList[ind])
        gkde_obj = stats.gaussian_kde(finalEpsList)
        x_pts = np.linspace(finalEpsList.min(), finalEpsList.max(), len(finalFlag))
        estimated_pdf = gkde_obj.evaluate(x_pts)
        testEpsKDE = x_pts[estimated_pdf.argmax()]
        epsKDE[b[0]] = finalEpsList[np.abs(finalEpsList-testEpsKDE).argmin()]
        
        #Final variables to track estimation parameters
        validFits[b[0]] = len(finalFlag)
        finalBounds = np.array(bounds)[finalFlag]
        validLB[b[0]] = finalBounds[:,0].min()
        finalLB = finalBounds[:,0].min()
        validUB[b[0]] = finalBounds[:,1].max()
        finalUB = finalBounds[:,1].max()

        #Calculate the R-ratio for unity between horizontal and vertical spectra
        #A psuedo-test for evaluating isotropic turbulence assumption
        Suv = (SuOmega[b[0]]+SvOmega[b[0]])/2
        SuvNF = Suv[ufc:].mean() #Noisefloor of horizontal velocity
        SuvUB = np.where(Suv<=SuvNF)[0][0] #The frequency array indices where the noisefloor starts

        a = 12/21 #Constant term used in the R-unity ratio equation from Feddersen (2010)

        #If the noise floor frequency in Suv is higher than the highest ISR frequency
        #continue with R-ratio calculation as normal
        if SuvUB >= finalUB:
            omegaR = omega[finalLB:finalUB]**(5/3)
            SuvR = Suv[finalLB:finalUB]
            SwwR = SwOmega[b[0]][finalLB:finalUB]
            R_ratio[b[0]] = ((a * (omegaR*(SuvR-SuvNF)))/(omegaR*SwwR)).mean()
            
        elif ((SuvUB<finalUB)&(SuvUB>=finalLB)):
            omegaR = omega[finalLB:SuvUB]**(5/3)
            SuvR = Suv[finalLB:SuvUB]
            SwwR = SwOmega[b[0]][finalLB:SuvUB]
            R_ratio[b[0]] = ((a * (omegaR*(SuvR-SuvNF)))/(omegaR*SwwR)).mean()
            
        elif (SuvUB<finalLB):
            R_ratio[b[0]] = np.nan
        else:
            R_ratio[b[0]] = np.nan
        
        print(testEpsFitPval[bestFit],len(finalFlag),int(testHighBound[bestFit])-bounds[bestFit][0])

        #END OF THE LOOP

    #Create a new dataset with all relevant variables and epsilon values
    print('Creating Dataset')
    spectraDS = xr.Dataset(
        data_vars=dict(
            BurstCounter = (["time_start"], BurstCounter),
            Suu = (["time_start","omega"], SuOmega,
                   {'Description':'Primary velocity spectra','Units':'[m/s]^2 * [rad/s]^-1'}),
            Svv = (["time_start","omega"], SvOmega,
                  {'Description':'Secondary velocity spectra','Units':'[m/s]^2 * [rad/s]^-1'}),
            Sww = (["time_start","omega"], SwOmega,
                  {'Description':'Vertical velocity spectra','Units':'[m/s]^2 * [rad/s]^-1'}),
            Spp = (["time_start","omega"], SpRawOmega,
                  {'Description':'Pressure spectra from ADV head','Units':'[dBar]^2 * [rad/s]^-1'}),
            SppModel = (["time_start","omega"], SpModel_arr,
                       {'Description':'Pressure spectra with an f^-4 tail','Units':'[dBar]^2 * [rad/s]^-1'}),
            Snn = (["time_start","omega"], Sn_arr,
                  {'Description':'Sea level spectra estimated from pressure via linear wave theory with an f^-4 tail','Units':'[m]^2 * [rad/s]^-1'}),
            SwwPrime = (["time_start","omega"], SwPrime_arr,
                       {'Description':'Vertical velocity spectra estimated from pressure via linear wave theory with an f^-4 tail',
                        'Units':'[m/s]^2 * [rad/s]^-1'}),
            PressureNoisefloor = (["time_start"], pNoisefloor_arr,
                                 {'Description':'Magnitude of noisefloor for pressure spectrum','Units':'[dBar]^2 * [rad/s]^-1'}),
            VelocityNoisefloor = (["time_start"], vNoisefloor_arr,
                                 {'Description':'Magnitude of noisefloor for vertical velocity spectrum','Units':'[m/s]^2 * [rad/s]^-1'}),
            WavetailStart = (["time_start"], Fc_arr.astype(int),
                            {'Description':'Start of the f^-4 model','Units':'omega array index'}),
            WavetailEnd = (["time_start"], wtCutoff_arr.astype(int),
                          {'Description':'End of the f^-4 model','Units':'omega array index'}),
            Rho = (["time_start"], Rho_arr,
                  {'Description':'Average density of water column','Units':'[Kg/m^3]'}),
            Z = (["time_start"], Z_arr,
                {'Description':'Depth of seafloor','Units':'m'}),
            Hrms = (["time_start"], Hrms_arr,
                   {'Description':'Root mean square wave height','Units':'m'}),
            Hs = (["time_start"], Hs_arr,
                 {'Description':'Significant wave height','Units':'m'}),
            Tavg = (["time_start"], Tavg_arr,
                   {'Description':'Average wave period','Units':'s'}),
            Tpeak = (["time_start"], Tpeak_arr,
                    {'Description':'Peak wave period','Units':'s'}),
            WaveOrbitalVel = (["time_start"], waveOrbital_arr,
                             {'Description':'Wave orbital velocity estimate','Units':'[m/s]'}),
            WaveOrbitalAdvec = (["time_start"], waveOrbital_arr*Tavg_arr,
                                {'Description':'Estimate advection distance of wave orbital','Units':'m'}),
            CSPD = (["time_start"], CSPD_arr,
                   {'Description':'Mean current speed','Units':'[m/s]'}),
            CSPDstd = (["time_start"], CSPDstd_arr,
                      {'Description':'Std. current speed','Units':'[m/s]'}),
            J11 = (["time_start"], J11,
                  {'Description':'Wavenumber integral of primary velocity from eq. A13 of Gerbi et. al (2009)'}),
            J22 = (["time_start"], J22,
                  {'Description':'Wavenumber integral of secondary velocity from eq. A13 of Gerbi et. al (2009)'}),
            J33 = (["time_start"], J33,
                  {'Description':'Wavenumber integral of vertical velocity from eq. A13 of Gerbi et. al (2009)'}),
            PrimOrigRatio = (["time_start"], PrimOrigRatio,
                            {'Description':'Ratio of bad data:total data for primary velocity'}),
            SecOrigRatio = (["time_start"], SecOrigRatio,
                           {'Description':'Ratio of bad data:total data for secondary velocity'}),
            UpOrigRatio = (["time_start"], UpOrigRatio,
                          {'Description':'Ratio of bad data:total data for vertical velocity'}),
            MaxISRMag = (["time_start"], maxSw,{'Description':'Vertical velocity spectra at the maximum ISR frequency',
                                                'Units':'[m/s]^2 * [rad/s]^-1'}),
            MinISRMag = (["time_start"], minSw,{'Description':'Vertical velocity spectra at the minimum ISR frequency',
                                                'Units':'[m/s]^2 * [rad/s]^-1'}),
            LowBound = (["time_start"], isrLower,{'Description':'Index number of the lower ISR boundary in omega array based on best fit',
                                                          'Units':'omega array index'}),
            HighBound = (["time_start"], isrUpper,{'Description':'Index number of the upper ISR boundary in omega array based on best fit',
                                                           'Units':'omega array index'}),
            Int = (["time_start"], Int,{'Description':'Intercept of ISR power curve fit',
                                        'Units':'[m/s]^2 * [rad/s]^-1'}),
            IntErr = (["time_start"], IntErr,{'Description':'Error of the power curve intercept'}),
            Mu = (["time_start"], Mu,{'Description':'Slope of ISR power curve fit'}),
            MuErr = (["time_start"], MuErr,{'Description':'Error of the power curve slope'}),
            KolFitInt = (["time_start"], KolInt,{'Description':'Intercept of Kolmogorov law',
                                                 'Units':'[m/s]^2 * [rad/s]^-1'}),
            KolFitIntErr = (["time_start"], KolIntErr,{'Description':'Error of Kolmogorov law intercept'}),
            ISRMisfit = (["time_start"], FitMisfit,{'Description':'Mean square error between ISR power curve and Kolmogorov law'}),
            eps = (["time_start"], epsMag,{'Description':'Turbulent kinetic energy dissipation rate',
                                           'Units':'[m^2/s^3]'}),
            epsErr = (["time_start"], epsErr,{'Description':'Error in epsilon estimate',
                                              'Units':'[m^2/s^3]'}),
            epsLRInt = (["time_start"], epsFitInt,{'Description':'Intercept of epsilon linear regression'}),
            epsLRIntErr = (["time_start"], epsFitIntErr,{'Description':'Intercept error of epsilon linear regression'}),
            epsLRSlope = (["time_start"], epsFitSlope,{'Description':'Slope of epsilon linear regression'}),
            epsLRSlopeErr = (["time_start"], epsFitSlopeErr,{'Description':'Slope error of epsilon linear regression'}),
            epsLRR2val = (["time_start"], epsFitR2val,{'Description':'R-squared value of epsilon linear regression'}),
            epsLRPval = (["time_start"], epsFitPval,{'Description':'P-value of epsilon linear regression'}),
            epsKDE = (["time_start"], epsKDE,{'Description':'Turbulent kinetic energy dissipation rate',
                                           'Units':'[m^2/s^3]'}),
            ValidFits = (["time_start"], validFits,{'Description':'Number of ISR fits that pass all qc tests'}),
            ValidLB = (["time_start"], validLB,{'Description':'Lowest valid frequency of the ISR'}),
            ValidUB = (["time_start"], validUB,{'Description':'Highest valid frequency of the ISR'}),

            R_ratio = (['time_start'],R_ratio,{'Description':'The unity between the horizontal and vertical spectra'})
            ),
        coords=dict(
            time_start=(["time_start"], TimeStart),
            omega=(["omega"], omega),
        ),
        attrs=dict(Description="Velocity and pressure spectra with estimated wave variables")
    )

    #Add Metadata to dataset and variables
    spectraDS.attrs['Sampling rate (Hz)'] = ADVds.attrs['Sampling rate']
    spectraDS.attrs['Samples per burst'] = ADVds.attrs['Samples per burst']
    spectraDS.attrs['Pressure sensor height offset (m)'] = ADVds.attrs['Pressure sensor height (m)']
    spectraDS.attrs['Velocity transducer height offset (m)'] = ADVds.attrs['Velocity sample height (m)']
    spectraDS.attrs['Segment length (s)'] = segLength
    spectraDS.attrs['Filter window'] = window
    spectraDS.attrs['Pressure noise frequency (radian frequency)'] = pressureFn * (2*np.pi)
    spectraDS.attrs['Velocity noise frequency (radian frequency)'] = velocityFn * (2*np.pi)
    spectraDS.attrs['Noisefloor factor'] = fnFactor
    spectraDS.attrs['Wavepeak factor'] = fpFactor
    spectraDS.attrs['Minimum ISR gap (radian frequency)'] = minGap*np.diff(omega)[0]

    return spectraDS

    
