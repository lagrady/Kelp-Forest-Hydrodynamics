import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import xarray as xr
import gsw # https://teos-10.github.io/GSW-Python/gsw_flat.html
import vector_tools as vt

import matplotlib.patches as mpatches
from physoce import tseries as ts 
from scipy.signal import welch 
from scipy.stats import chi2 
from scipy import stats
from scipy.signal import periodogram
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from datetime import timedelta

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def vec_angle(x,y):
    '''
    A function used to find the direction of a vector from 0-360 degrees given the x and y component. Used to find the current speed direction in the datfile_to_ds function.
    
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
    fs = 32 # Vector sampled at 32Hz
    ds = vt.datfile_to_ds(datfile, vhdfile, fs)
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
            CDIR = (["time"], vt.vec_angle(dat['Velocity_East'].to_xarray(), dat['Velocity_North'].to_xarray())),
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
        attrs=dict(description="ADV data", lat=36.56195999999164, lon=-121.94174126537672, Sampling_Rate=fs, coords='ENU'),
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
    
    # DATA TESTS
    # Fixable failure (excluding checksum and correlation)= 36
    # Fixable failure (with acceptable correlations) = 41
    # Complete failure = 42
    # Tests performed on the recorded velocity and pressure data found on the .dat file
    
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
    dat_flag = dat_flag + xr.where((ds.Corr1 > 70), 0, 0) # Full pass condition
    dat_flag = dat_flag + xr.where((ds.Corr1 <= 70) & (ds.Corr1 > 50), 3, 0) # Not ideal but acceptable
    dat_flag = dat_flag + xr.where((ds.Corr1 <= 50), 9, 0) # Full failure 
    
    dat_flag = dat_flag + xr.where((ds.Corr2 > 70), 0, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr2 <= 70) & (ds.Corr2 > 50), 3, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr2 <= 50), 9, 0) 
    
    dat_flag = dat_flag + xr.where((ds.Corr3 > 70), 0, 0) 
    dat_flag = dat_flag + xr.where((ds.Corr3 <= 70) & (ds.Corr3 > 50), 3, 0) 
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
    ds['DataFlag'] = (["time"], datFlagQartod)
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
    
    ds['SenFlag'] = (["time_sen"], sen_flag)
    ds['SenFlag'].attrs['Flag score'] = '[1, 3, 4]'
    ds['SenFlag'].attrs['Grade definition'] = '1 = Pass, 3 = Suspect, 4 = Fail'
    ds['SenFlag'].attrs['description'] = 'Flag value based on internal sensor tests: battery, heading, pitch and roll, temperature, and soundspeed. Sampled at 1Hz.'
    
    return ds
    
#===============================================================================================================================
def fastResample(ds, timeString):
    # Following sequence of code originally found on https://stackoverflow.com/questions/64282393/how-can-i-speed-up-xarray-resample-much-slower-than-pandas-resample
    df_h = ds.to_dataframe().resample(str(timeString)).mean().dropna()  # what we want (quickly), but in Pandas form
    vals = [xr.DataArray(ds=df_h[c], dims=['time'], coords={'time':df_h.index}, attrs=ds[c].attrs) for c in df_h.columns]
    dsResample = xr.Dataset(dict(zip(df_h.columns,vals)), attrs=ds.attrs)
    
    return dsResample

#===============================================================================================================================
def eps_data_prep(ds, snr_cutoff, corr_cutoff, nmax, fs):
    '''
    Takes the dataset generated from 'datfile_to_ds' and filters all components of velocity based on SNR and Correlation cutoffs
    for each beam.
    
    Methodology for this process is based on adv quality control parameters from Fedderson & Falk (2010)
    
    Additional information regarding suggested test parameters and theory behind tests can be located online at:
    https://support.nortekgroup.com/hc/en-us/articles/360029839351-The-Comprehensive-Manual-Velocimeters
    
    INPUTS:
    senfile: the .sen file imported directly from the Vector
    
    
    OUTPUTS:
    xarray dataset with "Flag" data array
    
    EXAMPLE:
    import numpy as np
    import xarray as xr
    import vector_tools as vt
    
    senfile = 'vector.sen'
    ds = vt.senfile_to_ds(senfile)
    '''
    # Check for sigmaSS and sigmaCORR in both adv datasets
    # ySS = 10 dB, yCORR = 60%
    
    # Start by filtering data that doesn't pass the SNR test
    print('Filtering low SNR data')
    ySS = xr.zeros_like(ds.East)
    ySS = ySS + xr.where((ds.SNR_B1 > snr_cutoff), 0, 1)
    ySS = ySS + xr.where((ds.SNR_B2 > snr_cutoff), 0, 1)
    ySS = ySS + xr.where((ds.SNR_B3 > snr_cutoff), 0, 1)
    ds_filt_ss = ds.where(ySS < 1)

    good_data = ds_filt_ss.where(ds_filt_ss.Burst_number.isnull() == False, drop = True)
    dSS = np.unique(good_data.Burst_number, return_counts=True)
    good_bursts = np.where(((nmax - dSS[1])/nmax) > .1, np.nan, dSS[0])
    good_bursts = good_bursts[~np.isnan(good_bursts)]

    ds_filt_ss['Burst_number'] = ds.Burst_number
    ds_filt_ss['Pressure'] = ds.Pressure

    ds_filt_ss = ds_filt_ss.where(ds_filt_ss.Burst_number.isin(good_bursts))
    ds_filt_ss = ds_filt_ss.where(ds_filt_ss.Burst_number.isin(good_bursts), drop=True)

    print('Filtering low correlation data')
    yCORR = xr.zeros_like(ds_filt_ss.East)
    yCORR = yCORR + xr.where((ds_filt_ss.Correlation_B1 > corr_cutoff), 0, 1)
    yCORR = yCORR + xr.where((ds_filt_ss.Correlation_B2 > corr_cutoff), 0, 1)
    yCORR = yCORR + xr.where((ds_filt_ss.Correlation_B3 > corr_cutoff), 0, 1)
    ds_filt_corr = ds_filt_ss.where(yCORR < 1)

    good_corrdata = ds_filt_corr.where(ds_filt_corr.Burst_number.isnull() == False, drop = True)
    dCORR = np.unique(good_corrdata.Burst_number, return_counts=True)
    good_corrbursts = np.where(((nmax - dCORR[1])/nmax) > .1, np.nan, dCORR[0])
    good_corrbursts = good_corrbursts[~np.isnan(good_corrbursts)]
    
    ds_filt_corr['Burst_number'] = ds_filt_ss.Burst_number
    ds_filt_corr['Pressure'] = ds_filt_ss.Pressure

    ds_filt = ds_filt_corr.where(ds_filt_corr.Burst_number.isin(good_corrbursts))
    ds_filt = ds_filt.where(ds_filt.Burst_number.isin(good_corrbursts), drop = True)
    ds_filt = ds_filt.drop(labels=['CSPD','CDIR','Checksum','Dat_flag','SNR_B1','SNR_B2','SNR_B3',
                                   'Correlation_B1','Correlation_B2','Correlation_B3'])
    
    print('Interpolating gaps < 1s')
    delta = timedelta(seconds=1 + (1/fs)) # Maximum gap of less than 1 second and 31.25 milliseconds
    ds_int = ds_filt.interpolate_na(dim="time", method="linear", max_gap = delta) # Linearly interpolates across gaps up to a limit denoted by time delta

    print('Sorting remaining gaps')
    # Interpolate_na cannot estimate data gaps without start or end points, or gaps exceeding the time delta
    gaps = ds_int.where(ds_int.East.isnull() == True, drop = True) # Generate a dataset with all leftover uninterpolated gaps
    gap_times = gaps.time
    tdiff = np.diff(gap_times)
    gap_ranges = np.split(gap_times, np.where(tdiff > np.timedelta64(int(1000000000/fs), 'ns'))[0]+1)
    gap_length = np.empty(len(gap_ranges))
    for i in range(len(gap_ranges)):
        gap_length[i] = len(gap_ranges[i])

    print('Patching gaps > 10s')
    patchable = np.where(gap_length > (fs*10))[0]
    patch_arr = xr.zeros_like(ds_int)
    for i in patchable:
        patch_arr = patch_arr + xr.where(ds_int.time.isin(gap_ranges[i]), 1, 0)
    ds_patch = ds_int.where(patch_arr == 0, drop=True)

    print('Averaging gaps between 1s and 10s')
    averageable = np.where((gap_length > fs) & (gap_length <= (fs*10)))[0]
    east_arr = xr.zeros_like(ds_patch.East)
    north_arr = xr.zeros_like(ds_patch.North)
    vert_arr = xr.zeros_like(ds_patch.Vertical)
    for i in averageable:

        east_avg = np.mean(ds.East.sel(time = slice(gap_ranges[i][0], gap_ranges[i][-1])))
        east_arr = east_arr + xr.where(ds_patch.time.isin(gap_ranges[i]), east_avg, 0)
    
        north_avg = np.mean(ds.North.sel(time = slice(gap_ranges[i][0], gap_ranges[i][-1])))
        north_arr = north_arr + xr.where(ds_patch.time.isin(gap_ranges[i]), north_avg, 0)
    
        vert_avg = np.mean(ds.Vertical.sel(time = slice(gap_ranges[i][0], gap_ranges[i][-1])))
        vert_arr = vert_arr + xr.where(ds_patch.time.isin(gap_ranges[i]), vert_avg, 0)

    ds_patch['East'] = xr.where(east_arr != 0, east_arr, ds_patch.East)
    ds_patch['North'] = xr.where(north_arr != 0, north_arr, ds_patch.North)
    ds_patch['Vertical'] = xr.where(vert_arr != 0, vert_arr, ds_patch.Vertical)
    
    print('Filling in front and back gaps < 1s')
    ds_final = ds_patch.copy(deep = True)
    ds_final = ds_final.bfill("time")
    ds_final = ds_final.ffill("time")
    
    return ds_final

#===============================================================================================================================
def patchVec(data):
    
    ds = data.copy(deep=True)

    fs = ds.attrs['Sampling_Rate'] #Sample frequency to determine time delta

    print('Interpolating gaps <= 1s')
    delta = timedelta(seconds=1 + (2/fs)) #Maximum gap of 1 second (2/fs ensures that points right at the margin are taken into account)
    dsPatch = ds.interpolate_na(dim="time", method="linear", max_gap = delta) #Linearly interpolates across gaps up to a limit denoted by time delta
    
    #Unnecessary sorting algorithm
    #Keeping just in case
    '''print('Organzing remaining gaps > 1s')
    gapTimes = dsPatch.time.where(dsPatch.East.isnull() == True, drop = True).values #Generate a dataset with all leftover uninterpolated gaps

    tDiff = np.diff(gapTimes) #Find differences between gaps to assess which ranges are non-sequential

    gapRanges = np.split(gapTimes, np.where(tDiff > np.timedelta64(int(1000000000/fs), 'ns'))[0]+1)
    gapLength = np.empty(len(gapRanges))
    for i in range(len(gapRanges)):
        gapLength[i] = len(gapRanges[i])

    #interpolate_na does not account for nan's at the start or end of a burst that lack a start or end point
    #These gaps are <= 1s and can be saved by using backfill or frontfill of the last acceptable point
    fixable = [gapRanges[i] for i in np.where(gapLength > (fs))[0]] #Seperate gaps > 1s for patching while the others are left for ffil and bfill

    print('Patching data')
    fixable1d = np.concatenate(fixable).ravel() #Merge all gaps together to prevent non-homogenous dimensions
    timePatched = dsPatch.time.where(dsPatch.time.isin(fixable1d)==False, drop=True) #Create array where all indices with gaps = 1, and good data = 0

    dsPatch.coords['time_patched'] = (["time_patched"], timePatched)
    dsPatch["BurstNum"] = (["time_patched"],dsPatch.BurstNum.where(dsPatch.time.isin(fixable1d)==False, drop=True))
    dsPatch["East"] = (["time_patched"],dsPatch.East.where(dsPatch.time.isin(fixable1d)==False, drop=True))
    dsPatch['North'] = (["time_patched"],dsPatch.North.where(dsPatch.time.isin(fixable1d)==False, drop=True))
    dsPatch['Up'] = (["time_patched"],dsPatch.Up.where(dsPatch.time.isin(fixable1d)==False, drop=True))
    dsPatch['Primary'] = (["time_patched"],dsPatch.Primary.where(dsPatch.time.isin(fixable1d)==False, drop=True))
    dsPatch['Secondary'] = (["time_patched"],dsPatch.Secondary.where(dsPatch.time.isin(fixable1d)==False, drop=True))
                        
    #Forward fill and backfill all leftover small gaps
    dsPatch = dsPatch.ffill("time_patched")
    dsPatch = dsPatch.bfill("time_patched")'''
    print('Generating patched dataset')
    return dsPatch
#===============================================================================================================================
def interpAvgVec(data):
    
    ds = data.copy(deep=True)

    fs = ds.attrs['Sampling_Rate'] #Sample frequency to determine time delta

    print('Interpolating gaps <= 1s')
    delta = timedelta(seconds=1 + (2/fs)) #Maximum gap of 1 second (2/fs ensures that points right at the margin are taken into account)
    dsInt = ds.interpolate_na(dim="time", method="linear", max_gap = delta) #Linearly interpolates across gaps up to a limit denoted by time delta

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
    
    primary, secondary, theta = vt.find_princaxs(dsInt.East, dsInt.North)
    
    dsInt["Primary"] = (["time"],primary)
    dsInt["Secondary"] = (["time"],-secondary)
    dsInt.attrs["Theta"] = theta
    
    return dsInt
#===============================================================================================================================
def badDataRatio(data, velComp, newVarName):
    
    #Max number of samples within a single burst
    nmax = data.NoVelSamples.values[0]
    
    #Find and drop all bad datapoints within BurstNum
    goodData = data.BurstNum.where(velComp==True, drop=True)
    
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
def spectraPlotter(ds, burstNumber, rho, plot = True, horizontalVel = True, fullOut = False):
    
    hz = ds.attrs['Sampling_Rate']
    burstUp = ds.Up.where((ds.BurstNum.isin(burstNumber)) & (ds.Up.isnull()==False), drop = True)
    burstPressure = ds.Pressure.where((ds.BurstNum.isin(burstNumber)) & (ds.Up.isnull()==False),drop=True) * 10000
    
    if horizontalVel:
        burstU = ds.Primary.where((ds.BurstNum.isin(burstNumber)) & (ds.Primary.isnull()==False), drop = True)
        burstV = ds.Secondary.where((ds.BurstNum.isin(burstNumber)) & (ds.Secondary.isnull()==False), drop = True)
        Fu, Su = welch(burstU, fs=hz, nperseg= 1920, window='hann')
        Fv, Sv = welch(burstV, fs=hz, nperseg= 1920, window='hann')
        
    Fw, Sw = welch(burstUp, fs=hz, nperseg= 1920, window='hann')
    Fp, Sp = welch(burstPressure, fs=hz, nperseg= 1920, window='hann') # Pressure spectra

    # Required variables
    g = 9.8 # Gravity
    z = burstPressure/(rho*g) # Depth (m): the recorded pressure converted to meters of seawater
    H = np.mean(z) + .578 # Sea level height (m): mean pressure detected by the pressure sensor plus the height of sensor from the bottom
    Zp = -(np.mean(z)) # Depth of pressure sensor (m)
    Zv = (-H) + .824 # Depth of velocity sensor (m): Sea level height plus the height of the velocity transducers from the bottom
    T = 1/Fp # Period (s^-1)

    # Omega (radian frequency) and wavenumber (k)
    omega,k,Cph,Cg = vt.wavedisp(T, H)

    # Generate empty arrays for p' and w' values
    p_prime = np.empty(len(omega))
    w_prime = np.empty(len(omega))

    # Calculate P' and W' using wave dispersion relationships over angular frequency spectrum
    for j in range(len(omega)): # For loops iterates over all values of omega
        p_prime[j] = (rho*g)*(np.cosh(k[j]*(Zp+H))/np.cosh(k[j]*H))
        w_prime[j] = (-omega[j])*(np.sinh(k[j]*(Zv+H)))/(np.sinh(k[j]*H))

    # Scale factor to convert Spp to Sw'w'
    scale_factor = w_prime**2 / p_prime**2

    # Converted vertical velocity spectra
    Sw_prime = (Sp * scale_factor)

    # Determine cutoff frequency by identifying pressure noise floor via local minimum
    lfc = argrelextrema(Sw_prime, np.less, order=10)[0][0]+1
    
    if plot & horizontalVel:
            plt.figure(figsize=(8,8))
            plt.title('Vertical velocity spectra from burst #' + str(burstNumber))
            plt.loglog(omega, Su + Sv, '-b')
            plt.loglog(omega, Sw, '-k') # Used the 60s hann window spectra from before
            plt.loglog(omega[:lfc], Sw_prime[:lfc], '-r')
            plt.xlabel('Omega [rad/s]')
            plt.ylabel('Sww [(m/s)^2/s]')
            plt.legend(['Suu + Svv', 'Sww', 'Sww from pressure'])
            plt.show()
            
    else:
        plt.figure()
        plt.title('Vertical velocity spectra from burst #' + str(burstNumber))
        plt.loglog(omega, Sw, '-k') # Used the 60s hann window spectra from before
        plt.loglog(omega[:lfc], Sw_prime[:lfc], '-r')
        plt.xlabel('Omega [rad/s]')
        plt.ylabel('Sww [(m/s)^2/s]')
        plt.legend(['Sww', 'Sww from pressure'])
        plt.show()
        
        
    if fullOut:
        
        return omega, Su, Sv, Su, Sw_prime

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

# Function for creating power law curve
def power_law(x, a, b):
    return a*np.power(x, b)

#===============================================================================================================================

def find_princaxs(u,v):
    # THE FOLLOWING CODE IS ADAPTED FROM STEVEN CUNNINGHAM'S MASTERS THESIS (2019)

    # Rotate velocity data along the principle axes U and V
    theta, major, minor = ts.princax(u, v) # theta = angle, major = SD major axis (U), SD minor axis (V)
    U, V = ts.rot(u, v, -theta)
    
    return U,V,theta

#===============================================================================================================================

def eval_Sww(ds, burst_number, rho=None, Spp_conv = False, Plot = False):
    
    burst = ds.where(ds.Burst_number.isin(burst_number), drop = True) # Burst 52 is best spectra so far
        
    if Spp_conv & (rho == None):
        raise ValueError('Value for rho is required for pressure spectra conversion.')
            
    elif Spp_conv:
        # Convert adv pressure from dbar to Pascals (1:10000)
        pressure = burst.Pressure * 10000

        # Since the goal is to find the specific spectra of waves, a spectrum of pressure is used over vertical velocity
        Fp, Sp = welch(pressure, fs=32, nperseg= 2240, window='hann') # Pressure spectra
        
    
        # Required variables
        g = 9.8 # Gravity
        z = pressure/(rho*g) # Depth: the recorded pressure converted to meters of seawater
        H = np.mean(z) + .578 # Sea level height: mean pressure detected by the pressure sensor plus the height of sensor from the bottom
        Zp = -(np.mean(z)) # Depth of pressure sensor
        Zv = (-H) + .824 # Depth of velocity sensor: Sea level height plus the height of the velocity transducers from the bottom
        T = 1/Fp # Period: uused

        # Omega (radian frequency) and wavenumber (k)
        omega,k,Cph,Cg = wavedisp(T, H)

        # Generate empty arrays for p' and w' values
        p_prime = np.empty(len(omega))
        w_prime = np.empty(len(omega))

        for j in range(len(omega)): # For loops iterates over all values of omega
            p_prime[j] = (rho*g)*(np.cosh(k[j]*(Zp+H))/np.cosh(k[j]*H))
            w_prime[j] = (-omega[j])*(np.sinh(k[j]*(Zv+H)))/(np.sinh(k[j]*H))
        scale_factor = w_prime**2 / p_prime**2

        Sw_prime = (Sp * scale_factor)

        # Determine cutoff frequency by identifying pressure noise floor via local minimum
        fc = argrelextrema(Sw_prime, np.less, order=10)[0][0]+1
    else:
        fc = 22
    Fw, Sw = welch(burst.Vertical, fs=32, nperseg= 2240, window='hann') # Vertical velocity spectra
        
    # Fit the power law between the two bounds
    pars, cov = curve_fit(f=power_law, xdata=Fw[fc:], ydata=Sw[fc:], p0=[0, 0], bounds=(-np.inf, np.inf))
    mu = pars[0]*(Fw**(-5/3)) # -5/3 slope line with fit constant a (pars[0])
        
    if Plot:
        plt.figure()
        plt.title('Velocity spectra from burst #' + str(burst_number))
        plt.loglog(Fw, Sw, '-k')
        plt.loglog(Fw, mu, '.y', ms = 2)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Sww [(m/s)^2/s]')
    
        if Spp_conv:
            plt.loglog(Fp[0:fc], Sw_prime[0:fc], '-r')
            plt.legend(['Sww', '-5/3 power-law fit', 'Sww from pressure'])
        else:
            plt.legend(['Sww', '-5/3 power-law fit'])
    if Spp_conv:
        return Sw, pars, cov, (Fw[fc],fc)
    else:
        return Sw, pars, cov
    
#===================================================================================================================================================    
# OBSOLETE DATA IMPORT FUNCTIONS
# SAVING JUST IN CASE THEY ARE NEEDED
"""def datfile_to_ds(datfile, vhdfile, fs):
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
    fs = 32 # Vector sampled at 32Hz
    ds = vt.datfile_to_ds(datfile, vhdfile, fs)
    '''
    # Create column names for pandas dataframe
    # 'dat_cols' pertains to the default .DAT file from the vector, 'sen_cols' pertains to the default .SEN file
    print('Importing data...')
    print(' ')
    dat_cols = ["Burst_counter", "Ensemble_counter", "Velocity_East", "Velocity_North", "Velocity_Up", "Amplitude_B1", 
                "Amplitude_B2", "Amplitude_B3", "SNR_B1", "SNR_B2", "SNR_B3", "Correlation_B1", "Correlation_B2", 
                "Correlation_B3", "Pressure", "AnalogInput1", "AnalogInput2", "Checksum"]
    vhd_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Burst_counter", "Num_samples", "NA1", "NA2", "NA3", "NC1", "NC2", "NC3", "Dist1_st", "Dist2_st", "Dist3_st", "Distavg_st",
               "distvol_st", "Dist1_end", "Dist2_end", "Dist3_end", "Distavg_end", "distvol_end"]

    dat = pd.read_csv(datfile, delimiter='\s+', names = dat_cols)
    vhd = pd.read_csv(vhdfile, delimiter='\s+', names = vhd_cols)
    
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
    
    print('Creating datetime for data file...')
    print(' ')
    
    samples = vhd.Num_samples[1]
    vhd_dates = np.empty(shape=(len(vhd)*samples), dtype='datetime64[ns]')
    
    for i in range(0, len(vhd)-1, 1):
        vhd_dates[i*samples:(i+1)*samples] = pd.date_range(start=(str(vhd.iloc[i,0])+'/'+ str(vhd.iloc[i,1])+'/'+ str(vhd.iloc[i,2])+' '+ str(vhd.iloc[i,3])+':'+ 
                                                     str(vhd.iloc[i,4])+':'+ str(vhd.iloc[i,5])), periods = samples, freq = t_step)
    vhd_dates = vhd_dates[:len(dat)]   
    dat.insert(0, 'datetime', vhd_dates)
    dat.datetime = pd.to_datetime(dat.datetime)
    
    print('Creating xarray dataset...')
    print('')
    # create coords

    # put data into a dataset
    ds = xr.Dataset(
        data_vars=dict(
            Burst_number = (["time"], dat['Burst_counter']),
            East = (["time"], dat['Velocity_East']),
            North = (["time"], dat['Velocity_North']),
            Up = (["time"], dat['Velocity_Up']),
            CSPD = (["time"], np.sqrt(((dat['Velocity_East'])**2) + ((dat['Velocity_North'])**2))),
            CDIR = (["time"], vec_angle(dat['Velocity_East'].to_xarray(), dat['Velocity_North'].to_xarray())),
            SNR_B1 = (["time"], dat['SNR_B1']),
            SNR_B2 = (["time"], dat['SNR_B2']),
            SNR_B3 = (["time"], dat['SNR_B3']),
            Correlation_B1 = (["time"], dat['Correlation_B1']),
            Correlation_B2 = (["time"], dat['Correlation_B2']),
            Correlation_B3 = (["time"], dat['Correlation_B3']),
            Pressure = (["time"], dat['Pressure']),
            Checksum = (["time"], dat['Checksum'])    
        ),
        coords=dict(
            time=(["time"], dat.datetime),
        ),
        attrs=dict(description="ADV data"),
    )
    ds['Burst_number'].attrs['description'] = 'Burst counter'
    ds['East'].attrs['units'] = 'm/s'
    ds['North'].attrs['units'] = 'm/s'
    ds['Vertical'].attrs['units'] = 'm/s'
    ds['CSPD'].attrs['units'] = 'm/s'
    ds['CSPD'].attrs['description'] = 'Horizontal current speed, the magnitude of the Eastern and Northern velocity vectors.'
    ds['CDIR'].attrs['units'] = 'Degrees'
    ds['CDIR'].attrs['description'] = 'Direction of the horizontal current speed derived from the vec_angle function in vector_tools.py.'
    ds['SNR_B1'].attrs['units'] = 'dB'
    ds['SNR_B1'].attrs['description'] = 'Signal to noise ratio of beam 1.'
    ds['SNR_B2'].attrs['units'] = 'dB'
    ds['SNR_B2'].attrs['description'] = 'Signal to noise ratio of beam 2.'
    ds['SNR_B3'].attrs['units'] = 'dB'
    ds['SNR_B3'].attrs['description'] = 'Signal to noise ratio of beam 3.'
    ds['Correlation_B1'].attrs['units'] = '%'
    ds['Correlation_B1'].attrs['description'] = 'Beam correlation measurment from beam 1.'
    ds['Correlation_B2'].attrs['units'] = '%'
    ds['Correlation_B2'].attrs['description'] = 'Beam correlation measurment from beam 2.'
    ds['Correlation_B3'].attrs['units'] = '%'
    ds['Correlation_B3'].attrs['description'] = 'Beam correlation measurment from beam 3.'
    ds['Pressure'].attrs['units'] = 'dBar'
    ds['Pressure'].attrs['description'] = 'Ambient pressure recorded by the instrument at the same frequency as the velocity data.'
    ds['Checksum'].attrs['description'] = 'A binary internal test conducted by the instrument which indicates successful or failed measurement (1 = failure). This test is conducted at the same frequency as the velocity data.'
    
    # DATA TESTS
    # Fixable failure (excluding checksum and correlation)= 36
    # Fixable failure (with acceptable correlations) = 41
    # Complete failure = 42
    # Tests performed on the recorded velocity and pressure data found on the .dat file
    
    dat_flag = xr.zeros_like(ds.East) # Same shape as .dat data arrays
    qartod_flag = xr.zeros_like(dat_flag)
    
    burst_diff = np.diff(ds.Burst_number, axis = 0, prepend = 0)
    burst_diff[0] = 0
    
    min_depth = np.int(np.mean(ds.Pressure) - np.std(ds.Pressure))
    
    # Checksum tests 
    dat_flag = dat_flag + xr.where(ds.Checksum == 0, 0, 61)
    
    # Pressure test
    dat_flag = dat_flag + xr.where(ds.Pressure >= min_depth, 0, 61)
    
    # Beam correlation/SNR test
    dat_flag = dat_flag + xr.where((ds.Correlation_B1 > 70), 0, 0) # Full pass condition
    dat_flag = dat_flag + xr.where((ds.Correlation_B1 <= 70) & (ds.Correlation_B1 > 50), 3, 0) # Not ideal but acceptable
    dat_flag = dat_flag + xr.where((ds.Correlation_B1 <= 50), 61, 0) # Full failure and should be uniquely flagged to identify in analysis
    
    dat_flag = dat_flag + xr.where((ds.Correlation_B2 > 70), 0, 0) 
    dat_flag = dat_flag + xr.where((ds.Correlation_B2 <= 70) & (ds.Correlation_B2 > 50), 3, 0) 
    dat_flag = dat_flag + xr.where((ds.Correlation_B2 <= 50), 61, 0) 
    
    dat_flag = dat_flag + xr.where((ds.Correlation_B3 > 70), 0, 0) 
    dat_flag = dat_flag + xr.where((ds.Correlation_B3 <= 70) & (ds.Correlation_B3 > 50), 3, 0) 
    dat_flag = dat_flag + xr.where((ds.Correlation_B3 <= 50), 61, 0) 
    
    # Horizontal velocity test
    # For East-West
    dat_flag = dat_flag + xr.where(np.abs(ds.East) >= 3, 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(ds.East) < 3) & (np.abs(ds.East) >= 1), 1, 0)
    dat_flag = dat_flag + xr.where(np.abs(ds.East) < 1, 0, 0)
    # For North-South
    dat_flag = dat_flag + xr.where(np.abs(ds.North) >= 3, 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(ds.North) < 3) & (np.abs(ds.North) >= 1), 1, 0)
    dat_flag = dat_flag + xr.where(np.abs(ds.North) < 1, 0, 0)
    
    # Vertical velocity test
    dat_flag = dat_flag + xr.where(np.abs(ds.Vertical) >= 2, 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(ds.Vertical) < 2) & (np.abs(ds.Vertical) >= 1), 1, 0)
    dat_flag = dat_flag + xr.where(np.abs(ds.Vertical) < 1, 0, 0)
    
    # u, v, w rate of change test
    # For East-west (u)
    du = np.diff(ds.East, axis = 0, prepend = 0) 
    du[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(du) >= 2) & (burst_diff==0), 4, 0)
    pw_flag = pw_flag + xr.where((np.abs(du) >= 2) & (burst_diff==0), 1, 0)
    
    dat_flag = dat_flag + xr.where((np.abs(du) < 2) & (np.abs(du) >= 1) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(du) < 1) & (np.abs(du) >= .25) & (burst_diff==0), 1, 0) 

    # For North-South (v)
    dv = np.diff(ds.North, axis = 0, prepend = 0) 
    dv[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dv) >= 2) & (burst_diff==0), 4, 0) 
    pw_flag = pw_flag + xr.where((np.abs(dv) >= 2) & (burst_diff==0), 1, 0)
    
    dat_flag = dat_flag + xr.where((np.abs(dv) < 2) & (np.abs(dv) >= 1) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dv) < 1) & (np.abs(dv) >= .25) & (burst_diff==0), 1, 0)

    # For vertical (w)
    dw = np.diff(ds.Vertical, axis = 0, prepend = 0) 
    dw[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dw) >= 1) & (burst_diff==0), 4, 0) # Magnitudes of vertical velocity are typically smaller than horizontal velocity, so thresholds are reduced
    pw_flag = pw_flag + xr.where((np.abs(dw) >= 1) & (burst_diff==0), 1, 0)
    
    dat_flag = dat_flag + xr.where((np.abs(dw) < 1) & (np.abs(dw) >= .5) & (burst_diff==0), 3, 0) 
    dat_flag = dat_flag + xr.where((np.abs(dw) < .5) & (np.abs(dw) >= .15) & (burst_diff==0), 1, 0)
    
    # Current speed test
    dat_flag = dat_flag + xr.where(ds.CSPD < 4, 0, 3)
    
    # Current speed and direction rate of change tests
    dCSPD = np.diff(ds.CSPD, axis = 0, prepend = 0)
    dCSPD[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) >= 4) & (burst_diff==0), 4, 0)
    pw_flag = pw_flag + xr.where((np.abs(dCSPD) >= 4) & (burst_diff==0), 1, 0)
    
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 4) & (np.abs(dCSPD) >= 1) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 1) & (np.abs(dCSPD) >= .25) & (burst_diff==0), 1, 0)

    # For current direction
    dCDIR = np.diff(ds.CDIR, axis = 0, prepend = 0)
    dCDIR[0] = dCDIR[0] - dCDIR[0]
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) >= 135) & (np.abs(dCDIR) <= 225)& (burst_diff==0), 4, 0)
    pw_flag = pw_flag + xr.where((np.abs(dCDIR) >= 135) & (np.abs(dCDIR) <= 225)& (burst_diff==0), 1, 0)
    
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) < 135) & (np.abs(dCDIR) >= 30) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) <= 330) & (np.abs(dCDIR) > 225) & (burst_diff==0), 3, 0)
    
    # Add the new flag data array to the existing dataset
    # Fail (5) is flag > 60
    # Suspect with pw (4) is 8 < flag <= 60 and pw_flag >= 1
    # Suspect (3) is 8 < flag <= 36 and pw_flag == 0
    # Pass with possible pw (2) is <= 8 and pw_flag >= 1
    # Pass (1) is flag <= 8 and pw_flag == 0
    qartod_flag = qartod_flag + xr.where(dat_flag > 60, 5, 0)
    qartod_flag = qartod_flag + xr.where((dat_flag <= 60) & (dat_flag > 8) & (pw_flag >= 1), 4, 0)
    qartod_flag = qartod_flag + xr.where((dat_flag <= 60) & (dat_flag > 8) & (pw_flag == 0), 3, 0)
    qartod_flag = qartod_flag + xr.where((dat_flag <= 8) & (pw_flag >= 1), 2, 0)
    qartod_flag = qartod_flag + xr.where((dat_flag <= 8) & (pw_flag == 0), 1, 0)
    ds['Dat_flag'] = (["time"], qartod_flag)
    ds['Dat_flag'].attrs['Flag score'] = '[1, 2, 3, 4, 5]'
    ds['Dat_flag'].attrs['Grade definition'] = '1 = Pass, 2 = Pass with potential phase wrapping, 3 = Suspect, 4 = Suspect with potential phase wrapping, 5 = Fail'
    ds['Dat_flag'].attrs['Description'] = 'Flag grading system is based on QARTOD quality control parameters and tests in Nortek ADV user manual'
    
    return ds
    
#===============================================================================================================================
def senfile_to_ds(senfile):
    '''
    Take the .sen file from the vector and generate an xarray dataset with all variables. Also conducts quality control tests as recommended by the 
    Nortek N3015-030 Comprehensive Manual for Velocimeters
    
    Additional information regarding suggested test parameters and theory behind tests can be located online at:
    https://support.nortekgroup.com/hc/en-us/articles/360029839351-The-Comprehensive-Manual-Velocimeters
    
    INPUTS:
    senfile: the .sen file imported directly from the Vector
    
    
    OUTPUTS:
    xarray dataset with "Flag" data array
    
    EXAMPLE:
    import numpy as np
    import xarray as xr
    import vector_tools as vt
    
    senfile = 'vector.sen'
    ds = vt.senfile_to_ds(senfile)
    '''
        
    # Create column names for pandas dataframe
    # 'dat_cols' pertains to the default .DAT file from the vector, 'sen_cols' pertains to the default .SEN file
    print('Importing data...')
    print(' ')
    sen_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Error_code", "Status_code", "Battery_voltage", "Soundspeed", "Heading", "Pitch", "Roll", "Temperature", 
                "Analog_input", "Checksum"]

    sen = pd.read_csv(senfile, delimiter='\s+', names = sen_cols)
    
    print('Creating datetime for sensor file...')
    print(' ')

    sen['datetime'] = sen['Month'].map(str)+'/'+sen['Day'].map(str)+'/'+sen['Year'].map(str)+' '+sen['Hour'].map(str)+':'+sen['Minute'].map(str)+':'+sen['Second'].map(str)
    sen['datetime'] = pd.to_datetime(sen['datetime'],utc=True)
    
    print('Creating xarray dataset...')
    print('')
    # create coords

    # put data into a dataset
    ds = xr.Dataset(
        data_vars=dict(
            Battery = (["time"], sen['Battery_voltage']),
            Soundspeed = (["time"], sen['Soundspeed']),
            Heading = (["time"], sen['Heading']),
            Pitch = (["time"], sen['Pitch']),
            Roll = (["time"], sen['Roll']),
            Temperature = (["time"], sen['Temperature']),
            Checksum = (["time"], sen['Checksum'])    
        ),
        coords=dict(
            time=(["time"], sen.datetime)
        ),
        attrs=dict(description="ADV data"),
    )
    ds['Battery'].attrs['units'] = 'Volts'
    ds['Battery'].attrs['description'] = 'Voltage of the instrument measure at 1Hz during sampling period.'
    ds['Soundspeed'].attrs['units'] = 'm/s'
    ds['Soundspeed'].attrs['description'] = 'Speed of sound recorded by the instrument based on recorded temperature and set salinity.'
    ds['Heading'].attrs['units'] = 'Degrees'
    ds['Heading'].attrs['units'] = 'Degrees'
    ds['Pitch'].attrs['units'] = 'Degrees'
    ds['Roll'].attrs['units'] = 'Degrees'
    ds['Temperature'].attrs['units'] = 'Celsius'
    ds['Checksum'].attrs['description'] = 'A binary internal test conducted by the instrument which indicates successful or failed measurement (1 = failure). This test is conducted at 1Hz during sampling period.'
    
    # SENSOR TESTS 
    # Tests for the quality of sensor parameters found on the .sen file  
    
    sen_flag = xr.zeros_like(ds.BatVolt) # Same shape as .sen data arrays
    
    # Battery voltage test
    sen_flag = sen_flag + xr.where(ds.BatVolt >= 9.6, 0, 3) # xr.where(condition, value if true, value if false)
    
    # Compass Heading test
    sen_flag = sen_flag + xr.where((ds.Heading >= 0) & (ds.Heading <= 360), 0, 4)
    
    # Soundspeed test
    sen_flag = sen_flag + xr.where((ds.SoundSpeed >= 1493) & (ds.Soundspeed <= 1502), 0, 4)
    
    # Tilt test
    sen_flag = sen_flag + xr.where(np.abs(ds.Roll) < 5, 0, 4)
    sen_flag = sen_flag + xr.where(np.abs(ds.Pitch) < 5, 0, 4)
    
    # Checksum tests
    sen_flag = sen_flag + xr.where(ds.ChecksumSen == 0, 0, 4) 
    
    ds['SenFlag'] = (["time_sen"], sen_flag)
    
    ds['SenFlag'].attrs['description'] = 'Flag value based on internal sensor tests: battery, heading, pitch and roll, temperature, and soundspeed. Sampled at 1Hz.'
    
    return ds"""