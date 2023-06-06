import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import xarray as xr
import gsw # https://teos-10.github.io/GSW-Python/gsw_flat.html
import vector_tools as vt
import Functions as fn

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
def datfile_to_ds(datfile, vhdfile, fs):
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
            Vertical = (["time"], dat['Velocity_Up']),
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
    pw_flag = xr.zeros_like(dat_flag)
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
    
    sen_flag = xr.zeros_like(ds.Battery) # Same shape as .sen data arrays
    
    # Battery voltage test
    sen_flag = sen_flag + xr.where(ds.Battery >= 9.6, 0, 3) # xr.where(condition, value if true, value if false)
    
    # Compass Heading test
    sen_flag = sen_flag + xr.where((ds.Heading >= 0) & (ds.Heading <= 360), 0, 4)
    
    # Soundspeed test
    sen_flag = sen_flag + xr.where((ds.Soundspeed >= 1493) & (ds.Soundspeed <= 1502), 0, 4)
    
    # Tilt test
    sen_flag = sen_flag + xr.where(np.abs(ds.Roll) < 5, 0, 4)
    sen_flag = sen_flag + xr.where(np.abs(ds.Pitch) < 5, 0, 4)
    
    # Checksum tests
    sen_flag = sen_flag + xr.where(ds.Checksum == 0, 0, 4) 
    
    ds['Sen_flag'] = (["time"], sen_flag)
    
    ds['Sen_flag'].attrs['description'] = 'Flag value based on internal sensor tests: battery, heading, pitch and roll, temperature, and soundspeed. Sampled at 1Hz.'
    
    return ds

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

def find_princaxs(ds,u,v):
    # THE FOLLOWING CODE IS ADAPTED FROM STEVEN CUNNINGHAM'S MASTERS THESIS (2019)

    # Rotate velocity data along the principle axes U and V
    theta, major, minor = ts.princax(u, v) # theta = angle, major = SD major axis (U), SD minor axis (V)
    U, V = ts.rot(u, v, -theta)
    ds['U'] = U
    ds['V'] = V
    
    return ds

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