import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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

def adv_qc(ds):
    '''
    Take the dataset produce from vector_to_ds and conduct quality control tests as recommended by the 
    Nortek N3015-030 Comprehensive Manual for Velocimeters
    
    Additional information regarding suggested test parameters and theory behind tests can be located online at:
    https://support.nortekgroup.com/hc/en-us/articles/360029839351-The-Comprehensive-Manual-Velocimeters
    
    INPUTS:
    ds: the dataset generated by the vector_to_ds function
    
    
    OUTPUTS:
    xarray dataset with "Flag" data array
    
    EXAMPLE:
    import numpy as np
    import xarray as xr
    import vector_tools as vt
    
    fs = 32 # sampling frequency of 32 Hz  
    ds = vt.vector_to_ds('vector.dat', 'vector.sen', 'vector.vhd', fs)
    ds_qc = vt.adv_qc(ds)
    '''
    
    # Generate 2D array with shape (y,x):(vertical_bins, datetime) 
    dat_flag = xr.zeros_like(ds.East) # Same shape as .dat data arrays
    sen_flag = xr.zeros_like(ds.Battery) # Same shape as .sen data arrays
    
    #======================== SENSOR TESTS ===================================================================
    # Tests for the quality of sensor parameters found on the .sen file  
    
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
    sen_flag = sen_flag + xr.where(ds.Sen_checksum == 0, 0, 4) 
    dat_flag = dat_flag + xr.where(ds.Dat_checksum == 0, 0, 4)                                       
                                                
    #======================== DATA TESTS ===================================================================        
    # Tests performed on the recorded velocity and pressure data found on the .dat file
    
    burst_diff = np.diff(ds.Burst_number, axis = 0, prepend = 0)
    burst_diff[0] = 0
    
    min_depth = np.int(np.mean(ds.Pressure) - np.std(ds.Pressure))
    
    # Beam correlation test
    dat_flag = dat_flag + xr.where(ds.Correlation_B1 >= 70, 0, 30)
    dat_flag = dat_flag + xr.where(ds.Correlation_B2 >= 70, 0, 30)
    dat_flag = dat_flag + xr.where(ds.Correlation_B3 >= 70, 0, 30)
    
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
    dat_flag = dat_flag + xr.where((np.abs(du) >= 1) & (burst_diff==0), 3, 0) 
    dat_flag = dat_flag + xr.where((np.abs(du) < 1) & (np.abs(du) >= .25) & (burst_diff==0), 1, 0) 

    # For North-South (v)
    dv = np.diff(ds.North, axis = 0, prepend = 0) 
    dv[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dv) >= 1) & (burst_diff==0), 3, 0) 
    dat_flag = dat_flag + xr.where((np.abs(dv) < 1) & (np.abs(dv) >= .25) & (burst_diff==0), 1, 0)

    # For vertical (w)
    dw = np.diff(ds.Vertical, axis = 0, prepend = 0) 
    dw[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dw) >= 1) & (burst_diff==0), 3, 0) 
    dat_flag = dat_flag + xr.where((np.abs(dw) < 1) & (np.abs(dw) >= .15) & (burst_diff==0), 1, 0) # Magnitudes of vertical velocity are typically smaller than horizontal velocity, so thresholds are reduced
    
    # Current speed test
    dat_flag = dat_flag + xr.where(ds.CSPD < 4, 0, 4)
    
    # Current speed and direction rate of change tests
    dCSPD = np.diff(ds.CSPD, axis = 0, prepend = 0)
    dCSPD[0] = 0
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) >= 4) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCSPD) < 4) & (np.abs(dCSPD) > .5) & (burst_diff==0), 1, 0)

    # For current direction
    dCDIR = np.diff(ds.CDIR, axis = 0, prepend = 0)
    dCDIR[0] = dCDIR[0] - dCDIR[0]
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) >= 135) & (np.abs(dCDIR) <= 225)& (burst_diff==0), 4, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) < 135) & (np.abs(dCDIR) >= 30) & (burst_diff==0), 3, 0)
    dat_flag = dat_flag + xr.where((np.abs(dCDIR) <= 330) & (np.abs(dCDIR) > 225) & (burst_diff==0), 3, 0)
    
    # Add the new flag data array to the existing dataset
    ds['Dat_flag'] = (["dat_time"], dat_flag)
    ds['Sen_flag'] = (["sen_time"], sen_flag)
    
    ds['Dat_flag'].attrs['description'] = 'Flag value based on SNR, beam correlation, velocity magnitudes, velocity rate of change, current magnitude and direction. Sampled at 32Hz'
    ds['Sen_flag'].attrs['description'] = 'Flag value based on internal sensor tests: battery, heading, pitch and roll, temperature, and soundspeed. Sampled at 1Hz.'
    
    return ds












#======================================================== UNUSED FUNCTIONS ===================================================================
def vector_to_df(datfile, senfile, vhdfile, fs):
    # Create column names for pandas dataframe
    # 'dat_cols' pertains to the default .DAT file from the vector, 'sen_cols' pertains to the default .SEN file
    print('Importing data...')
    dat_cols = ["Burst_counter", "Ensemble_counter", "Velocity_East", "Velocity_North", "Velocity_Up", "Amplitude_B1", 
                "Amplitude_B2", "Amplitude_B3", "SNR_B1", "SNR_B2", "SNR_B3", "Correlation_B1", "Correlation_B2", 
                "Correlation_B3", "Pressure", "AnalogInput1", "AnalogInput2", "Checksum"]
    sen_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Error_code", "Status_code", "Battery_voltage", "Soundspeed", "Heading", "Pitch", "Roll", "Temperature", 
                "Analog_input", "Checksum"]
    vhd_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Burst_counter", "Num_samples", "NA1", "NA2", "NA3", "NC1", "NC2", "NC3", "Dist1_st", "Dist2_st", "Dist3_st", "Distavg_st",
               "distvol_st", "Dist1_end", "Dist2_end", "Dist3_end", "Distavg_end", "distvol_end"]

    dat = pd.read_csv(datfile, delimiter='\s+', names = dat_cols)
    sen = pd.read_csv(senfile, delimiter='\s+', names = sen_cols)
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
    
    print('Data imported!')
    print('Patching in correct datetimes to datafile from .vhd file...')
    
    samples = vhd.Num_samples[1]
    vhd_dates = np.empty(shape=(len(vhd)*samples), dtype='datetime64[ns]')
    
    for i in range(0, len(vhd)-1, 1):
        if i % (len(vhd)/4) == 0: # Progress check every 50000 rows
            print('Currently on row', i, 'of', len(vhd))
        
        vhd_dates[i*samples:(i+1)*samples] = pd.date_range(start=(str(vhd.iloc[i,0])+'/'+ str(vhd.iloc[i,1])+'/'+ str(vhd.iloc[i,2])+' '+ str(vhd.iloc[i,3])+':'+ 
                                                     str(vhd.iloc[i,4])+':'+ str(vhd.iloc[i,5])), periods = samples, freq = t_step)

    vhd_dates = vhd_dates[:len(dat)]   
    dat.insert(0, 'Datetime', vhd_dates)
    dat.Datetime = pd.to_datetime(dat.Datetime)
    
    print('Data file patched!')
    print('Compiling sensor log information from .sen file...')

    sen_dates = np.empty(len(sen)*fs, dtype='datetime64[ns]')
    Battery = np.empty(len(sen)*fs)
    Soundspeed = np.empty(len(sen)*fs)
    Heading = np.empty(len(sen)*fs)
    Pitch = np.empty(len(sen)*fs)
    Roll = np.empty(len(sen)*fs)
    Temperature = np.empty(len(sen)*fs)

    for i in range(0, len(sen), 1):
        if i % (len(sen)/10) == 0: # Progress check every 50000 rows
                print('Currently on row', i, 'of', len(sen))
        sen_dates[i*fs:(i+1)*fs] = pd.date_range(start=(str(sen.iloc[i,0])+'/'+ str(sen.iloc[i,1])+'/'+ str(sen.iloc[i,2])+' '+ str(sen.iloc[i,3])+':'+ 
                                                     str(sen.iloc[i,4])+':'+ str(sen.iloc[i,5])), periods = fs, freq = t_step)
        Battery[i*fs:(i+1)*fs] = sen.iloc[i, 8]
        Soundspeed[i*fs:(i+1)*fs] = sen.iloc[i, 9]
        Heading[i*fs:(i+1)*fs] = sen.iloc[i, 10]
        Pitch[i*fs:(i+1)*fs] = sen.iloc[i, 11]
        Roll[i*fs:(i+1)*fs] = sen.iloc[i, 12]
        Temperature[i*fs:(i+1)*fs] = sen.iloc[i, 13]

    sen_fixed = pd.DataFrame({"Datetime":sen_dates, "Battery":Battery, "Soundspeed":Soundspeed, "Heading":Heading, "Pitch":Pitch,
                          "Roll":Roll, "Temperature":Temperature})
    
    print('Sensor log information completed!')
    print('Merging data and sensor files...')
    
    dat_fixed = pd.merge(dat, sen_fixed, how='left', on='Datetime')
    
    print('DataFrame complete!')
    return dat_fixed

#================================================================================================================================

def adv_df_flag(filepath, tilt_threshold, pressure_threshold):
    '''
    Flag adv data for correlation, tilt, pressure, and acceleration/pw_wrapping
    
    INPUTS:
    filepath: csv of adv dataframe with renamed headers from .dat and .sen files
    tilt_threshold: The maximum degree of tilt by the instrument before data becomes too biased
    pressure_threshold: The minimum pressure the adv should experience during its deployment
    
    
    RETURNS:
    pandas dataframe with additional 'Flag' and 'Flag_pw' columns for quality control
    Also returns columns for difference and absolute difference between consecutive velocities
    
    EXAMPLE:
    import numpy as np
    import pandas as pd
    
    filepath = 'adv_data.csv'
    adv_flagged = adv_df_flag(filpath, 5, 8) # Flags all data with tilt outside of 5 degrees and pressure lower than 8dbar
    '''
    
    # Read the csv file that has been modified for pandas
    df = pd.read_csv(filepath)
    
    if 'Unnamed: 0' in df.columns:
        df.drop(df.columns[0], axis=1, inplace=True)
        df.Datetime = pd.to_datetime(df.Datetime)
        
    # Add qc columns to each dataframe
    df['Flag'] = 0 # Flag column for basic tests
    df['Flag_pw'] = 0 # Flag column for acceleration/phase wrapping tests
    #===============================================================================================================================   
    # Start with beam correlation tests
    # Nortek recommends a minimum threshold of >= 70% for all beams for acceptable data, though parameters may vary per situation
    # Beam 1 correlation
    df.loc[(df["Correlation_B1"] < 80) & (df["Correlation_B1"] >= 70), "Flag"] = df.loc[(df["Correlation_B1"] < 80) & (df["Correlation_B1"] >= 70), "Flag"] + 1
    df.loc[df["Correlation_B1"] < 70, "Flag"] = df.loc[df["Correlation_B1"] < 70, "Flag"] + 20 
    # Flag score is set significantly higher since a single beam failing the correlation test needs to visibly stand out while pruning the data

    # Beam 2 correlation
    df.loc[(df["Correlation_B2"] < 80) & (df["Correlation_B2"] >= 70), "Flag"] = df.loc[(df["Correlation_B2"] < 80) & (df["Correlation_B2"] >= 70), "Flag"] + 1
    df.loc[df["Correlation_B2"] < 70, "Flag"] = df.loc[df["Correlation_B2"] < 70, "Flag"] + 20

    # Beam 3 correlation
    df.loc[(df["Correlation_B3"] < 80) & (df["Correlation_B3"] >= 70), "Flag"] = df.loc[(df["Correlation_B3"] < 80) & (df["Correlation_B3"] >= 70), "Flag"] + 1
    df.loc[df["Correlation_B3"] < 70, "Flag"] = df.loc[df["Correlation_B3"] < 70, "Flag"] + 20
    #===============================================================================================================================      
    # Tilt tests
    # Nortek recommends <30 degrees of tilt before data becomes too biased
    df.loc[(df["Roll"] > tilt_threshold) | (df["Roll"] < -tilt_threshold), "Flag"] = df.loc[(df["Roll"] > tilt_threshold) | (df["Roll"] < -tilt_threshold), "Flag"] + 4

    # Pitch
    df.loc[(df["Pitch"] > tilt_threshold) | (df["Pitch"] < -tilt_threshold), "Flag"] = df.loc[(df["Pitch"] > tilt_threshold) | (df["Pitch"] < -tilt_threshold), "Flag"] + 4
    #===============================================================================================================================
    # Pressure test
    # Pressure threshold depends on height of low tide
    # All pressure below this threshold indicates deployment/recovery of the instrument and is flagged as a fail
    df.loc[df['Pressure'] < pressure_threshold, "Flag"] = df.loc[df['Pressure'] < pressure_threshold, "Flag"] + 4
    #===============================================================================================================================
    # Acceleration tests
    # Filters high acceleration between consecutive velocities
    # Accelerations > 1m/s2 may be indicative of low beam correlation or phase wrapping if recorded velocity exceeds nominal velocity range
    
    # Create lists of accelerations for ENU velocities
    df['Nor_diff'] = df['Velocity_North'].diff() # List of differences between consecutive velocities (accelerations)
    df['Nor_absdiff'] = np.abs(df['Velocity_North'].diff()) # Absolute value of accelerations (makes future screening easier)

    df['Eas_diff'] = df['Velocity_East'].diff()
    df['Eas_absdiff'] = np.abs(df['Velocity_East'].diff())

    df['Up_diff'] = df['Velocity_Up'].diff()
    df['Up_absdiff'] = np.abs(df['Velocity_Up'].diff())

    # Assign phase wrapping ('pw') flag values
    # North
    df.loc[df['Nor_absdiff'] > 2, "Flag_pw"] = df.loc[df['Nor_absdiff'] > 2, "Flag_pw"] + 4 # Auto fail accelerations greater than 2 (+4)
    df.loc[(df['Nor_absdiff'] <= 2) & (df['Nor_absdiff'] > 1), "Flag_pw"] = df.loc[(df['Nor_absdiff'] <= 2) & 
                                                                                    (df['Nor_absdiff'] > 1), "Flag_pw"] + 1 # Accelerations between 1-2 are suspect but passable (+1)
    # Anything below 1 m/s2 is a pass (+0), though this depends on sampling environment

    # East
    df.loc[df['Eas_absdiff'] > 2, "Flag_pw"] = df.loc[df['Eas_absdiff'] > 2, "Flag_pw"] + 4 
    df.loc[(df['Eas_absdiff'] <= 2) & (df['Eas_absdiff'] > 1), "Flag_pw"] = df.loc[(df['Eas_absdiff'] <= 2) & 
                                                                                    (df['Eas_absdiff'] > 1), "Flag_pw"] + 1 

    # Upwards
    df.loc[df['Up_absdiff'] > 2, "Flag_pw"] = df.loc[df['Up_absdiff'] > 2, "Flag_pw"] + 4 
    df.loc[(df['Up_absdiff'] <= 2) & (df['Up_absdiff'] > 1), "Flag_pw"] = df.loc[(df['Up_absdiff'] <= 2) & 
                                                                                  (df['Up_absdiff'] > 1), "Flag_pw"] + 1
    
    return df

#===============================================================================================================================
    
def adv_df_to_ds(adv):
    # create coords
    time = pd.to_datetime(adv['Datetime'])
    burst = adv.Burst_counter
    adv_hvel = np.sqrt(((adv['Velocity_East'])**2) + ((adv['Velocity_North'])**2))
    adv_dir = np.arctan(adv['Velocity_North']/adv['Velocity_East']) + adv['Heading']

    # put data into a dataset
    adv_ds = xr.Dataset(
        data_vars=dict(
            Burst_number = (["time"], adv['Burst_counter']),
            East = (["time"], adv['Velocity_East']),
            North = (["time"], adv['Velocity_North']),
            Vertical = (["time"], adv['Velocity_Up']),
            Magnitude = (["time"], adv_hvel),
            Direction = (["time"], adv_dir),
            Heading = (["time"], adv['Heading']),
            Pitch = (["time"], adv['Pitch']),
            Roll = (["time"], adv['Roll']),
            Temperature = (["time"], adv['Temperature']),
            Pressure = (["time"], adv['Pressure']),
            Correlation_B1 = (["time"], adv['Correlation_B1']),
            Correlation_B2 = (["time"], adv['Correlation_B2']),
            Correlation_B3 = (["time"], adv['Correlation_B3']),
            Flag = (["time"], adv['Flag']),
            Flag_pw = (["time"], adv['Flag_pw'])
        ),
        coords=dict(
            time=(["time"], time),
            burst=(["burst"], burst)
        ),
        attrs=dict(description="ADV data"),
    )
    adv_ds['Burst_number'].attrs['description'] = 'Burst counter'
    adv_ds['East'].attrs['units'] = 'm/s'
    adv_ds['North'].attrs['units'] = 'm/s'
    adv_ds['Vertical'].attrs['units'] = 'm/s'
    adv_ds['Magnitude'].attrs['units'] = 'm/s'
    adv_ds['Magnitude'].attrs['description'] = 'Magnitude of horizontal velocity from Eastern and Northern vectors'
    adv_ds['Direction'].attrs['units'] = 'Degrees'
    adv_ds['Direction'].attrs['description'] = 'True direction of horizontal velocity (direction of vector added to the heading of the instrument)'
    adv_ds['Heading'].attrs['units'] = 'Degrees'
    adv_ds['Heading'].attrs['description'] = 'Direction of the instrument'
    adv_ds['Pitch'].attrs['units'] = 'Degrees'
    adv_ds['Roll'].attrs['units'] = 'Degrees'
    adv_ds['Temperature'].attrs['units'] = 'Celsius'
    adv_ds['Pressure'].attrs['units'] = 'dbar'
    adv_ds['Correlation_B1'].attrs['units'] = '%'
    adv_ds['Correlation_B1'].attrs['units'] = '%'
    adv_ds['Correlation_B1'].attrs['units'] = '%'
    adv_ds['Flag'].attrs['description'] = 'Flag value based on beam correlation, tilt, and pressure tests, higher score is lower quality data'
    adv_ds['Flag_pw'].attrs['description'] = 'Flag value for velocity acceleration and phase wrapping, higher score is higher likelihood of phase wrapping'

    return adv_ds

#===============================================================================================================================