# OBSOLETE DATA IMPORT FUNCTIONS
# SAVING JUST IN CASE THEY ARE NEEDED
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
    
    return ds

#===============================================================================================================================
def eps_data_prep(ds, snr_cutoff, corr_cutoff, nmax, fs):
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

#===================================================================================================================================================
def rollingEpsCalc(vecDS, tempDS, nperseg=None, minGap=1, fNoise=3.1):

    ds = vecDS.copy(deep=True)
    tempData = tempDS.copy(deep=True)

    goodBursts = ds.burst.where(ds.dUp < .25, drop=True)
    burstList = np.unique(goodBursts)
    time_start = ds.time_start.where(ds.BurstCounter.isin(burstList), drop=True)
    
    print('Initializing arrays')
    #Initialize dimensions of frequency and spectrum using a burst from the dataset
    testBurst = ds.Up.where((ds.BurstNum.isin(burstList[0])) & (ds.Up.isnull()==False), drop = True)
    Ftest, Stest = welch(testBurst, fs=32, nperseg= nperseg, window='hann') # Vertical velocity spectra

    #Convert frequency to radian frequency and wavenumber
    T = 1/Ftest
    H = np.mean(ds.Depth)
    omega,k,Cph,Cg = wavedisp(T, H)

    J33 = ds.J33.where(ds.burst.isin(goodBursts), drop=True).values # Wavenumber space integral
    dUp = ds.dUp.where(ds.burst.isin(goodBursts), drop=True).values # Ratio of bad datapoints within the burst

    #Initialize arrays to hold all variables generated by the dataset
    fullSw = np.empty((len(burstList),len(omega))) #Full spectrum of velocity components
    fullSu = np.empty((len(burstList),len(omega)))
    fullSv = np.empty((len(burstList),len(omega)))
    maxSw = np.empty(len(burstList)) 
    minSw = np.empty(len(burstList))
    isrUpper = np.empty(len(burstList))
    isrLower = np.empty(len(burstList))
    Int = np.empty(len(burstList)) #Intercept of ISR fit
    IntErr = np.empty(len(burstList)) #Error of slope
    Mu = np.empty(len(burstList)) #Slope of ISR fit
    MuErr = np.empty(len(burstList)) #Error of slope
    Noise = np.empty(len(burstList)) #Magnitude of noise floor
    Rtest = np.empty(len(burstList)) #Ratio from Sww - Suv coherence test (for quality control)
    
    #Initialize arrays to hold all dissipation estimates and eps variables
    epsMag = np.empty(len(burstList)) # Mean of eps values over isr
    epsFitInt = np.empty(len(burstList)) # Intercept of eps estimate linear regression model
    epsFitSlope = np.empty(len(burstList)) # Slope
    epsFitR2val = np.empty(len(burstList)) # R2-value
    epsFitPval = np.empty(len(burstList)) # P-value
    
    #Initialize array to hold Ozmidov and Komogorov length scale values once eps has been estimated
    LenOz = np.empty(len(burstList))
    LenKol = np.empty(len(burstList))

    for b in enumerate(burstList):
        print('Evaluating burst '+str(b[0])+' of '+str(len(burstList)))
        #Identify burst number to be evaluated
        burstNumber = b[1]
        
        #Retrieve variables from the burst time period
        burstTime = ds.time.where(ds.BurstNum.isin(burstNumber), drop = True)
        burstTemp = tempData.Temperature.sel(depth=5,time=slice(burstTime[0],burstTime[-1])) #Temperature recorded at depth of adv head
        burstPressure = ds.Pressure.where(ds.BurstNum.isin(burstNumber), drop=True)
        
        burstUp = ds.Up.where((ds.BurstNum.isin(burstNumber)) & (ds.Up.isnull()==False), drop = True)
        burstU = ds.Primary.where((ds.BurstNum.isin(burstNumber)) & (ds.Primary.isnull()==False), drop = True)
        burstV = ds.Secondary.where((ds.BurstNum.isin(burstNumber)) & (ds.Secondary.isnull()==False), drop = True)
    
        # Generate vertical and horizontal velocity spectra
        Fw, Sw = welch(burstUp, fs=32, nperseg= nperseg, window='hann') # Vertical velocity spectra
        Fu, Su = welch(burstU, fs=32, nperseg= nperseg, window='hann') # Horiztonal velocity spectra
        Fv, Sv = welch(burstV, fs=32, nperseg= nperseg, window='hann')

        # Calculate and convert pressure spectra to vertical velocity to find lower cutoff frequency
        pressure = burstPressure * 10000 #Convert from dBar to Pascals for spectra conversion via linear wave theory
        Fp, Sp = welch(pressure, fs=32, nperseg= nperseg, window='hann') # Pressure spectra

        rho = np.mean(tempData.Rho.sel(depth=slice(4,7),time=slice(burstTime[0],burstTime[-1]))) + 1000 #Density at adv during the burst
        g = 9.8 # Gravity
        z = pressure/(rho*g) # Depth (m): the recorded pressure converted to meters of seawater
        H = np.mean(z) + .578 # Sea level height (m): mean pressure detected by the pressure sensor plus the height of sensor from the bottom
        Zp = -(np.mean(z)) # Depth of pressure sensor (m)
        Zv = (-H) + .824 # Depth of velocity sensor (m): Sea level height plus the height of the velocity transducers from the bottom

        # Generate empty arrays for p' and w' values
        p_prime = np.empty(len(omega))
        w_prime = np.empty(len(omega))

        for j in range(len(omega)): # For loop iterates over all values of omega
            p_prime[j] = (rho*g)*(np.cosh(k[j]*(Zp+H))/np.cosh(k[j]*H))
            w_prime[j] = (-omega[j])*(np.sinh(k[j]*(Zv+H)))/(np.sinh(k[j]*H))
        scaleFactor = w_prime**2 / p_prime**2
    
        #Calculate the equivalent Sw spectra from Sp
        Sw_prime = (Sp * scaleFactor)

        #Define the lower cutoff frequency as the end of surface gravity wave band
        try:
            lfc = argrelextrema(Sw_prime, np.less, order=10)[0][0]
        except IndexError:
            #Any issue with indexes should yield a default wave cutoff of .5 Hz
            #.5 is a conservative estiamte of where the wave band ends, but still preceeds most of the potential ISR
            lfc = np.where(Fw == .5)[0][0]

        #Define the upper cutoff frequency as the beginning of the noise floor
        Fn = np.where(Ftest == fNoise)[0][0] #Minimum frequency before the noise floor begins
        noiseFloor = np.mean(Sw[Fn:]) #Noise floor magnitude of vertical velocity spectra
    
        #Define ufc as final point above the noise floor
        ufc = np.where(Sw < noiseFloor)[0][0] 
    
        #Generate list of every possible boundary combination within lower and upper cutoff frequencies
        #ISR should be at least 1hz in length (which is forgiving compared to 2.5hz in Gerbi et al.)

        minISRGap = int((minGap/np.diff(Ftest)[0])) #Calculate a gap compatible with the spectrum index
        startRange = np.arange(lfc, ufc) #Initialize an array of all frequencies within lfc and ufc
        bounds = [] #List of all ISR boundaries
        if (ufc-lfc) > minISRGap:
            #Create an array that is offset by the minimum gap
            iteratorRange = np.arange(lfc+minISRGap, ufc) #First combination of points will be lfc : lfc + 1Hz gap
        
            for i in range(ufc-(lfc+minISRGap)):
                for j in range(len(iteratorRange)):
                    #For each combination, record the boundaries into a list
                    bounds.append((startRange[i], iteratorRange[j]))
                
                #Each iteration shortens iterator range by 1 to prevent repeat and backwards combinations 
                iteratorRange = iteratorRange[1:ufc] 
    
        # If the range is shorter than 1hz, boundaries become the wave cutoff frequency and the noise floor
        else:
            bounds.append((lfc, ufc))
            
        lfc = bounds[0][0]
        ufc = bounds[0][1]
        
        #Fit all combinations of boundaries to evaluate where the "best" ISR range is located
        try:
            pars, cov = curve_fit(f=power_law, xdata=omega[lfc:ufc+1], #Power law fit between lfc and ufc
                              ydata=Sw[lfc:ufc+1], p0=[0, 0], bounds=(-np.inf, np.inf), maxfev=10000)

            testInt = pars[0] #Fit intercept
            testIntErr = np.sqrt(np.diag(cov))[0] #intercept error
            testMu = pars[1] #Fit slope
            testMuErr = np.sqrt(np.diag(cov))[1] #Slope error
    
        #If curve_fit can't fit properly, use 9999 as error values
        except RuntimeError:
            lfc = bounds[0][0]
            ufc = bounds[0][1]
            #Errors should be set to unrealistic values
            testInt = 99999
            testIntErr = 99999
            testMu = 99999
            testMuErr = 99999
    
        #Begin iteration and optimization of ISR fits
        for i in np.arange(1,len(bounds)):
            muDiff1 = np.abs((-5/3)-testMu)
            try:
                pars, cov = curve_fit(f=power_law, xdata=omega[bounds[i][0]:bounds[i][1]],
                                      ydata=Sw[bounds[i][0]:bounds[i][1]], p0=[0, 0], bounds=(-np.inf, np.inf), maxfev=10000)
                
                muDiff2 = np.abs((-5/3)-pars[1]) #Assess misfit between fit slope and -5/3 slope
                MuErr2 = np.sqrt(np.diag(cov))[1] #Error of the current fit
                
                #Run QC guantlet to test if fit passes all tests
                
                
                
                
                
                if ((muDiff2 <= muDiff1) & (MuErr2 <= .15)):
                    lfc = bounds[i][0]
                    ufc = bounds[i][1]
                    testInt = pars[0] #Fit intercept
                    testIntErr = np.sqrt(np.diag(cov))[0] #intercept error
                    testMu = pars[1] #Fit slope
                    testMuErr = np.sqrt(np.diag(cov))[1] #Slope error
                else:
                    continue
        
            except RuntimeError:
                continue
        #Using the new boundaries of the ISR, record min/max SW and the boundary positions in the spectrum
        fullSw[b[0]] = Sw
        fullSu[b[0]] = Su
        fullSv[b[0]] = Sv
    
        maxSw[b[0]] = Sw[lfc]
        isrLower[b[0]] = lfc
    
        minSw[b[0]] = Sw[ufc]
        isrUpper[b[0]] = ufc
    
        Int[b[0]] = testInt #Intercept of ISR fit
        IntErr[b[0]] = testIntErr #Error of slope
        Mu[b[0]] = testMu #Slope of ISR fit
        MuErr[b[0]] = testMuErr #Error of slope
    
        # Find the Sw - Suv coherence ratio R
        RNoise = np.mean(Su[Fn:] + Sv[Fn:]) # Average all frequencies above horizontal noise floor 
        RFreq = Ftest[lfc:ufc+1] #The frequencies of the new ISR
        RSuv = (Su[lfc:ufc+1]+Sv[lfc:ufc+1])-RNoise #Horizontal velocity spectra within ISR, minus the noise
        RSw = Sw[lfc:ufc+1] #Vertical velocity spectra within ISR
    
        #Plug into R ratio formula (Eq. 8 from Fedderson, 2010)
        RNumerator = (12/21) * np.mean((RFreq**(5/3)) * RSuv) # Numerator
        RDenominator = np.mean((RFreq**(5/3)) * RSw) # Denominator
        R = RNumerator/RDenominator
    
        #Populate arrays
        Noise[b[0]] = noiseFloor
        Rtest[b[0]] = R
    
        #Estimate turbulent dissipation (Epsilon/eps)
        alpha = 1.5 # Kolomogorov constant
        isrOmega = omega[lfc:ufc+1] #Radian frequency range
        S33 = Sw[lfc:ufc+1] #Vertical velocity spectra within ISR
    
        #Dissipation formula (Eq. A14 from Gerbi et al., 2009)
        eps = ((S33 * (isrOmega**(5/3)))/(alpha * J33[b[0]]))**(3/2) #Returns array of eps estimates across ISR
    
        #Fir a linear regression to eps estimates
        res = stats.linregress(isrOmega, eps)
    
        #Populate arrays
        epsMag[b[0]] = np.mean(eps) #Mean value of eps for the entire burst
        epsFitInt[b[0]] = res.intercept #Linear regression intercept
        epsFitSlope[b[0]] = res.slope #Linear regression slope
        epsFitR2val[b[0]] = res.rvalue**2 #R2 value of linear regression
        epsFitPval[b[0]] = res.pvalue #P-value of linear regression (used for qc)
        
        #Ozmidov length scale
        rho1 = tempData.Rho.sel(depth=4,time=slice(burstTime[0],burstTime[-1])).mean().values + 1000 #Depths 4 and 6 correspond to 9.1 and 9.7m respectively
        rho2 = tempData.Rho.sel(depth=6,time=slice(burstTime[0],burstTime[-1])).mean().values + 1000
        dRho = np.abs(rho2 - rho1)/.6 #Change in density over depth
        rhoBar = tempData.Rho.sel(time=slice(burstTime[0],burstTime[-1])).mean().values + 1000 #Mean density during the burst
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
            maxSw = (["bNum"], maxSw),
            minSw = (["bNum"], minSw),
            lowBound = (['bNum'], isrLower),
            highBound = (['bNum'], isrUpper),
            L_Ozmidov = (['bNum'], LenOz),
            L_Kolmogorov = (['bNum'], LenKol),
            J33 = (["bNum"], J33),
            Int = (["bNum"], Int),
            IntErr = (["bNum"], IntErr),
            Mu = (["bNum"], Mu),
            MuErr = (["bNum"], MuErr),
            epsMag = (["bNum"], epsMag),
            R_ratio = (["bNum"], Rtest),
            NoiseFloor = (["bNum"], Noise),
            epsFitInt = (["bNum"], epsFitInt),
            epsFitSlope = (["bNum"], epsFitSlope),
            epsFitR2val = (["bNum"], epsFitR2val),
            epsFitPval = (["bNum"], epsFitPval),
            dUp = (['bNum'], dUp)
        ),
        coords=dict(
            time_start=(["time_start"], time_start),
            bNum=(["bNum"], burstList),
            omega=(["omega"], omega),
            frequency=(["frequency"], Ftest),
            wavenumber=(["wavenumber"], k)
        ),
        attrs=dict(description="All data from estimated turbulent dissipation"),
    )

    print('Testing epsilon values')
    intFlag = xr.zeros_like(epsDS.bNum)
    noiseFlag = xr.zeros_like(epsDS.bNum)
    slopeFlag = xr.zeros_like(epsDS.bNum)
    linRegFlag = xr.zeros_like(epsDS.bNum)
    RFlag = xr.zeros_like(epsDS.bNum)

    # Isr fit intercept test
    intFlag = intFlag + xr.where((epsDS.Int > epsDS.IntErr), 0, 1) 

    # Noise floor test
    noiseFlag = noiseFlag + xr.where(epsDS.maxSw/2 > epsDS.NoiseFloor, 0, 1)

    # Isr slope test
    lowMu = epsDS.Mu - (2*epsDS.MuErr) - .06
    highMu = epsDS.Mu + (2*epsDS.MuErr) + .06

    slopeFlag = slopeFlag + xr.where((lowMu < (-5/3)) & (highMu > (-5/3)), 0, 1)

    # eps estimate linear regression test
    linRegFlag = linRegFlag + xr.where(epsDS.epsFitPval > .01, 0, 1)

    # Sww - Suv unity test
    RFlag = RFlag + xr.where((epsDS.R_ratio >=.5) & (epsDS.R_ratio <=2), 0, 1)

    epsDS['intFlag'] = (['bNum'],intFlag)
    epsDS['noiseFlag'] = (['bNum'],noiseFlag)
    epsDS['slopeFlag'] = (['bNum'],slopeFlag)
    epsDS['linRegFlag'] = (['bNum'],linRegFlag)
    epsDS['RFlag'] = (['bNum'],RFlag)
    
    return epsDS

#===================================================================================================================================================
def fixedEpsCalc(data, Jlm, lowerBound, upperBound):
    
    ds = data.copy(deep=True)
    goodBurst
    burstList = np.unique(ds.BurstNum)

    #Initialize dimensions of frequency and spectrum using a burst from the dataset
    testBurst = ds.Up.where((ds.BurstNum.isin(burstList[0])) & (ds.Up.isnull()==False), drop = True)
    Ftest, Stest = welch(testBurst, fs=32, nperseg= 2240, window='hann') # Vertical velocity spectra

    #Convert frequency to radian frequency and wavenumber
    T = 1/Ftest
    H = np.mean(ds.Depth)
    omega,k,Cph,Cg = wavedisp(T, H)

    #Define the lower and upper frequency cutoffs for the ISR
    lfc = np.where(Ftest == lowerBound)[0][0]
    ufc = np.where(Ftest == upperBound)[0][0]
    
    #Initialize arrays to hold all variables generated by the dataset
    fullSw = np.empty((len(burstList),len(omega))) #Full spectrum of velocity components
    fullSu = np.empty((len(burstList),len(omega)))
    fullSv = np.empty((len(burstList),len(omega)))
    maxSw = np.empty(len(burstList)) 
    minSw = np.empty(len(burstList))
    Int = np.empty(len(burstList)) #Intercept of ISR fit
    IntErr = np.empty(len(burstList)) #Error of slope
    Mu = np.empty(len(burstList)) #Slope of ISR fit
    MuErr = np.empty(len(burstList)) #Error of slope
    Noise = np.empty(len(burstList)) #Magnitude of noise floor
    Rtest = np.empty(len(burstList)) #Ratio from Sww - Suv coherence test (for quality control)
    
    #Initialize arrays to hold all dissipation estimates and eps variables
    J33 = Jlm # Wavenumber space integral
    epsMag = np.empty(len(burstList)) # Mean of eps values over isr
    epsFitInt = np.empty(len(burstList)) # Intercept of eps estimate linear regression model
    epsFitSlope = np.empty(len(burstList)) # Slope
    epsFitR2val = np.empty(len(burstList)) # R2-value
    epsFitPval = np.empty(len(burstList)) # P-value

    for i in enumerate(burstList):
    
        print('Evaluating burst '+str(i[1]))
    
        #Find all non-nan velocity data
        burstNumber = i[1]
        burstUp = ds.Up.where((ds.BurstNum.isin(burstNumber)) & (ds.Up.isnull()==False), drop = True)
        burstU = ds.Primary.where((ds.BurstNum.isin(burstNumber)) & (ds.Primary.isnull()==False), drop = True)
        burstV = ds.Secondary.where((ds.BurstNum.isin(burstNumber)) & (ds.Secondary.isnull()==False), drop = True)
    
        #Calculate spectra from Hann windowed data over 60s overlapping segments
        Fw, Sw = welch(burstUp, fs=32, nperseg= 2240, window='hann') # Vertical velocity spectra
        fullSw[i[0]] = Sw
        maxSw[i[0]] = Sw[lfc]
        minSw[i[0]] = Sw[ufc]
        Fu, Su = welch(burstU, fs=32, nperseg= 2240, window='hann') # U velocity spectra
        fullSu[i[0]] = Su
        Fv, Sv = welch(burstV, fs=32, nperseg= 2240, window='hann') # V velocity spectra
        fullSv[i[0]] = Sv

        #Fit a power law curve using the fixed ISR boundaries provided
        try:
            pars, cov = curve_fit(f=power_law, xdata=omega[lfc:ufc+1],
                                  ydata=Sw[lfc:ufc+1], p0=[0, 0], bounds=(-np.inf, np.inf), maxfev=10000)
        
            #Record the slope and intercept, as well as errors for both
            Int[i[0]] = pars[0] #Fit intercept
            IntErr[i[0]] = np.sqrt(np.diag(cov))[0] #intercept error
            Mu[i[0]] = pars[1] #Fit slope
            MuErr[i[0]] = np.sqrt(np.diag(cov))[1] #Slope error
        
        #Some fits exceed the maxfev; in this case, return nan's for all and start evaluating the next burst
        except:
            Int[i[0]] = np.nan
            IntErr[i[0]] = np.nan
            Mu[i[0]] = np.nan
            MuErr[i[0]] = np.nan
            Noise[i[0]] = np.nan
            Rtest[i[0]] = np.nan
            epsMag[i[0]] = np.nan
            epsFitInt[i[0]] = np.nan
            epsFitSlope[i[0]] = np.nan
            epsFitR2val[i[0]] = np.nan
            epsFitPval[i[0]] = np.nan
            continue
        
        # Find the Sw - Suv coherence ratio R
        Fn = np.where(Fw == 4)[0][0] #Minimum frequency before the noise floor
        noiseFloor = np.mean(Sw[Fn:]) #Noise floor of vertical velocity spectra (for qc)
        RNoise = np.mean(Su[Fn:] + Sv[Fn:]) # Average all frequencies above horizontal noise floor 
        RFreq = Fw[lfc:ufc+1] #The frequencies of the fixed ISR
        RSuv = (Su[lfc:ufc+1]+Sv[lfc:ufc+1])-RNoise #Horizontal velocity spectra within fixed ISR, minus the noise
        RSw = Sw[lfc:ufc+1] #Vertical velocity spectra within fixed ISR
    
        #Plug into R ratio formula (Eq. 8 from Fedderson, 2010)
        RNumerator = (12/21) * np.mean((RFreq**(5/3)) * RSuv) # Numerator
        RDenominator = np.mean((RFreq**(5/3)) * RSw) # Denominator
        R = RNumerator/RDenominator
    
        #Populate arrays arrays
        Noise[i[0]] = noiseFloor
        Rtest[i[0]] = R
    
        #Estimate turbulent dissipation (Epsilon/eps)
        alpha = 1.5 # Kolomogorov constant

        isrOmega = omega[lfc:ufc+1] #Radian frequency range
        S33 = Sw[lfc:ufc+1] #Vertical velocity spectra within ISR
    
        #Dissipation formula (Eq. A14 from Gerbi et al., 2009)
        eps = ((S33 * (isrOmega**(5/3)))/(alpha * J33[i[0]]))**(3/2) #Returns array of eps estimates across ISR
    
        #Fir a linear regression to eps estimates
        res = stats.linregress(isrOmega, eps)
    
        #Populate arrays
        epsMag[i[0]] = np.mean(eps) #Mean value of eps for the entire burst
        epsFitInt[i[0]] = res.intercept #Linear regression intercept
        epsFitSlope[i[0]] = res.slope #Linear regression slope
        epsFitR2val[i[0]] = res.rvalue**2 #R2 value of linear regression
        epsFitPval[i[0]] = res.pvalue #P-value of linear regression (used for qc)
    
    # Create a new dataset with all relevant variables and epsilon values
    print('Creating Dataset')
    epsDS = xr.Dataset(
        data_vars=dict(
            Su = (["bNum","omega"], fullSu),
            Sv = (["bNum","omega"], fullSv),
            Sw = (["bNum","omega"], fullSw),
            maxSw = (["bNum"], maxSw),
            minSw = (["bNum"], minSw),
            J33 = (["bNum"], J33),
            Int = (["bNum"], Int),
            IntErr = (["bNum"], IntErr),
            Mu = (["bNum"], Mu),
            MuErr = (["bNum"], MuErr),
            epsMag = (["bNum"], epsMag),
            R_ratio = (["bNum"], Rtest),
            NoiseFloor = (["bNum"], Noise),
            epsFitInt = (["bNum"], epsFitInt),
            epsFitSlope = (["bNum"], epsFitSlope),
            epsFitR2val = (["bNum"], epsFitR2val),
            epsFitPval = (["bNum"], epsFitPval),
                
        ),
        coords=dict(
            bNum=(["bNum"], burstList),
            omega=(["omega"], omega),
            frequency=(["frequency"], Ftest),
            wavenumber=(["wavenumber"], k)
        ),
        attrs=dict(description="All data from estimated turbulent dissipation", LowerBoundary=str(lfc),
              UpperBoundary=str(ufc)),
    )

    print('Testing epsilon values')
    epsFlag = xr.zeros_like(epsDS.bNum)

    # Isr fit intercept test
    epsFlag = epsFlag + xr.where((epsDS.Int > epsDS.IntErr), 0, 1) 

    # Noise floor test
    epsFlag = epsFlag + xr.where((epsDS.maxSw/2) > epsDS.NoiseFloor, 0, 1)

    # Isr slope test
    lowMu = epsDS.Mu - (2*epsDS.MuErr) - .06
    highMu = epsDS.Mu + (2*epsDS.MuErr) + .06

    epsFlag = epsFlag + xr.where((lowMu < (-5/3)) & (highMu > (-5/3)), 0, 1)

    # eps estimate linear regression test
    epsFlag = epsFlag + xr.where(epsDS.epsFitPval > .05, 0, 1)

    # Sww - Suv unity test
    epsFlag = epsFlag + xr.where((epsDS.R_ratio >=.5) & (epsDS.R_ratio <=2), 0, 1)

    epsDS['epsFlag'] = (['bNum'],epsFlag)
    
    return epsDS