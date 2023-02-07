import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def vector_to_df(datfile, senfile, samp_frequency):
    # Create column names for pandas dataframe
    # 'dat_cols' pertains to the default .DAT file from the vector, 'sen_cols' pertains to the default .SEN file
    print('Importing data...')
    dat_cols = ["Burst_counter", "Ensemble_counter", "Velocity_East(m/s)", "Velocity_North(m/s)", "Velocity_Up(m/s)", "Amplitude_B1(counts)", 
                "Amplitude_B2(counts)", "Amplitude_B3(counts)", "SNR_B1(dB)", "SNR_B2(dB)", "SNR_B3(dB)", "Correlation_B1(%)", "Correlation_B2(%)", 
                "Correlation_B3(%)", "Pressure(dbar)", "AnalogInput1", "AnalogInput2", "Checksum(1=failed)"]
    sen_cols = ["Month", "Day", "Year", "Hour", "Minute", "Second", "Error_code", "Status_code", "Battery_voltage", "Soundspeed", "Heading", "Pitch", "Roll", "Temperature", 
                "Analog_input", "Checksum"]
    
    # Create separate dataframes for .dat file and .sen file
    dat = pd.read_csv(datfile, delimiter='\s+', names = dat_cols)
    sen = pd.read_csv(senfile, delimiter='\s+', names = sen_cols)
    
    print('Adding sensor log information...')
    i_range = np.int(len(dat)/samp_frequency)
    
    sen = sen.iloc[:i_range]
    dat = dat.iloc[:i_range * samp_frequency]
    
    dates = np.empty(shape=(len(dat)), dtype='datetime64[ns]')
    Soundspeed = np.empty(len(dat))
    Heading = np.empty(len(dat))
    Pitch = np.empty(len(dat))
    Roll = np.empty(len(dat))
    Temperature = np.empty(len(dat))
    
    if samp_frequency == 32:
        t_step = '31.25L'
    elif samp_frequency == 16:
        t_step = '62.5L'
    elif samp_frequency == 8:
        t_step = '125L'
    elif samp_frequency == 4:
        t_step = '250L'
    elif samp_frequency == 2:
        t_step = '500L'
    elif samp_frequency == 1:
        t_step = '1000L'

    for i in range(0, i_range, 1):
        if i % 50000 == 0: # Progress check every 50000 rows
            print('Currently on row', i, 'of', i_range)
        dates[i*samp_frequency:(i+1)*samp_frequency] = pd.date_range(start=(str(sen.iloc[i,0])+'/'+ str(sen.iloc[i,1])+'/'+ str(sen.iloc[i,2])+' '+ str(sen.iloc[i,3])+':'+ 
                                                     str(sen.iloc[i,4])+':'+ str(sen.iloc[i,5])), periods = samp_frequency, freq = t_step)
        Soundspeed[i*samp_frequency:(i+1)*samp_frequency] = sen.iloc[i, 9]
        Heading[i*samp_frequency:(i+1)*samp_frequency] = sen.iloc[i, 10]
        Pitch[i*samp_frequency:(i+1)*samp_frequency] = sen.iloc[i, 11]
        Roll[i*samp_frequency:(i+1)*samp_frequency] = sen.iloc[i, 12]
        Temperature[i*samp_frequency:(i+1)*samp_frequency] = sen.iloc[i, 13]
        
    dat.insert(0, 'Datetime', dates)
    dat.Datetime = pd.to_datetime(dat.Datetime)
    
    dat['Soundspeed'] = Soundspeed
    dat['Heading'] = Heading
    dat['Pitch'] = Pitch
    dat['Roll'] = Roll
    dat['Temperature'] = Temperature

    return dat

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
    df.loc[(df["Correlation_B1(%)"] < 80) & (df["Correlation_B1(%)"] >= 70), "Flag"] = df.loc[(df["Correlation_B1(%)"] < 80) & (df["Correlation_B1(%)"] >= 70), "Flag"] + 1
    df.loc[df["Correlation_B1(%)"] < 70, "Flag"] = df.loc[df["Correlation_B1(%)"] < 70, "Flag"] + 20 
    # Flag score is set significantly higher since a single beam failing the correlation test needs to visibly stand out while pruning the data

    # Beam 2 correlation
    df.loc[(df["Correlation_B2(%)"] < 80) & (df["Correlation_B2(%)"] >= 70), "Flag"] = df.loc[(df["Correlation_B2(%)"] < 80) & (df["Correlation_B2(%)"] >= 70), "Flag"] + 1
    df.loc[df["Correlation_B2(%)"] < 70, "Flag"] = df.loc[df["Correlation_B2(%)"] < 70, "Flag"] + 20

    # Beam 3 correlation
    df.loc[(df["Correlation_B3(%)"] < 80) & (df["Correlation_B3(%)"] >= 70), "Flag"] = df.loc[(df["Correlation_B3(%)"] < 80) & (df["Correlation_B3(%)"] >= 70), "Flag"] + 1
    df.loc[df["Correlation_B3(%)"] < 70, "Flag"] = df.loc[df["Correlation_B3(%)"] < 70, "Flag"] + 20
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
    df.loc[df['Pressure(dbar)'] < pressure_threshold, "Flag"] = df.loc[df['Pressure(dbar)'] < pressure_threshold, "Flag"] + 4
    #===============================================================================================================================
    # Acceleration tests
    # Filters high acceleration between consecutive velocities
    # Accelerations > 1m/s2 may be indicative of low beam correlation or phase wrapping if recorded velocity exceeds nominal velocity range
    
    # Create lists of accelerations for ENU velocities
    df['Nor_diff'] = df['Velocity_North(m/s)'].diff() # List of differences between consecutive velocities (accelerations)
    df['Nor_absdiff'] = np.abs(df['Velocity_North(m/s)'].diff()) # Absolute value of accelerations (makes future screening easier)

    df['Eas_diff'] = df['Velocity_East(m/s)'].diff()
    df['Eas_absdiff'] = np.abs(df['Velocity_East(m/s)'].diff())

    df['Up_diff'] = df['Velocity_Up(m/s)'].diff()
    df['Up_absdiff'] = np.abs(df['Velocity_Up(m/s)'].diff())

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
    adv_hvel = np.sqrt(((adv['Velocity_East(m/s)'])**2) + ((adv['Velocity_North(m/s)'])**2))
    adv_dir = np.arctan(adv['Velocity_North(m/s)']/adv['Velocity_East(m/s)']) + adv['Heading']

    # put data into a dataset
    adv_ds = xr.Dataset(
        data_vars=dict(
            Burst_number = (["time"], adv['Burst_counter']),
            East = (["time"], adv['Velocity_East(m/s)']),
            North = (["time"], adv['Velocity_North(m/s)']),
            Vertical = (["time"], adv['Velocity_Up(m/s)']),
            Magnitude = (["time"], adv_hvel),
            Direction = (["time"], adv_dir),
            Heading = (["time"], adv['Heading']),
            Pitch = (["time"], adv['Pitch']),
            Roll = (["time"], adv['Roll']),
            Temperature = (["time"], adv['Temperature']),
            Pressure = (["time"], adv['Pressure(dbar)']),
            Correlation_B1 = (["time"], adv['Correlation_B1(%)']),
            Correlation_B2 = (["time"], adv['Correlation_B2(%)']),
            Correlation_B3 = (["time"], adv['Correlation_B3(%)']),
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