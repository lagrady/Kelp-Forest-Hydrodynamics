import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import xarray as xr

def nan_filter(df):
    eas = df.filter(like='Eas').columns
    x = df[eas].isnull().sum(axis=1)
    nor = df.filter(like='Nor').columns
    y = df[eas].isnull().sum(axis=1)
    ver = df.filter(like='Ver').columns
    z = df[eas].isnull().sum(axis=1)

    df['nansum'] = x + y + z
    df = df[(df['nansum'] < (len(eas) * 3))]
    df = df.reset_index(drop=True)
    
    
    return df

def badbeam_test(x):
    return x > 30

def adcp_qc_test(df):
    print('Beginning Quality Control Tests')
    print(' ')
    print('Starting ADCP system requirements tests...')
    
    var_names = df.columns
    
    
    # BIT (Built in test) must be 0 to pass
    if 'BIT' in df.columns:
        df = df[(df['BIT'] == 0)]
        print('After BIT:', len(df))
    else:
        print('Could not run BIT test: No "BIT" column')
    df = nan_filter(df)
    
    # Battery must be above 150 counts to pass
    if 'Bat' in df.columns:
        df = df[(df['Bat'] >= 140)]
    else:
        print('Could not run Battery test: No "Bat" column')
    df = nan_filter(df)
    print('After Bat:', len(df)) 
    
    # Pitch and Roll should be within 5 degrees to minimize bias in velocity caused by tilting
    if 'Pit' in df.columns:
        df = df[(df["Pit"] < 5) & (df["Pit"] > -5)]
    else:
        print('Could not run Pitch test: No "Pit" column')
    df = nan_filter(df)
    print('After Pit:', len(df))
    
    if 'Rol' in df.columns:
        df = df[(df["Rol"] < 5) & (df["Rol"] > -5)]
    else:
        print('Could not run Roll test: No "Rol" column')
    df = nan_filter(df)
    print('After Rol:', len(df))
    
    # Compass heading should be from 0-359, anything else would imply the compass is malfunctioning
    if 'Hea' in df.columns:
        df = df[(df["Hea"] <= 359.99) & (df["Hea"] >= 0)]
    else:
        print('Could not run Heading test: No "Hea" column')
    

    # Reset index of dataframe and drop any profiles completely filled with NaN's
    #df = df.reset_index(drop=True)
    df = nan_filter(df)

    print('ADCP system requirements tests complete')
    print('After Heading:', len(df))
    print(' ')
    print('Starting Beam Quality tests...')
    
    # Create list of all velocity component names (THESE ARE NECESSARY VARIABLES)
    if all(value in df.columns for value in ['Eas1', 'Nor1', 'Ver1', 'Mag1', 'Dir1']):
        eas_cols = df.filter(like='Eas').columns # All East velocity columns
        nor_cols = df.filter(like='Nor').columns # All North velocity columns
        ver_cols = df.filter(like='Ver').columns # All Vertical velocity columns
        mag_cols = df.filter(like='Mag').columns # All Current Magnitude columns
        dir_cols = df.filter(like='Dir').columns # All Current Direction columns
    else:
        print('Cannot continue: Missing one or more required velocity variables (East, North, Vertical, Magnitude, Direction)')

    # Correllation test
    # Average correllation over all four beams must be greater than 65 to pass

    # Create list of average correllation column names
    if 'C51' in df.columns:
        ca_cols = df.filter(like='C5').columns

        # Iterates down the list of columns names corresponsing to C5, or the average correlation over all 4 beams
        for i in range(len(ca_cols)): # Determines which points of C5 < 66 for every depth bin(i)
            df.loc[df[ca_cols[i]] < 66, eas_cols[i]] = np.nan # If C5 is less than the threshold, velocity is unusable, and therefore converted to NaN
            df.loc[df[ca_cols[i]] < 66, nor_cols[i]] = np.nan
            df.loc[df[ca_cols[i]] < 66, mag_cols[i]] = np.nan
            df.loc[df[ca_cols[i]] < 66, ver_cols[i]] = np.nan
    
        print('Correllation test complete')
    else:
        print('Could not run correlation test: No "C5" column')

    # Percent Good test
    # Combined value of PG1 and PG4 must be greater than 75 to pass

    # Create list of percent good 1 and 4 column names
    if all(value in df.columns for value in ['PG11', 'PG41']):
        PG1_cols = df.filter(like='PG1').columns
        PG4_cols = df.filter(like='PG4').columns

        # Same concept as the correlation test, only the test determines if the sum of PG1 and PG2 is less than 75
        for i in range(len(PG1_cols)):
            df.loc[(df[PG1_cols[i]] + df[PG4_cols[i]]) < 75, eas_cols[i]] = np.nan
            df.loc[(df[PG1_cols[i]] + df[PG4_cols[i]]) < 75, nor_cols[i]] = np.nan
            df.loc[(df[PG1_cols[i]] + df[PG4_cols[i]]) < 75, mag_cols[i]] = np.nan
            df.loc[(df[PG1_cols[i]] + df[PG4_cols[i]]) < 75, ver_cols[i]] = np.nan
    
        print('Percent Good test complete')
    else:
        print('Could not run Percent Good test: No "PG1" and/or "PG4" columns')
    df = nan_filter(df)
    print(len(df))
    print(' ')
    print('Starting Current Magnitude/Direction tests...')
    # Horizontal current magnitude and direction test
    # Currents must be within an acceptable velocity range and logical direction (0-359 degrees)

    # Create list of binned current directions
    for i in range(len(dir_cols)): # Any datapoint with a direction outside of this range indicates a broken compass, and is therefore bad data
        df.loc[((df[dir_cols[i]] < 0) | (df[dir_cols[i]] > 359.9)), eas_cols[i]] = np.nan
        df.loc[((df[dir_cols[i]] < 0) | (df[dir_cols[i]] > 359.9)), nor_cols[i]] = np.nan
        df.loc[((df[dir_cols[i]] < 0) | (df[dir_cols[i]] > 359.9)), mag_cols[i]] = np.nan
        df.loc[((df[dir_cols[i]] < 0) | (df[dir_cols[i]] > 359.9)), ver_cols[i]] = np.nan

    print('Current Direction test complete')

    for i in range(len(mag_cols)): # Any current magnitude above this threshold is too high for the natural environment sampled
        df.loc[(df[mag_cols[i]] > 700), eas_cols[i]] = np.nan
        df.loc[(df[mag_cols[i]] > 700), nor_cols[i]] = np.nan
        df.loc[(df[mag_cols[i]] > 700), mag_cols[i]] = np.nan

    print('Current Magnitude test complete')
    df = nan_filter(df)
    print(len(df))
    print(' ')
    print('Starting Component Velocity tests...')
    # Component velocity tests
    # East/North/Vertical/Error velocities must be within acceptable range
    # Pertains to the error between horizontal velocity measurements

    # Create list of column names corresponding to error velocity
    if 'Err1' in df.columns:
        err_cols = df.filter(like='Err').columns

        for i in range(len(err_cols)): # Values of error velocity higher than 30 cm/s are generally seen as unreliable
            df.loc[(np.abs(df[mag_cols[i]]) > 300), eas_cols[i]] = np.nan
            df.loc[(np.abs(df[mag_cols[i]]) > 300), nor_cols[i]] = np.nan
            df.loc[(np.abs(df[mag_cols[i]]) > 300), mag_cols[i]] = np.nan

        print('Error Velocity test complete')
    else:
        print('Could not run Error Velocity test: No "Err" columns')

    for i in range(len(ver_cols)): # Any values of N/E/V velocity above these thresholds are too high for the environment sampled
        df.loc[((np.abs(df[eas_cols[i]]) > 500) | (np.abs(df[nor_cols[i]]) > 500) | (np.abs(df[ver_cols[i]]) > 200)),
                eas_cols[i]] = np.nan
        df.loc[((np.abs(df[eas_cols[i]]) > 500) | (np.abs(df[nor_cols[i]]) > 500) | (np.abs(df[ver_cols[i]]) > 200)),
                nor_cols[i]] = np.nan
        df.loc[((np.abs(df[eas_cols[i]]) > 500) | (np.abs(df[nor_cols[i]]) > 500) | (np.abs(df[ver_cols[i]]) > 200)),
                mag_cols[i]] = np.nan
        df.loc[((np.abs(df[eas_cols[i]]) > 500) | (np.abs(df[nor_cols[i]]) > 500) | (np.abs(df[ver_cols[i]]) > 200)),
                ver_cols[i]] = np.nan

    print('Component Velocity test complete')
    df = nan_filter(df)
    print(len(df))
    print(' ')
    print('Starting Velocity Shearing and Gradient tests...')
    # Horizontal Velocity shear test
    # If the horizontal velocity components change by an unrealistic amount in the span of a single depth bin, it may be indicative of high error
    # A general rule of thumb is that if the change in velocity between bins exceeds max velocity threshold of your site, it's unreliable
    # EX: My site has a max velocity of 50 cm/s, if the change in velocity from bin 1 to bin 2 (.25 m), is 50 cm/s, there's likely error
    for i in reversed(range(1, len(eas_cols))):
        df.loc[((np.abs(df[eas_cols[i]] - df[eas_cols[i-1]])) > 500) | 
                ((np.abs(df[nor_cols[i]] - df[nor_cols[i-1]])) > 500), eas_cols[i]] = np.nan
        df.loc[((np.abs(df[eas_cols[i]] - df[eas_cols[i-1]])) > 500) | 
                ((np.abs(df[nor_cols[i]] - df[nor_cols[i-1]])) > 500), nor_cols[i]] = np.nan
        df.loc[((np.abs(df[eas_cols[i]] - df[eas_cols[i-1]])) > 500) | 
                ((np.abs(df[nor_cols[i]] - df[nor_cols[i-1]])) > 500), mag_cols[i]] = np.nan
        df.loc[((np.abs(df[eas_cols[i]] - df[eas_cols[i-1]])) > 500) | 
                ((np.abs(df[nor_cols[i]] - df[nor_cols[i-1]])) > 500), ver_cols[i]] = np.nan
    
    print('Horizontal velocity shear test complete')

    # Do the same for Vertical Velocity over depth
    # Threshold for vertical velocity is almost always far less than horizontal velocity
    for i in reversed(range(1, len(ver_cols))):
        df.loc[((np.abs(df[ver_cols[i]] - df[ver_cols[i-1]])) > 200), ver_cols[i]] = np.nan
        df.loc[((np.abs(df[ver_cols[i]] - df[ver_cols[i-1]])) > 200), eas_cols[i]] = np.nan
        df.loc[((np.abs(df[ver_cols[i]] - df[ver_cols[i-1]])) > 200), nor_cols[i]] = np.nan
        df.loc[((np.abs(df[ver_cols[i]] - df[ver_cols[i-1]])) > 200), mag_cols[i]] = np.nan
    
    print('Vertical velocity shear test complete')

    # Current direction and magnitude with depth
    # If the magnitude changes by its max. threshold, there's likely error
    # If the direction changes by ~180 degrees over a single depth bin, there's like error
    for i in reversed(range(1, len(mag_cols))):
        df.loc[((np.abs(df[mag_cols[i]] - df[mag_cols[i-1]])) > 700) | 
                (((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) >= 160) & ((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) <= 200)),
                mag_cols[i]] = np.nan
    
        df.loc[((np.abs(df[mag_cols[i]] - df[mag_cols[i-1]])) > 700) | 
                (((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) >= 160) & ((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) <= 200)),
                eas_cols[i]] = np.nan
    
        df.loc[((np.abs(df[mag_cols[i]] - df[mag_cols[i-1]])) > 700) | 
                (((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) >= 160) & ((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) <= 200)),
                nor_cols[i]] = np.nan
    
        df.loc[((np.abs(df[mag_cols[i]] - df[mag_cols[i-1]])) > 700) | 
                (((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) >= 160) & ((np.abs(df[dir_cols[i]] - df[dir_cols[i-1]])) <= 200)),
                ver_cols[i]] = np.nan

    print('Horizontal Current Gradient test complete')
    df = nan_filter(df)
    print(len(df))
    print(' ')
    print('Starting Echo Intensity test... (This one takes the longest)')
    # Echo Amplitude test, i is depth bin and j is the beam (the ADCP has 4 beams)
    # If EA(i,j) - EA(i-1,j) is greater than 30, beam j and depth bin i is marked as a "bad beam"
    # If the number of bad beams for a depth bin is greater than the number of beams required for 3D velocity measurements (3 beams required)
    # then the measurements recorded at that depth are not reliable

    # If ABS(EA(i,j) - EA(i-1,j)) > 30 counts
    #     badbeam_counter = 1

    # If Number_of_beams(4) - badbeam_counter < 3D(3 good beams)
    #     North/East/Vertical velocity = NaN
    if all(value in df.columns for value in ['EA11', 'EA21', 'EA31', 'EA41']):
    
        ea1 = df.filter(like = 'EA1').columns
        ea2 = df.filter(like = 'EA2').columns
        ea3 = df.filter(like = 'EA3').columns
        ea4 = df.filter(like = 'EA4').columns

        for j in range(len(df)):
            if j % 10000 == 0: # Progress check every 10000 rows
                print('Currently on row:', j)
            for i in reversed(range(1, len(ea1))):
                badbeams = [] # Creates an array for storing EA values
                badbeams.append(np.abs(df[ea1[i]][j] - df[ea1[i-1]][j])) # Find the different between EA(i,j) - EA(i-1,j) for each beam
                badbeams.append(np.abs(df[ea2[i]][j] - df[ea2[i-1]][j])) # append the badbeams array with the value
                badbeams.append(np.abs(df[ea3[i]][j] - df[ea3[i-1]][j]))
                badbeams.append(np.abs(df[ea4[i]][j] - df[ea4[i-1]][j]))
                x = sum(badbeam_test(x) for x in badbeams) # Count how many values in the array exceed the threshold
                if x > 1:                                  # If the count exceeds 1 bad beam, convert velocities for that specific row and bin to NaN's
                    df[eas_cols[i]][j] = np.nan
                    df[nor_cols[i]][j] = np.nan
                    df[ver_cols[i]][j] = np.nan
                    df[mag_cols[i]][j] = np.nan
        print('Echo Intensity test complete')
    else:
        print('Could not run Echo Intensity test: No "EA" columns')
    print(' ')
    print('Removing all empty profiles...')
    df = nan_filter(df) # Filter out any completely empty profiles
    df = df.drop(['nansum'], axis=1) # Erase the now unusable 'nansum' column
    print('Final Number of Datapoints:', len(df))
    print('Quality control tests complete!')
    return df

def WinADCP_to_dataset(data_path):
    '''
    Load RDI ADCP text data file into pandas dataframe.
    
    INPUTS:
    data_path: path to series text file created by RDI software (WinADCP)
    
    
    RETURNS
    pandas dataframe
    
    EXAMPLE:
    import numpy as np
    from workhorse import rditext_to_dataset
    data_path = 'RDI_ADCP_data.txt'  
    df = rditext_to_dataset(data_path)
    '''
    pd.set_option("mode.chained_assignment", None)
    
    print('Opening file and extracting data...')
    print(' ')
    nhead = 16
    headerlines = list()

    with open(data_path) as f: 
        for i in range(nhead):
            line = f.readline()
            line = line.rstrip('\n')
            line = line.replace('\"','')
            headerlines.append(line)

    # create list of variable names from header line
    var_names = headerlines[12].split('\t')
    var_names[7] = 'HH.1'
    var_units = headerlines[13].split('\t')
    bin_num = headerlines[14].split('\t')
    for i,var in enumerate(var_names):
        var_names[i] = var_names[i].strip() + str(bin_num[i])
    var_units = headerlines[13].split('\t')
    
    print('Creating Pandas dataframe...')
    print(' ')
    df = pd.read_csv(data_path,skip_blank_lines=False,header=None,names=var_names,skiprows=nhead,delimiter='\t',skipinitialspace=True)
    
    datestr = '20'+df['YR'].map(str)+'-'+df['MO'].map(str)+'-'+df['DA'].map(str)
    timestr = df['HH'].map(str)+':'+df['MM'].map(str)+':'+df['SS'].map(str)+'.'+df['HH.1'].map(str)
    df['datetime'] = pd.to_datetime(datestr+' '+timestr,utc=True)
    
    df = df.drop(columns=['YR', 'MO', 'DA', 'HH', 'MM', 'SS', 'HH.1', ''])
    
    # Create metadata variables for the dataset from the textfile
    hrow, = np.where(['Pings/Ens' in s for s in headerlines])
    PingsPerEns = int(headerlines[hrow.squeeze()].split('\t')[1])

    hrow, = np.where(['Time/Ping' in s for s in headerlines])
    TimePerPing = headerlines[hrow.squeeze()].split(' = ')[1]

    hrow, = np.where(['First Ensemble Date' in s for s in headerlines])
    FirstEnsDate = headerlines[hrow.squeeze()].split(' = ')[1]

    hrow, = np.where(['First Ensemble Time' in s for s in headerlines])
    FirstEnsTime = headerlines[hrow.squeeze()].split(' = ')[1]

    hrow, = np.where(['Ensemble Interval' in s for s in headerlines])
    EnsInterval = float(headerlines[hrow.squeeze()].split(' = ')[1])

    hrow, = np.where(['1st Bin Range' in s for s in headerlines])
    BlankDist = float(headerlines[hrow.squeeze()].split(' = ')[1])

    hrow, = np.where(['Bin Size' in s for s in headerlines])
    BinSize = float(headerlines[hrow.squeeze()].split('\t')[1])

    binnumbers = np.unique(headerlines[14].split('\t'))[1:]
    nbins = np.max([int(binnum) for binnum in binnumbers])

    #ntime = len(df['datetime'])
    print('Dataframe complete')
    print(' ')
    # If user selects 'True' for data_qc, run the 'adcp_qc_test()' with the generated dataframe
    df = adcp_qc_test(df)
    
    print(' ')
    print('Generating xarray dataset...')
    print(' ')
    # Using variables from the cleaned dataframe, compress important data into arrays for xarray to read
    Pitch = df['Pit'].to_numpy()
    Roll = df['Rol'].to_numpy()
    Heading = df['Hea'].to_numpy()
    Temperature = df['Tem'].to_numpy()
    Depth = df['Dep'].to_numpy()
    eas = df.filter(like='Eas')
    eas = eas.to_numpy() / 1000
    nor = df.filter(like='Nor')
    nor = nor.to_numpy() / 1000
    ver = df.filter(like='Ver')
    ver = ver.to_numpy() / 1000
    mag = df.filter(like='Mag')
    mag = mag.to_numpy() / 1000
    direction = df.filter(like='Dir')
    direction = direction.to_numpy()
    cor = df.filter(like='C5')
    cor = cor.to_numpy()
    
    time = df['datetime']
    time = pd.to_datetime(time)
    
    BinDist = np.arange(BlankDist,BlankDist+nbins*BinSize,BinSize)
    
    # create coords
    rows = BinDist
    cols = time

    # put data into a dataset
    ds = xr.Dataset(
        data_vars=dict(
            Pitch = (["time"], Pitch),
            Roll = (["time"], Roll),
            Heading = (["time"], Heading),
            Temperature = (["time"], Temperature),
            Depth = (["time"], Depth),
            East = (["time", "BinDist"], eas),
            North = (["time", "BinDist"], nor),
            Vertical = (["time", "BinDist"], ver),
            Magnitude = (["time", "BinDist"], mag),
            Direction = (["time", "BinDist"], direction),
            Correlation = (["time", "BinDist"], cor)
        ),
        coords=dict(
            BinDist=(["BinDist"], rows),
            time=(["time"], cols),
        ),
        attrs=dict(description="Velocity in beam coordinates"),
    )
    
    # Add units and descriptions for variables
    ds['East'].attrs['units'] = 'm/s'
    ds['East'].attrs['description'] = 'Time series of Eastern velocity profiles'
    
    ds['North'].attrs['units'] = 'm/s'
    ds['North'].attrs['description'] = 'Time series of Northern velocity profiles'
    
    ds['Vertical'].attrs['units'] = 'm/s'
    ds['Vertical'].attrs['description'] = 'Time series of vertical velocity profiles'
    
    ds['Magnitude'].attrs['units'] = 'm/s'
    ds['Magnitude'].attrs['description'] = 'Time series of horizontal current magnitude profiles'
    
    ds['Direction'].attrs['units'] = 'Degrees'
    ds['Direction'].attrs['description'] = 'Direction associated with horizontal current magnitude profiles'
    
    ds['Correlation'].attrs['units'] = 'Counts (cnts)'
    ds['Correlation'].attrs['description'] = 'Average beam correlation profiles across all four beams. Counts are on a scale of 0-255'
    
    # Distance between bins and instrument
    ds['BinDist'] = ('bin',BinDist)
    ds['BinDist'].attrs['units'] = 'm'
    ds['BinDist'].attrs['description'] = 'distance between bins and instrument'
    
    # Add metadata
    ds.attrs['PingsPerEns'] = PingsPerEns
    ds.attrs['TimePerPing'] = TimePerPing    
    ds.attrs['First Ensemble Date'] = FirstEnsDate 
    ds.attrs['First Ensemble Time'] = FirstEnsTime
    ds.attrs['Ensemble Interval'] = EnsInterval
    ds.attrs['1st Bin Range'] = BlankDist
    ds.attrs['Bin Size'] = BinSize
    ds.attrs['RDI binary file'] = headerlines[1]
    ds.attrs['Instrument'] = headerlines[2]
    
    print('Conversion complete!')
    
    return ds