import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import xarray as xr

def WinADCP_to_dataset(data_path, depthOffset = 0):
    '''
    Load RDI ADCP text data file into xarray dataset.
    
    INPUTS:
    data_path: path to series text file created by RDI software (WinADCP)
    
    
    RETURNS
    xarray dataset
    
    EXAMPLE:
    import numpy as np
    import pandas as pd
    import xarray as xr
    import sentinnel_tools as st
    data_path = 'RDI_ADCP_data.txt'  
    ds = st.WinADCP_to_dataset(data_path)
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

    print('Generating xarray dataset...')
    print(' ')
    # Using variables from the cleaned dataframe, compress important data into arrays for xarray to read
    time = df['datetime']
    time = pd.to_datetime(time)
    BinDist = np.arange(BlankDist,BlankDist+nbins*BinSize,BinSize) + depthOffset

    Pitch = df['Pit'].to_numpy()
    Roll = df['Rol'].to_numpy()
    Heading = df['Hea'].to_numpy()
    Temperature = df['Tem'].to_numpy()
    Depth = df['Dep'].to_numpy()
    Bat = df['Bat'].to_numpy()
    eas = df.filter(like='Eas')
    eas = eas.to_numpy().T / 1000
    nor = df.filter(like='Nor')
    nor = nor.to_numpy().T / 1000
    ver = df.filter(like='Ver')
    ver = ver.to_numpy().T / 1000
    mag = df.filter(like='Mag')
    mag = mag.to_numpy().T / 1000
    direction = df.filter(like='Dir')
    direction = direction.to_numpy().T
    
    err = df.filter(like='Err')
    err = err.to_numpy().T / 1000
    
    cor = df.filter(like='C5')
    cor = cor.to_numpy().T
    
    pg1 = df.filter(like='PG1')
    pg1 = pg1.to_numpy().T
    pg4 = df.filter(like='PG4')
    pg4 = pg4.to_numpy().T
    
    ea1 = df.filter(like='EA1')
    ea1 = ea1.to_numpy().T
    ea2 = df.filter(like='EA2')
    ea2 = ea2.to_numpy().T
    ea3 = df.filter(like='EA3')
    ea3 = ea3.to_numpy().T
    ea4 = df.filter(like='EA4')
    ea4 = ea4.to_numpy().T
    ea5 = (ea1 + ea2 + ea3 + ea4)/4
    
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
            Battery = (["time"], Bat),
            East = (["BinDist", "time"], eas),
            North = (["BinDist", "time"], nor),
            Vertical = (["BinDist", "time"], ver),
            Magnitude = (["BinDist", "time"], mag),
            Direction = (["BinDist", "time"], direction),
            Error_velocity = (["BinDist", "time"], err),
            Correlation = (["BinDist", "time"], cor),
            PG1 = (["BinDist", "time"], pg1),
            PG4 = (["BinDist", "time"], pg4),
            EA1 = (["BinDist", "time"], ea1),
            EA2 = (["BinDist", "time"], ea2),
            EA3 = (["BinDist", "time"], ea3),
            EA4 = (["BinDist", "time"], ea4),
            EA5 = (["BinDist", "time"], ea5)
            
        ),
        coords=dict(
            BinDist=(["BinDist"], rows),
            time=(["time"], cols),
        ),
    )
    
    if 'BIT' in df.columns:
        BIT = df['BIT'].to_numpy()
        ds['BIT'] = (["time"], BIT)
        ds['BIT'].attrs['description'] = "The ADCP's own built in test metric. 0 indicate a pass, 1 indicates an error"
    
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
    
    ds['Error_velocity'].attrs['units'] = 'm/s'
    ds['Error_velocity'].attrs['description'] = "The difference between the vertical velocity measured by opposing pairs of beams (b1/b3, b2/b4)"
    
    ds['Correlation'].attrs['units'] = 'Counts (cnts)'
    ds['Correlation'].attrs['description'] = 'Average beam correlation profiles across all four beams. Counts are on a scale of 0-255'
    
    ds['PG1'].attrs['units'] = '%'
    ds['PG1'].attrs['description'] = "The 'percent good' of beam one, an indicator of measurement quality"
    
    ds['PG4'].attrs['units'] = '%'
    ds['PG4'].attrs['description'] = "The 'percent good' of beam four, an indicator of measurement quality"
    
    ds['EA1'].attrs['units'] = 'Cnt'
    ds['EA1'].attrs['description'] = "The 'echo amplitude' of beam one, an indicator of beam reflectivity off of a boundary."
    ds['EA2'].attrs['units'] = 'Cnt'
    ds['EA2'].attrs['description'] = "The 'echo amplitude' of beam two, an indicator of beam reflectivity off of a boundary."
    ds['EA3'].attrs['units'] = 'Cnt'
    ds['EA3'].attrs['description'] = "The 'echo amplitude' of beam three, an indicator of beam reflectivity off of a boundary."
    ds['EA4'].attrs['units'] = 'Cnt'
    ds['EA4'].attrs['description'] = "The 'echo amplitude' of beam four, an indicator of beam reflectivity off of a boundary."
    ds['EA5'].attrs['units'] = 'Cnt'
    ds['EA5'].attrs['description'] = "The average 'echo amplitude' across all beams."
    
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
    ds.attrs['1st Bin Range'] = BlankDist + depthOffset
    ds.attrs['Bin Size'] = BinSize
    ds.attrs['RDI binary file'] = headerlines[1]
    ds.attrs['Instrument'] = headerlines[2]
    
    print('Conversion complete!')
    
    return ds

#========================================================================================================================

def adcp_qc(ds):
    '''
    Take the dataset produce from WinADCP_to_dataset and conduct quality control tests as recommended by the 
    IOOS QARTOD Manual for Real-time Quality Control of In-Situ Current Observations
    
    Additional information regarding suggested test parameters and theory behind tests can be found in the QARTOD manual located at:
    https://ioos.noaa.gov/ioos-in-action/currents/
    
    INPUTS:
    ds: the dataset generated by the WinADCP_to_dataset function
    
    
    OUTPUTS:
    xarray dataset with "Flag" data array
    
    EXAMPLE:
    import numpy as np
    import xarray as xr
    import sentinnel_tools as st
    
    data_path = 'RDI_ADCP_data.txt'  
    ds = st.WinADCP_to_dataset(data_path)
    ds_qc = st.adcp_qc(ds)
    '''
    
    # Generate 2D array with shape (y,x):(vertical_bins, datetime) 
    flag = xr.zeros_like(ds.East) # Same shape as typical data array
    qartod_flag = xr.zeros_like(flag)
    
    testCounter = 0 #Counts number of tests to find average flag value at the end
    #======================== REQUIRED TESTS ===================================================================
    # Tests which indicate more systematic error with the ADCP and its deployment if failed  
    # highest suspect score (3) = 45
    # lowest suspect score = 33
    # Failure score (4) = 46
    # 4 is flag > 45
    # 3 is 12 < flag <=45
    # 1 is <= 12
    # BIT test
    if 'BIT' in list(ds.keys()):
        flag = flag + xr.where(ds.BIT == 0, 0, 99) # xr.where(condition, value if true, value if false)                                      
                                                  # (test, pass value, fail value)
        testCounter = testCounter + 1
    else:
        print ('Cannot conduct BIT test because variable does not exist in dataset')
        
    # Tilt test
    flag = flag + xr.where(np.abs(ds.Roll) < 5, 0, 99)
    flag = flag + xr.where(np.abs(ds.Pitch) < 5, 0, 99)
    testCounter = testCounter + 1
    
    # Current speed test
    flag = flag + xr.where(ds.Magnitude < 1, 0, 3)
    testCounter = testCounter + 1
    
    # Current direction test
    flag = flag + xr.where((ds.Direction >= 0) & (ds.Direction <= 360), 0, 99)
    testCounter = testCounter + 1
    
    # Horizontal velocity test
    # For East-West
    flag = flag + xr.where(np.abs(ds.East) < 1, 0, 3)
    testCounter = testCounter + 1
    # For North-South
    flag = flag + xr.where(np.abs(ds.North) < 1, 0, 3)
    testCounter = testCounter + 1
    
    # Echo intensity test
    # Dimensions (D) = 3
    # Number of beams (N) = 4
    # For each beam of EA, find the difference between EA(i,j) - EA(i-1,j) throughout the entire profile, where i is bin and j is beam (1-4)
    # If the difference is greater than 30 cnts, that bin for that beam is marked as a bad beam (badbeam)
    # This is repeated for each of the 4 beams
    # If N - badbeam < D, the measurement is marked as a fail
    # If N - badbeam >= D and badbeam >=1, measurement is marked as suspect
    # If badbeam = 0, measurement passes

    # Create seperate badbeams array
    badbeams = xr.zeros_like(ds.EA1)

    # Find the difference in Echo amplitude between bins for each beam
    
    # np.diff finds the differences between each row or column depending on the axis selected
    # As a result, the first element is erased
    # EX: array_a has dimensions (5,5)
    # np.diff(array_a, axis = 0) creates an array of differences with dimensions (4,5), where row 0 is now the difference between row 1 and the original row 0 
    # The original row 0 is eliminated because the function cannot find the difference between itself and the nonexistent row -1
    # To fix this, np.diff offers the 'prepend' tool
    dEA1 = np.diff(ds.EA1, axis = 0, prepend = 0) # prepend replaces the original row 0 and subtracts it by the input value (0 in this case)
    # The result is an array of differences with the same dimensions as the original array_a (5,5)
    # However, this prepended row is actually just the original row 0, not a difference between rows
    # Since this is the first measurement of the dataset, the difference for this row would be 0
    
    dEA1[0] = dEA1[0] - dEA1[0] # To achieve this, the prepended row is subtracted by itself, creating a row of 0's
    # We now have an accurate and properly sized array of differences, where the starting values are 0
    
    # With proper differences, the where function can be used to easily flag the proper datapoints
    badbeams = badbeams + xr.where(np.abs(dEA1) <= 30, 0, 1) # Mark the bin as a +1 in badbeams if EA exceeds 30 counts

    dEA2 = np.diff(ds.EA2, axis = 0, prepend = 0)
    dEA2[0] = dEA2[0] - dEA2[0]
    badbeams = badbeams + xr.where(np.abs(dEA2) <= 30, 0, 1)

    dEA3 = np.diff(ds.EA3, axis = 0, prepend = 0)
    dEA3[0] = dEA3[0] - dEA3[0]
    badbeams = badbeams + xr.where(np.abs(dEA3) <= 30, 0, 1)

    dEA4 = np.diff(ds.EA4, axis = 0, prepend = 0)
    dEA4[0] = dEA4[0] - dEA4[0]
    badbeams = badbeams + xr.where(np.abs(dEA4) <= 30, 0, 1)

    # Using the badbeams array, we can now isolate the exact positions where 1 or more beams are invalid
    flag = flag + xr.where(badbeams == 1, 3, 0) # 1 bad beam is acceptable, but not ideal
    flag = flag + xr.where(badbeams > 1, 99, 0) # 2 or more bad beams invalidates the measurement
    testCounter = testCounter + 1
    
    # Sea surface test
    #All depth bins greater than the recorded depth and anomalously high echo amplitude are flagged as out of water
    #Threshold of 170 counts is subjective based on observations from the data
    flag = flag + xr.where((((ds.East.BinDist * ds.attrs["Bin Size"]) + ds.attrs["1st Bin Range"]) < ds.Depth) & (ds.EA5.values < 170), 0, 99)
    testCounter = testCounter + 1
    
    #======================== RECOMMENDED TESTS ===================================================================
    # Tests which are still useful for assessing quality of data, but indicate more subjective issues caused by the environment or natural limitations of the ADCP
    # Thresholds for these tests are more subjective than the required tests
    
    # Battery voltage test
    flag = flag + xr.where(ds.Battery > 145, 0, 3)
    testCounter = testCounter + 1
    
    # Beam correlation test
    flag = flag + xr.where((ds.Correlation <= 140) & (ds.Correlation > 65), 3, 0)
    flag = flag + xr.where(ds.Correlation <= 65, 99, 0)
    testCounter = testCounter + 1
    
    # Percent good test
    pg = ds.PG1 + ds.PG4

    flag = flag + xr.where((pg > 25) & (pg <= 75), 3, 0)
    flag = flag + xr.where(pg <= 25, 99, 0)
    testCounter = testCounter + 1
    
    # Vertical velocity test
    flag = flag + xr.where(np.abs(ds.Vertical) <= .15, 0, 3)
    testCounter = testCounter + 1
    
    # Error velocity test
    flag = flag + xr.where(np.abs(ds.Error_velocity) > .25, 99, 0)
    flag = flag + xr.where((np.abs(ds.Error_velocity) <= .25) & (np.abs(ds.Error_velocity) > .15), 3, 0)
    testCounter = testCounter + 1
    
    # u, v, w rate of change test
    # This test uses the same np.diff with prepend method as the echo-intensity test, only across columns instead of  rows
    
    # For East-west (u)
    # Using axis = 1, np.diff eliminates the first column, or the first timestamp
    du = np.diff(ds.East, prepend = 0).T # The prepend tool replaces the original column, but numpy doesn't have an efficient way to apply a function down a column, only across rows
    # '.T' transposes the array, so now the first column becomes the first row, which can be modified the same as before
    du[0] = du[0] - du[0] # Use the same method as the Echo Intensity test to make the first row all 0's
    du = du.T # Transpose the array back to its original format, now the array has a beginning column of 0's
    
    flag = flag + xr.where(np.abs(du) >= 1, 99, 0) # Changes in velocity greater than 1 m/s are unlikely to occur naturally in the ocean, so these should be flagged as failure
    flag = flag + xr.where((np.abs(du) < 1) & (np.abs(du) >= .25), 3, 0) # Within a range of 1 and .25 m/s is suspect for our site and deserves to be marked, but not failed
    testCounter = testCounter + 1
    
    # For North-South (v)
    dv = np.diff(ds.North, prepend = 0).T
    dv[0] = dv[0] - dv[0]
    dv = dv.T
    flag = flag + xr.where(np.abs(dv) >= 1, 99, 0)
    flag = flag + xr.where((np.abs(dv) < 1) & (np.abs(dv) >= .25), 3, 0)
    testCounter = testCounter + 1
    
    # For vertical (w)
    dw = np.diff(ds.Vertical, prepend = 0).T
    dw[0] = dw[0] - dw[0]
    dw = dw.T
    flag = flag + xr.where(np.abs(dw) > .15, 3, 0) # Magnitudes of vertical velocity are almost always smaller than horizontal velocity, so the thresholds are reduced
    testCounter = testCounter + 1
    
    # Echo intensity drop off test
    # Similar to echo intensity test, only it's a flat threshold for each individual bin, not the difference between bins
    # Ff EA(i,j) < 20, badbeam + 1
    # If N - badbeam < D, the measurement is marked as a fail
    # If N - badbeam >= D and badbeam >=1, measurement is marked as suspect
    # If badbeam = 0, measurement passes
    badbeams = xr.zeros_like(ds.EA1)

    badbeams = badbeams + xr.where(ds.EA1 < 20, 1, 0) # Mark the bin as a +1 in badbeams if EA goes below 20 counts
    badbeams = badbeams + xr.where(ds.EA2 < 20, 1, 0)
    badbeams = badbeams + xr.where(ds.EA3 < 20, 1, 0)
    badbeams = badbeams + xr.where(ds.EA4 < 20, 1, 0)
    flag = flag + xr.where(badbeams == 1, 3, 0)
    flag = flag + xr.where(badbeams > 1, 99, 0)
    testCounter = testCounter + 1
    
    # Current gradient tests
    # For current speed
    dCSPD = np.diff(ds.Magnitude, axis = 0, prepend = 0)
    dCSPD[0] = dCSPD[0] - dCSPD[0]
    flag = flag + xr.where(np.abs(dCSPD) < .3, 0, 3)
    testCounter = testCounter + 1
    
    # For change in current direction
    dCDIR = np.diff(ds.Direction, axis = 0, prepend = 0)
    dCDIR[0] = dCDIR[0] - dCDIR[0]
    flag = flag + xr.where(np.abs(dCDIR) < 45, 0, 3)
    testCounter = testCounter + 1
    
    # Add the new flag data array to the existing dataset
    flag = flag/testCounter
    qartod_flag = qartod_flag + xr.where(flag >= 4, 4, 0)
    qartod_flag = qartod_flag + xr.where((flag < 4) & (flag > 1), 3, 0)
    qartod_flag = qartod_flag + xr.where(flag <= 1, 1, 0)
    ds['Flag'] = (["BinDist", "time"], qartod_flag.values)
    ds['Flag'].attrs['Flag score'] = '[1, 2, 3, 4]'
    ds['Flag'].attrs['Grade definition'] = '1 = Pass, 2 = Not evaluated, 3 = Suspect, 4 = Fail'
    ds['Flag'].attrs['Description'] = 'Flag grading system is based on QARTOD quality control parameters and tests for ADCPs'
    
    return ds

#========================================================================================================================

def WinADCP_to_dataframe(data_path):
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
    
    return df, nbins

#========================================================================================================================

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

#========================================================================================================================

def badbeam_test(x):
    return x > 30

#========================================================================================================================

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

