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
    Depth = df['Dep'].to_numpy() + depthOffset
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
            East = (["bindist", "time"], eas),
            North = (["bindist", "time"], nor),
            Vertical = (["bindist", "time"], ver),
            Magnitude = (["bindist", "time"], mag),
            Direction = (["bindist", "time"], direction),
            Error_velocity = (["bindist", "time"], err),
            Correlation = (["bindist", "time"], cor),
            PG1 = (["bindist", "time"], pg1),
            PG4 = (["bindist", "time"], pg4),
            EA1 = (["bindist", "time"], ea1),
            EA2 = (["bindist", "time"], ea2),
            EA3 = (["bindist", "time"], ea3),
            EA4 = (["bindist", "time"], ea4),
            EA5 = (["bindist", "time"], ea5)
            
        ),
        coords=dict(
            bindist=(["bindist"], rows),
            time=(["time"], cols),
        ),
    )
    
    if 'BIT' in df.columns:
        BIT = df['BIT'].to_numpy()
        ds['BIT'] = (["time"], BIT)
        ds['BIT'].attrs['description'] = "The ADCP's own built in test metric. 0 indicate a pass, 1 indicates an error"
    
    # Add units and descriptions for variables
    ds['Pitch'].attrs['units'] = 'deg'
    ds['Pitch'].attrs['description'] = 'Pitch of the instrument'
    
    ds['Roll'].attrs['units'] = 'deg'
    ds['Roll'].attrs['description'] = 'Roll of the instrument'
    
    ds['Heading'].attrs['units'] = 'deg'
    ds['Heading'].attrs['description'] = 'Heading of the instrument'
    
    ds['Temperature'].attrs['units'] = 'Celcius'
    ds['Temperature'].attrs['description'] = 'Temperature recorded by the instrument'
    
    ds['Depth'].attrs['units'] = 'm'
    ds['Depth'].attrs['description'] = 'The estimated depth of the instrument based on the recorded pressure, temperature, and given salinity'
    
    ds['Battery'].attrs['units'] = 'cnts'
    ds['Battery'].attrs['description'] = 'Battery power of the instrument'
    
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
    ds["bindist"].attrs['units'] = 'm'
    ds["bindist"].attrs['description'] = 'Height above the seafloor'
    
    # Add metadata
    ds.attrs['PingsPerEns'] = PingsPerEns
    ds.attrs['TimePerPing'] = TimePerPing    
    ds.attrs['First Ensemble Date'] = FirstEnsDate 
    ds.attrs['First Ensemble Time'] = FirstEnsTime
    ds.attrs['Ensemble Interval'] = EnsInterval
    ds.attrs['Instrument Height'] = depthOffset
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
    
    flag = flag + xr.where(ds.EA5.values < 170, 0, 4)
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
    
    ds['Flag'] = (["bindist", "time"], qartod_flag.values)
    ds['Flag'].attrs['Flag score'] = '[1, 2, 3, 4]'
    ds['Flag'].attrs['Grade definition'] = '1 = Pass, 2 = Not evaluated, 3 = Suspect, 4 = Fail'
    ds['Flag'].attrs['Description'] = 'Flag grading system is based on QARTOD quality control parameters and tests for ADCPs'
    
    # Sea surface test
    #All depth bins greater than the recorded depth and anomalously high echo amplitude are flagged as out of water
    ds['Flag'] = ds.Flag.where(ds.Flag.bindist < ds.Depth-1, 4)
    
    return ds
#===================================================================================================================================
def adcp_seaSurface_removal(adcpDS):
    
    ADCPds = adcpDS.copy(deep=True)

    #NaN out all individual vector vales that flagged as 'fails'
    ADCPds['East'] = ADCPds.East.where(ADCPds.Flag < 4)
    ADCPds['North'] = ADCPds.North.where(ADCPds.Flag < 4)
    ADCPds['Vertical'] = ADCPds.Vertical.where(ADCPds.Flag < 4)
    ADCPds['Magnitude'] = ADCPds.Vertical.where(ADCPds.Flag < 4)
    ADCPds['Direction'] = ADCPds.Vertical.where(ADCPds.Flag < 4)
    
    return ADCPds


