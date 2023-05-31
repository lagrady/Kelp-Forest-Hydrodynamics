"""
Functions for IG dissipation methods paper
@author: Duncan Wheeler
"""

#region setup
#from configparser import MAX_INTERPOLATION_DEPTH
import numpy as np
import matplotlib.pyplot as plt
import palettable
from scipy import signal
from scipy import stats
import xarray as xr
import pandas as pd
import gsw
from pyproj import Proj
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import digamma
from iapws import iapws95
import pdb


color_cycle = palettable.cmocean.sequential.Oxy_3.mpl_colors
color_cycle = [color_cycle[1],color_cycle[0],color_cycle[2]]

#endregion


#region wave statistic functions

#region bandVrms
def bandVrms(data,hz,high,low):
    #calculate root mean squared velocity of specified frequency band

    #data is the velocity data
    #hz is the frequency of data
    #high and low are the high and low frequency 
    #cutoffs used to define the band (defined in cycles per second)

    #calculate spectrum with no averaging
    (f,s) = specSegnum(data,1,hz)

    #determine indices of appropriate frequency band
    ind = np.logical_and(f<=high,f>=low)

    #calculate root mean squared velocity
    vRms = np.sqrt(np.sum(s[ind])*np.diff(f)[0])

    return vRms
#endregion

#endregion


#region general functions

#region nan interpolation
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

#region runMean
def runMean(data, length, dt=1, keepnans=True):
    # perform a running mean on the vector data
    # length is the number of points in running mean or the time length of the running
    # mean if dt (frequency of the data) is specified.
    # uses masked array methods to handle nans
    # running mean is done with cumsum - found it in an online forum - lost link :(
    # but the link didn't do a centered mean, so I had to modify it to do a centered mean
    # also, this is going to be biased at the ends, because I am averaging over fewer points

    # This is if you want to specify a certain time, like a 30 minute mean on 8 hz data

    # check if array is masked and make it masked if not
    if not np.ma.is_masked(data):
        data = np.ma.masked_invalid(data)

    # determine how many points over which to average
    length = length/dt
    if np.remainder(length,1)!=0:
        print('the time given is not an integer number of data points. \
                casting to int')
    length = int(length)

    # It makes more sense to do a running mean over a centered point (odd length)
    if np.remainder(length,2)==0:
        print('the time given resulted in an even number of data points. \
                the resulting mean will be over 1 extra point')
        length = length+1

    # determine where the nans are
    mask = np.ma.getmaskarray(data)
    # handle nans by making them all 0s in the sum
    temp = np.cumsum(np.ma.filled(data,0))

    # find the sums of all the middle points
    ret = np.ones(np.shape(data))
    ret[int(np.ceil(length/2)):int(-np.floor(length/2))] = temp[length:] - temp[:-length]

    # find the sums of the end points
    ret[0:int(np.ceil(length/2))] = temp[int(np.floor(length/2)):length]
    ret[int(-np.floor(length/2)):] = temp[-1] - temp[-(length):-int(np.ceil(length/2))]

    # Now determine how many entries were actually added so the 0's don't bias the mean
    temp = np.cumsum(~mask)
    
    counts = np.ones(np.shape(data))
    counts[int(np.ceil(length/2)):int(-np.floor(length/2))] = temp[length:] - temp[:-length]

    # find the sums of the end points
    counts[0:int(np.ceil(length/2))] = temp[int(np.floor(length/2)):length]
    counts[int(-np.floor(length/2)):] = temp[-1] - temp[-(length):-int(np.ceil(length/2))]

    if keepnans:
        # Perform the averaging and return the nans to nans
        ret[~mask] = ret[~mask] / counts[~mask]
        ret[mask] = np.nan
    else:
        # Perform averaging and nan areas with too few datapoints
        ret = ret/counts
        ret[counts<(60/dt)] = np.nan

    return ret
#endregion

#region runSum
def runSum(data, length, dt=1, keepnans=True):
    # perform a running Sum on the vector data
    # length is the number of points in running mean or the time length of the running
    # mean if dt (frequency of the data) is specified.
    # uses masked array methods to handle nans
    # works the same as the runMean function but without the division

    # This is if you want to specify a certain time, like a 30 minute mean on 8 hz data

    # check if array is masked and make it masked if not
    if not np.ma.is_masked(data):
        data = np.ma.masked_invalid(data)

    # determine how many points over which to sum
    length = length/dt
    if np.remainder(length,1)!=0:
        print('the time given is not an integer number of data points. \
                casting to int')
    length = int(length)

    # It makes more sense to do a running sum over a centered point (odd length)
    if np.remainder(length,2)==0:
        print('the time given resulted in an even number of data points. \
                the resulting sum will be over 1 extra point')
        length = length+1

    # determine where the nans are
    mask = np.ma.getmaskarray(data)
    # handle nans by making them all 0s in the sum
    temp = np.cumsum(np.ma.filled(data,0))

    # find the sums of all the middle points
    ret = np.ones(np.shape(data))
    ret[int(np.ceil(length/2)):int(-np.floor(length/2))] = temp[length:] - temp[:-length]

    # find the sums of the end points
    ret[0:int(np.ceil(length/2))] = temp[int(np.floor(length/2)):length]
    ret[int(-np.floor(length/2)):] = temp[-1] - temp[-(length):-int(np.ceil(length/2))]

    if keepnans:
        # return the nans to nans
        ret[mask] = np.nan

    return ret
#endregion

#region runVariance
def runVariance(data,length,dt=1):
    # perform a running variance on the vector data
    # length is the number of points in running window or the time length of the running
    # window if dt (frequency of the data) is specified.
    # uses pandas rolling functions

    # put into a pandas series
    s = pd.Series(data)

    # determine how many points over which to take variance
    length = length/dt
    if np.remainder(length,1)!=0:
        print('the time given is not an integer number of data points. \
                casting to int')
    length = int(length)

    # It makes more sense to do a running variance over a centered point (odd length)
    if np.remainder(length,2)==0:
        print('the time given resulted in an even number of data points. \
                the resulting mean will be over 1 extra point')
        length = length+1

    var = s.rolling(length).var(skipna=True)

    return var

#endregion

#region rotate
def rotate(theta,x,y,transpose = False):
    #theta is a scalar or a 1d array
    #x and y are 1d arrays or 2d arrays with
    #each column representing unique dataset
    #So, size of theta = number of columns in x and y
    #transpose = True if rows are unique datasets

    if transpose:
        z = (np.exp(-1j*theta)*(x.T+1j*y.T)).T
    else:
        z = np.exp(-1j*theta)*(x+1j*y)

    return(np.real(z),np.imag(z))

#endregion

#region PCrotate
def PCrotate(z):
    # take complex velocity data z (real part is x and imaginary part is y) and
    # perform principal component analysis to rotate the data so that new data
    # is uncorelated

    #separate real and imaginary component
    x = np.real(z)
    y = np.imag(z)

    #calculate correlation
    corr = np.nanmean(np.multiply(x,y))

    #calculate rotation angle
    theta = 0.5*np.arctan(np.divide(2*corr,np.nanmean(np.square(x)) \
                                    -np.nanmean(np.square(y))))

    #rotate z
    zrot = np.multiply(z,np.exp(-1j*theta))

    #pull primary and secondary vectors from rotated z
    primary = np.real(zrot)
    secondary = np.imag(zrot)

    return(theta,primary,secondary)
#endregion

#region PCrotateMultiple
def PCrotateMultiple(x,y,transpose=False):
    # take complex velocity data z (real part is x and imaginary part is y) and
    # perform principal component analysis to rotate the data so that new data
    # is uncorelated. In this case, assumes that z is two dimensional, and that each
    # column is a unique dataset to be rotated independently (averages along first dimension (index=0))
    # transpose = True if rows are unique datasets

    if transpose:
        x = x.T
        y = y.T

    #calculate means
    xmean = np.nanmean(x,0)
    ymean = np.nanmean(y,0)

    #remove means, because it is the variations we want to use, not the means
    xdat = x-xmean
    ydat = y-ymean

    #calculate correlation
    corr = np.nanmean(np.multiply(xdat,ydat),0)

    #calculate rotation angle
    theta = 0.5*np.arctan(np.divide(2*corr,np.nanmean(np.square(xdat),0) \
                                    -np.nanmean(np.square(ydat),0)))

    #rotate original (with mean) data
    (primary,secondary) = rotate(theta,x,y)

    if transpose:
        return theta,primary.T,secondary.T
    else:
        return theta,primary,secondary

#endregion

#region meanRotateMultiple
def meanRotateMultiple(x,y,transpose=False):
    # take complex velocity x and y components and
    # rotate the data to the direction of the mean velocity
    # In this case, assumes that x and y are two dimensional, and that each
    # column is a unique dataset to be rotated independently (averages along first dimension (index=0))
    # transpose = True if rows are unique datasets
    if transpose:
        x = x.T
        y = y.T
    
    #calculate mean
    xmean = np.nanmean(x,0)
    ymean = np.nanmean(y,0)

    #calculate angle
    theta = np.arctan(np.divide(ymean,xmean))

    #rotate
    (primary,secondary) = rotate(theta,x,y)

    if transpose:
        return theta, primary.T,secondary.T
    else:
        return theta,primary,secondary

#endregion

#region Find Nearest Index
def iNear(array,value):
    return np.argmin(np.abs(array-value))
#endregion

#endregion

#region adv processing, cleaning, and despiking functions

#region cleanVec
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
    if v.attrs['coords'] == 'XYZ':
        v.U[index] = np.nan
        v.V[index] = np.nan
        v.W[index] = np.nan 

    if v.attrs['coords'] == 'ENU':
        v.North[index] = np.nan
        v.East[index] = np.nan
        v.Up[index] = np.nan

    #store what cutoffs were used
    v.attrs['CorrCutoff'] = corrCutoff
    v.attrs['SnrCutoff'] = snrCutoff
    v.attrs['AngleCutoff'] = angleCutoff
    
    return v
#endregion

#region avTime30(vector)
def avTime30(vector):
    #create a 30 minute burst timestamp and burst numbers

    #don't modify in place
    vnew = vector.copy(deep=True)

    #pull current timestamp
    time = vnew.time.values

    #set a 15 minute stime step
    dt = np.timedelta64(15,'m')

    #create new 30 minute time stamp starting 15 minutes after the first time value
    tav = np.arange(time[0].astype('datetime64[m]') + dt, time[-1].astype('datetime64[m]'),2*dt)

    #initialize a burst number array
    bNum30 = np.empty(time.size)*np.nan

    #create a variable with the burst number for every data point (on the original time stamp)
    for i in np.arange(tav.size):
        bNum30[np.logical_and(time>=tav[i]-dt,time<tav[i]+dt)] = i
    
    #save new time stamp and burst number variables
    vnew.coords['avTime30'] = tav
    vnew['bNum30'] = ('time',bNum30)
    vnew['burst30'] = ('avTime30', np.arange(tav.size))

    return vnew
#endregion    

#region rotateVec
def rotateVec(vector):

    #rotate vector data to the principal axis
    #vector is an xarray dataset

    #don't perform modifications in place
    v = vector.copy(deep=True)

    #initiealize ENU variable for deciding what variables to rotate (assume we have ENU)
    ENU = 1

    #if in XYZ coordinates, try to convert to ENU coordinates before rotating
    #remember to propagate non-original data through
    if v.attrs['coords'] == 'XYZ':
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
        [theta, primary, secondary] = PCrotate(v.East.values + 1j * v.North.values)
        orig = np.logical_and(v.EOrig.values,v.NOrig.values)
    else:
        # Using principle component analysis to rotate data to uncorrelated axes
        [theta, primary, secondary] = PCrotate(v.U.values + 1j * v.V.values)
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
#endregion

#region save Despike Figures
def saveDespikeFig(new,old,file):
    #create and save figure on how despiking went
    #new is despiked data
    #old is non-despiked data
    #file is the file path where the figure gets saved

    #initialize arrays for first and second derivatives
    dnew = np.empty(new.size)*np.nan
    dnew2 = np.empty(new.size)*np.nan
    dold = np.empty(old.size)*np.nan
    dold2 = np.empty(old.size)*np.nan

    #calculate first and second derivatives of despiked data
    dnew[1:-1] = (new[2:]-new[0:-2])/2
    dnew2[2:-2] = (dnew[3:-1]-dnew[1:-3])/2

    #calculate first and second derivatives of original data
    dold[1:-1] = (old[2:]-old[0:-2])/2
    dold2[2:-2] = (dold[3:-1]-dold[1:-3])/2

    #create figure
    fig = plt.figure(figsize=(15,7))

    #plot u-du phase space scatter plot
    plt.subplot(2,3,1)
    plt.plot(old,dold,'r*')
    plt.plot(new,dnew,'b*')
    plt.axis('equal')
    plt.xlabel('u')
    plt.ylabel('du')

    #plot du-du2 phase space scatter plot
    plt.subplot(2,3,2)
    plt.plot(dold,dold2,'r*')
    plt.plot(dnew,dnew2,'b*')
    plt.axis('equal')
    plt.xlabel('du')
    plt.ylabel('du2')

    #plot u-du2 phase space scatter plot
    plt.subplot(2,3,3)
    plt.plot(old,dold2,'r*')
    plt.plot(new,dnew2,'b*')
    plt.axis('equal')
    plt.xlabel('u')
    plt.ylabel('du2')

    #plot time series of original and despiked data
    plt.subplot(2,1,2)
    plt.plot(old,color='red')
    plt.plot(new,'.-',color='blue')
    plt.xlabel('sample number')
    plt.ylabel('vel (m/s)')
    plt.legend(('original data','despiked data'),loc='upper right')

    #title figure
    fig.suptitle(file.split('/')[-1])

    #save figure
    fig.savefig(file+'.png')
    
    #close figure
    plt.close(fig)

    return

#endregion

#region plotEllipse(xrad,yrad,theta,ax,color='red',label='')
def plotEllipse(xrad,yrad,theta,ax,color=color_cycle[1],label='',ls='-',lw=3):
    #plot pahse space ellipse with x radius = xrad and y radius = yrad
    #theta is the angle the ellipse makes with the x axis
    #plotted on axis ax, with color=color, label=label, line style = ls, and line width = lw

    #create x points for ellipse
    x = np.linspace(-xrad,xrad,100)

    #create positive y values corresponding to x points
    y = np.sqrt((1-np.square(x)/np.square(xrad))*np.square(yrad))

    #create rotated points for top of ellipse
    Xrot = (x+1j*y)*np.exp(1j*theta)

    #create rotated points for bottom of ellipse
    Xrot2 = (x-1j*y)*np.exp(1j*theta)

    #decompose imaginary variables
    x = np.real(Xrot)
    y = np.imag(Xrot)
    x2 = np.real(Xrot2)
    y2 = np.imag(Xrot2)

    #plot
    line = ax.plot(x,y,color=color,label=label,lw=lw,ls=ls)
    ax.plot(x2,y2,color=color,lw=lw,ls=ls)
    
    return line
#endregion

#region plotDespikePlane()
def plotDespikePlane(a,b,alim,blim,theta,label1,label2,ax,c1 = color_cycle[0],c2 = color_cycle[1],histogram=False,size=3):
    #plot phase space plane of a vs. b with ellipse limits alim and blim
    #at a rottation angle of theta on axis ax, with labels label1 and label2

    #plot data
    if histogram:
        inan = np.logical_and(~np.isnan(a),~np.isnan(b))
        ax.hist2d(a[inan],b[inan],label='Original Data')
    else:
        ax.plot(a,b,'.',color=c1,label='Original Data',markersize=size)

    #plot ellipse
    line = plotEllipse(alim,blim,theta,ax,color=c2)

    #fix aspect ratio and label axis
    ax.axis('equal')
    ax.set_xlabel(label1,fontsize=17)
    ax.set_ylabel(label2,fontsize=17)
    return line

#endregion

#region findLimits(despike helper function)
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
#endregion

#region despike(velocities, expand=False)
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

#endregion

#region despike_iterate
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
#endregion

#region despike_all
def despike_all(vector,expand=False,lp=1/20,savefig=False,savePath=None,expSize = 0.01,expEnd = 0.95):
    #Despike all velocity components for all bursts in vector.
    #vector is an xarray dataset with the data collected from an adv
    #expand = true uses the expanding cutoff, while false uses gaussian cutoffs
    #lp gives the frequency at which the data is low pass filtered before despiking
    #savefig = true tells the function to save all despiking figures
    #savePath is the path to the folder in which the figures are saved

    #copy vector to not modify it in place
    v = vector.copy(deep=True)

    #pull sampling frequency of data
    hz = vector.attrs['Sampling Rate (Hz)']

    #pull serial number of instrument for labelling and saving figures
    serial = vector.attrs['Hardware Serial Number'].split(' ')[-1]

    #save total number of bursts for outputing progress status
    maxOut = str(v.bNum30.max().values)

    #check what coordinate system we are working with
    if v.attrs['coords'] == 'XYZ':

        #create new variables for storing what data is original
        v['UOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 
        v['VOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 
        v['WOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 

        #loop through all bursts
        for i in v.burst30.values.astype('int'):

            #output progress status
            print(str(float(i)) + ' of ' + maxOut)

            #get indices of current burst data
            index = np.where(v.bNum30==i)[0]

            #despike each velocity component
            Urep, detectedU = despike_iterate(v.U.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)
            Vrep, detectedV = despike_iterate(v.V.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)
            Wrep, detectedW = despike_iterate(v.W.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)

            #create and save figures
            if savefig:
                print('saving burst ' + str(i))
                saveDespikeFig(Urep,v.U[index].values,savePath+'/v'+serial+'_U_Burst_{:04d}'.format(i))
                saveDespikeFig(Vrep,v.V[index].values,savePath+'/v'+serial+'_V_Burst_{:04d}'.format(i))
                saveDespikeFig(Wrep,v.W[index].values,savePath+'/v'+serial+'_W_Burst_{:04d}'.format(i))

            #store info on what points are original or not
            v.UOrig[index] = ~detectedU
            v.VOrig[index] = ~detectedV
            v.WOrig[index] = ~detectedW

            #update with despiked data
            v.U[index] = Urep
            v.V[index] = Vrep
            v.W[index] = Wrep

    elif v.attrs['coords'] == 'ENU':

        #create new variables for storing what data is original
        v['NOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 
        v['EOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 
        v['UpOrig'] = ('time',np.ones(v.time.shape,dtype='bool')) 

        #loop through all bursts
        for i in v.burst30.values.astype('int'):
            
            #output progress status
            print(str(float(i)) + ' of ' + maxOut)

            #get indices of current burst data
            index = np.where(v.bNum30==i)[0]

            #despike each velocity component
            Nrep, detectedN = despike_iterate(v.North.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)
            Erep, detectedE = despike_iterate(v.East.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd) 
            Uprep, detectedUp = despike_iterate(v.Up.values[index], hz=hz, lp=lp, expand=expand,expSize=expSize,expEnd=expEnd)

            #create and save figures
            if savefig:
                print('saving burst ' + str(i))
                saveDespikeFig(Nrep,v.North[index].values,savePath+'/v'+serial+'_North_Burst_{:04d}'.format(i))
                saveDespikeFig(Erep,v.East[index].values,savePath+'/v'+serial+'_East_Burst_{:04d}'.format(i))
                saveDespikeFig(Uprep,v.Up[index].values,savePath+'/v'+serial+'_Up_Burst_{:04d}'.format(i))

            #store info on what points are original or not
            v.NOrig[index] = ~detectedN
            v.EOrig[index] = ~detectedE
            v.UpOrig[index] = ~detectedUp

            #update with despiked data
            v.North[index] = Nrep
            v.East[index] = Erep
            v.Up[index] = Uprep

    return v
#endregion

#region ProcessVec(data,met,Pdate1,Padate2,badSections,reverse)

def ProcessVec(data,BPressure,Pdate1,Pdate2,badSections,reverse,expand = True,lp = 1/20,expSize = 0.01,expEnd = 0.95):
    #initial processing of adv data. adjusts pressure,
    #calculates depth, eliminates bad data due to
    #bad time step, out of water transitions, snr,
    #correlation, and despiking
    #
    #data is an xarray dataset
    #BPressure is an xarray dataarray with atmospheric pressure
    #Pdate1 and Pdate2 are the beginning and ending times for calculating pressure offset
    #badSections are data that should be deleted for other reasons
    #if reverse is true, reverses the primary and secondary velocity components (to make sure flood is positive)
    #expand, lp, expSize, and expEnd are variables passed to despike_all
    #if expand = True, an expanding cutoff is used
    #lp is the frequency cutoff for the lowpass filter used to remove the low pass signal
    #expSize is the step size for expanding the cutoff in the expanding cutoff algorithm
    #expEnd is the density change cutoff for determining when to stop expanding the phase space cutoffs

    hz = data.attrs['Sampling Rate (Hz)']

    print('keeping only data before timestamp goes crazy')

    #pull time data
    t = data.time.values

    #select all data up until first time time difference is greater than 6 seconds
    tnew = t[:np.where(np.diff(t)>np.timedelta64(6,'s'))[0][0]]

    #use new time stamp to select only desired portions of xarray dataset
    data_temp = data.sel(time=tnew,time_start=slice(None,tnew[-1]),time_sen=slice(None,tnew[-1]))


    print('Adjust Pressure, calc lat/lon, and Calculate Depth')

    if Pdate1 is not None:
        #pull pressure data specifically from an out of water period to correct for atmospheric pressure
        Pcal = data.Pressure.sel(time=slice(Pdate1,Pdate2))

        #use that time period to calculate a pressure offset based on difference between met pressure and instrument
        offset = BPressure*0.01 - np.mean((BPressure.interp(time=Pcal.time)*.01)-Pcal)  

        #remove offset from data to get correct pressure information
        data_temp.Pressure[:] = data_temp.Pressure.values - offset.interp(time=data_temp.time).values

        #use utm projection to convert gps RTK coordinates into lat/lon location
        #myProj = Proj(proj="utm", zone='11N', ellps="GRS80", datum='NAD83', south= False)  
        #(lon, lat) = myProj(data_temp.attrs['GPS RTK W'],data_temp.attrs['GPS RTK N'], inverse=True)
    (lon, lat) = (-117.3,32.95)
    #save lat lon info
    data_temp.attrs['lon'] = lon
    data_temp.attrs['lat'] = lat

    #use gibbs sea water package to convert from pressure to depth
    data_temp['Depth'] = ('time', np.abs(gsw.conversions.z_from_p(data_temp.Pressure.values,data_temp.lat)))


    print('cleaning vector based on correlation and snr cutoffs')

    #use correlation and snr cutoffs to clean data
    data_temp = cleanVec(data_temp,corrCutoff=70,snrCutoff=10)


    print('Assigning Burst Numbers')

    #take already present burst numbers and put them on high frequency time stamp
    BurstNum = np.hstack([np.ones(data_temp.time.sel(time = slice(data_temp.time_start[i-1],\
        data_temp.time_start[i])).size-1)*i for i in data_temp.BurstCounter[:-1].values.astype('int')]+\
            [np.ones(data_temp.time.sel(time=slice(data_temp.time_start[-1],None)).size)*data_temp.BurstCounter[-1].values.astype('int')])

    data_temp['BurstNum'] = ('time',BurstNum)

    #Also create 30 minute bursts and corresponding burst numbers and time stamp
    data_temp = avTime30(data_temp)


    print('naning out of water transitions') 

    #calculate 20 minute running fraction of points that are not nanned
    Ueval = runMean(np.array(~np.isnan(data_temp.U.values),dtype='int'),20*60*hz)

    #find where that fraction is less than 10%
    i = np.where(Ueval<.1)[0]

    #consider these as sections that mostly out of water
    #(and therefore transition zones), and nan them
    data_temp.U[i] = np.nan
    data_temp.V[i] = np.nan
    data_temp.W[i] = np.nan

    #create a 15 minute long index
    ind = np.arange(15*60*hz)

    #for each of these stretches of out of water sections, identify the last point
    #and further nan the next 15 minutes of data, because we found these to be bad data
    for i in np.where(Ueval<.1)[0][np.where(np.diff(np.where(Ueval<.1)[0])>1)[0]]:
        data_temp.U[ind+i] = np.nan
        data_temp.V[ind+i] = np.nan
        data_temp.W[ind+i] = np.nan


    print('despiking vector')

    plt.ioff()

    data_temp_All = despike_all(data_temp,expand=expand,lp=lp,savefig=False,savePath=None,expSize=expSize,expEnd=expEnd)

    print('removing bad sections')

    for i in badSections:
        
        #pull out indices of data in bad sections
        ind = np.logical_and(data_temp_All.time>= i[0],data_temp_All.time<=i[1])

        #set data to nan
        data_temp_All.U[ind] = np.nan
        data_temp_All.V[ind] = np.nan
        data_temp_All.W[ind] = np.nan

    print('rotating vector')

    data_temp_All = rotateVec(data_temp_All)

    if reverse:
        print('reversing direction so primary is in flooding direction')
        data_temp_All.Primary[:] = -data_temp_All.Primary[:]
        data_temp_All.Secondary[:] = -data_temp_All.Secondary[:]
        data_temp_All.attrs['Theta'] = data_temp_All.Theta - np.pi

    data_temp_All.attrs['despike_lp_freq (hz)'] = lp
    data_temp_All.attrs['despike_cutoff_expansion_fraction'] = expSize
    data_temp_All.attrs['despike_cutoff_expansion_densityChange_end_condition'] = expEnd

    return data_temp_All

#endregion

"""

I think this is an obsolete function, but keeping it here in case I broke something

#region DespikeTolerance(data,met,Pdate1,Padate2,badSections,reverse)

def DespikeTolerance(data,badSections,expSize = (0.005,0.01,0.02),expEnd = (0.9,0.95,1)):
    #initial processing of adv data. adjusts pressure,
    #calculates depth, eliminates bad data due to
    #bad time step, out of water transitions, snr,
    #correlation, and despiking
    #
    #data is an xarray dataset
    #BPressure is an xarray dataarray with atmospheric pressure
    #Pdate1 and Pdate2 are the beginning and ending times for calculating pressure offset
    #badSections are data that should be deleted for other reasons
    #if reverse is true, reverses the primary and secondary velocity components (to make sure flood is positive)

    hz = data.attrs['Sampling Rate (Hz)']

    print('keeping only data before timestamp goes crazy')

    #pull time data
    t = data.time.values

    #select all data up until first time time difference is greater than 6 seconds
    tnew = t[:np.where(np.diff(t)>np.timedelta64(6,'s'))[0][0]]

    #use new time stamp to select only desired portions of xarray dataset
    data_temp = data.sel(time=tnew,time_start=slice(None,tnew[-1]),time_sen=slice(None,tnew[-1]))

    print('cleaning vector based on correlation and snr cutoffs')

    #use correlation and snr cutoffs to clean data
    data_temp = cleanVec(data_temp,corrCutoff=70,snrCutoff=10)

    print('Assigning Burst Numbers')

    #take already present burst numbers and put them on high frequency time stamp
    BurstNum = np.hstack([np.ones(data_temp.time.sel(time = slice(data_temp.time_start[i-1],\
        data_temp.time_start[i])).size-1)*i for i in data_temp.BurstCounter[:-1].values.astype('int')]+\
            [np.ones(data_temp.time.sel(time=slice(data_temp.time_start[-1],None)).size)*data_temp.BurstCounter[-1].values.astype('int')])

    data_temp['BurstNum'] = ('time',BurstNum)

    #Also create 30 minute bursts and corresponding burst numbers and time stamp
    data_temp = avTime30(data_temp)

    print('naning out of water transitions') 

    #calculate 20 minute running fraction of points that are not nanned
    Ueval = runMean(np.array(~np.isnan(data_temp.U.values),dtype='int'),20*60*hz)

    #find where that fraction is less than 10%
    i = np.where(Ueval<.1)[0]

    #consider these as sections that mostly out of water
    #(and therefore transition zones), and nan them
    data_temp.U[i] = np.nan
    data_temp.V[i] = np.nan
    data_temp.W[i] = np.nan

    #create a 15 minute long index
    ind = np.arange(15*60*hz)

    #for each of these stretches of out of water sections, identify the last point
    #and further nan the next 15 minutes of data, because we found these to be bad data
    for i in np.where(Ueval<.1)[0][np.where(np.diff(np.where(Ueval<.1)[0])>1)[0]]:
        data_temp.U[ind+i] = np.nan
        data_temp.V[ind+i] = np.nan
        data_temp.W[ind+i] = np.nan

    print('despiking vector')

    plt.ioff()

    data_temp_All = despike_all(data_temp,expand=True,lp=1/20,savefig=False,savePath=None,expSize=expSize,expEnd=expEnd)

    print('removing bad sections')

    for i in badSections:
        
        #pull out indices of data in bad sections
        ind = np.logical_and(data_temp_All.time>= i[0],data_temp_All.time<=i[1])

        #set data to nan
        data_temp_All.U[ind] = np.nan
        data_temp_All.V[ind] = np.nan
        data_temp_All.W[ind] = np.nan

    return data_temp_All

#endregion

"""

#endregion


#region spectrum calculation functions

#region normal

#region specSegnum
def specSegnum(data,segnum,hz,plot=False,errorReturn=False,filt=False,wind=True):
    seglength = np.floor(np.shape(data)[0]/segnum)
    return spectrum(data,seglength,hz,plot,errorReturn,filt,wind)
#endregion

#region specSeglength
def specSeglength(data,seglength,hz,plot=False,errorReturn=False,filt=False,wind=True):
    return spectrum(data,seglength,hz,plot,errorReturn,filt,wind)
#endregion

#region specSegtime
def specSegtime(data,time,hz,plot=False,errorReturn=False,filt=False,wind=True):
    #time should be reported in seconds, just as hz is in 1/sec
    seglength = int(time*hz)
    return spectrum(data,seglength,hz,plot,errorReturn,filt,wind)
#endregion

#region specSegmentNorm
def specSegmentNorm(data,seglength,segnum):
    #helper function for splittling data into segments before taking fourier transform

    # create variable to use in creating segments
    ltotal = np.shape(data)[0]

    # split data into appropriate segments
    if np.remainder(ltotal,seglength) == 0:
        segments = np.concatenate((np.reshape(data,(np.int((segnum+1)/2),np.int(seglength))),\
            np.reshape(data[np.int(seglength/2):np.int(np.floor(-seglength/2))],\
                (np.int((segnum-1)/2),np.int(seglength)))),0)
    else:
        segments = np.concatenate((np.reshape(data[0:-np.int(np.remainder(ltotal,seglength))],(np.int((segnum+1)/2),np.int(seglength))),\
            np.reshape(data[np.int(seglength/2):np.int(np.floor(-np.remainder(ltotal,seglength)\
                -seglength/2))],(np.int((segnum-1)/2),np.int(seglength)))),0)

    return segments
#endregion

#region segmentFilt
def segmentFilt(segments):
    #keep only segments without too many nans

    i = np.sum(np.isnan(segments),-1)<segments.shape[-1]/10

    return segments[i,:],i
#endregion

#region spectrum
def spectrum(data,seglength,hz,plot=True,errorReturn=False,filt=False,wind=True):
    # Calculate and plot a spectrum of data with all real values spaced
    # evenly in time by spacing (in units of seconds). Splits data into
    # overlapping segments of length seglength.  Input data should be a row vector.
    # Made based on Data 1 class.
    
    if filt:
        sos = signal.butter(4,hz/seglength,'hp',fs=hz,output='sos')

    divnum = np.floor(np.shape(data)[0]/seglength)
    segnum = 2*divnum-1

    # calculate degrees of freedom based on 50% overlap
    dof = (36/19)*segnum

    # want error bars with roughly 95% confidence
    alpha = 0.05

    # determine the frequency spacing of the final spectrum
    spacing = 1/hz
    dfreq = 1/(seglength*spacing) #cycles per second

    # create a hanning window for the data
    if wind:
        window = np.hanning(seglength+2)[1:-1]*np.sqrt(8/3)
    else:
        window = np.ones(seglength)

    # create a vector of the frequencies present in the spectrum
    #frequencies = np.arange(0,seglength/2+1)*dfreq
    frequencies = np.fft.fftfreq(int(seglength),spacing)
    freqInd = np.where(frequencies>0)
    frequencies = frequencies[freqInd]

    # determine the error bar size
    err_low = dof/stats.chi2.ppf(1-alpha/2,df=dof)
    err_high = dof/stats.chi2.ppf(alpha/2,df=dof)
    error = np.array([err_low,err_high])

    #split data into segments
    segments = specSegmentNorm(data,seglength,segnum)

    # take fft of all the segments
    if filt:
        f = np.fft.fft(window*signal.sosfiltfilt(sos,segments))/seglength
    else:
        f = np.fft.fft(window*signal.detrend(segments))/seglength

    # Calculate the spectra and average segments together.
    # Note that because data is real, only use first half of the data
    # and double the spectra values for correct normalization
    s = np.mean(np.square(np.abs(f[:,freqInd])),0).flatten()/dfreq
    if np.remainder(f.shape[-1],2)==0:
        s[:-1] = 2*s[:-1]
    else:
        s = 2*s

    if plot:
        plt.figure()
        plt.loglog(frequencies,s)
        plt.xlabel('Cycles Per Second')

        #adding lines to look at IG frequencies
        plt.loglog([0.03,0.03],[np.max(s),np.min(s)],'r:')
        plt.loglog([0.003,0.003],[np.max(s),np.min(s)],'r:')

        plt.legend(['spectrum','ig band'])

    if errorReturn:
        return frequencies, s, dof
    else:
        return frequencies, s

#endregion

#endregion

#region wave corrected

#region spectrum corrections

def specCorrectionValue(sigmaW,uMean):
    #equation (25) from Rosman and Gerbi 2017
    
    ratio = sigmaW/uMean

    def integrand(x, a):
        return (np.abs(1-a*x)**(2/3))*np.exp(-(1/2)*(x**2))

    return quad(integrand,-np.inf,np.inf,args=(ratio))[0]/np.sqrt(2*np.pi)

#calculate correction values for following function
testRatios = np.arange(0.001,1.201,0.001)
corrections = np.empty(testRatios.size)
for i in np.arange(testRatios.size):
    corrections[i] = specCorrectionValue(testRatios[i],1)

def correctionValues(ratios):
    #pull correction values for specified ratios (faster than redoing integration every time)

    #find nearest test ratio
    i = np.searchsorted(testRatios,np.abs(ratios))

    #because searchsorted always rounds up, find where it needs to be adjusted back down
    rounding = np.modf(np.abs(ratios)*1000)[0]
    indRounding = np.logical_and(rounding>0,rounding<.5)
    i[indRounding] = i[indRounding]-1
    i[i==testRatios.size] = i[i==testRatios.size]-1

    #return corrections
    return corrections[i]

#endregion

#region specSegment

def specSegment(data,seglength):
    #segments data for waveCorrectedSpectrum method
    #data is a touple containing the 1d numpy arrays of
    #each data that should be segmented
    #seglength is the length in number of data points of each segment

    #initialize return tuple
    segments = ()

    #segment each dataset found in data
    for d in data:

        #determine size of current data
        ltotal = d.size

        #determine number of segments
        segnum = np.floor(ltotal/seglength)

        #segment with appropriate method depending if data can be split evenly or not
        if np.remainder(ltotal,seglength) == 0:
            segments = segments + (np.reshape(d,(np.int(segnum),np.int(seglength))),)
        else:
            segments = segments + (np.reshape(d[0:-np.int(np.remainder(ltotal,seglength))],(np.int(segnum),np.int(seglength))),)
        
    return segments

#endregion

#region keepSegments
def keepSegments(pMeans,pLowStd,pHighStd,origSegments,turbVarCutoff = 5, waveCutoff = 1.025,\
    unorigionalCutoff = 0.01):

    test = np.concatenate((np.ones((origSegments.shape[0],1),dtype='bool'),origSegments,np.ones((origSegments.shape[0],1),dtype='bool')),axis=1)
    consecutiveBad = np.array([np.max(np.diff(np.where(t)))-1 for t in test])

    # count number of unoriginal data points in each segment
    badNum = np.sum(~origSegments,-1)

    # mark segments that satisfy stationarity requirements to keep
    keep = np.logical_and(pLowStd <= waveCutoff*np.abs(pMeans),\
        pHighStd <= np.abs(pMeans)/turbVarCutoff)

    # mark entire segments already deleted as bad data
    badNum[~keep] = origSegments.shape[-1]

    # of the remaining segments, eliminate from worst to best until less than 1% of remaining data is bad
    test = np.cumsum(np.sort(badNum))/((np.arange(badNum.size)+1)*origSegments.shape[-1])
    keep[np.argsort(badNum)[np.searchsorted(test,unorigionalCutoff):]] = False

    if np.sum(keep) > 0:
        maxBad = np.nanmax(badNum[keep])
        maxConsec = np.nanmax(consecutiveBad[keep])
    else:
        maxBad = np.nan
        maxConsec = np.nan

    return keep,maxBad,maxConsec

#endregion
    
#region waveCorrectedSpectrum

def waveCorrectedSpectrum(dataOrig,u,v,orig,seconds=10,hz=16,fullOut=False,useW=False, highLowSep = 1/5, turbVarCutoff = 5, waveCutoff = 1.025, unorigionalCutoff = 0.01,\
    lowWavenumberRemoval = 2, highSpectrumRemoval = 4, binSize = 50, correct = True):
    #calculates a wave corrected spectrum using a segmented approach
    #dataOrig is the data to make a spectrum of - it should already have the nan's interpolated
    #u and v are the primary and secondary advection velocities
    #orig is a boolean array that is true where the data hasn't been changed from original observations
    #seconds is the length of the segments used for calculating the spectra
    #hz is the frequency at which the data was sampled
    #returns final wavenumbers, spectrum, and degrees of freedom
    #if fullOut=True, also returns individual spectra before averaging occurs
    #if useW=True, uses the high pass vertical velocity to determine what segments to throw out
    #highLowSep is the frequency cutoff for determining the high and low frequency portions of the signal
    #turbVarCutoff is the cutoff used for determining if the turbulence is too big relative to the mean current for frozen turbulence
    #waveCutoff is the cutoff used for determining if the wave driven velocity variance is too big relative to the mean current
    #unorigionalCutoff is the cutoff used for determining when to stop eliminating segments of data because they have unorigional data points
    #lowWavenumberRemoval is the number of low wavenumber points removed from the spectrum to avoid bias
    #highSpectrumRemoval is the factor avove the nyquist frequency we require a spectrum value to be to be kept for avoiding aliasing bias
    #binSize is the minimum number of points used in each bin when averaging the spectrum
    #correct is a boolean to determine whether or not to apply the high frequency wave correction constant to each segment

    #region setup

    #high pass filter the data at the segment length to eliminate low frequency contamination
    #sos = signal.butter(4,1/seconds,'hp',fs=hz,output='sos')
    #data = signal.sosfiltfilt(sos,dataOrig)
    data = dataOrig

    #use low pass filter at 1/5 hz (or highLowSep frequency) to split advection vs. turbulent horizontal velocities
    sos = signal.butter(4,highLowSep,'lp',fs=hz,output='sos')
    uLow = signal.sosfiltfilt(sos,u)
    vLow = signal.sosfiltfilt(sos,v)
    uHigh = u-uLow
    vHigh = v-vLow

    if useW:
        uHigh = dataOrig - signal.sosfiltfilt(sos,dataOrig)

    #determine segment number and size for splitting data
    seglength = seconds*hz

    # determine the frequency spacing of the final spectrum
    spacing = 1/hz
    dfreq = 2*np.pi/(seconds) #radians per second

    # create a hanning window for the data
    window = np.hanning(seglength+2)[1:-1]*np.sqrt(8/3)

    # create a vector of the frequencies present in the spectrum
    frequencies = np.fft.fftfreq(seglength,spacing)
    freqInd = np.where(frequencies>0)
    frequencies = frequencies[freqInd]*2*np.pi

    #endregion

    #region segment the data
    
    #create a time vector for segmenting
    time = np.arange(data.size)*1/hz

    #split data into appropriate segments
    (segments,tsegs,uLowSegments,vLowSegments,uHighSegments,vHighSegments,origSegments) = \
        specSegment((data,time,uLow,vLow,uHigh,vHigh,orig),seglength)    

    #endregion

    #region rotate segments

    (theta,pLow,sLow) = meanRotateMultiple(uLowSegments,vLowSegments,transpose=True)

    if useW:
        pHigh = uHighSegments
    else:
        (pHigh,sHigh) = rotate(theta,uHighSegments,vHighSegments,transpose=True)

    #endregion

    #region deal with bad data

    # calculate mean primary velocity
    pMeans = np.mean(pLow,1)

    # get standard deviations of advection velocities for each segment
    pLowStd = np.std(pLow,1)
    pHighStd = np.std(pHigh,1)

    #mark segments to keep in spectrum calculation
    keep,maxBad,maxConsec = keepSegments(pMeans,pLowStd,pHighStd,origSegments,turbVarCutoff=turbVarCutoff,\
        waveCutoff=waveCutoff,unorigionalCutoff=unorigionalCutoff)

    # if too many bursts are eliminated, don't bother calculating a spectrum
    if np.sum(keep) < 2:
        if fullOut:
            return [np.empty(1)*np.nan]*5+[keep,maxBad,maxConsec]
        else:
            return [np.empty(1)*np.nan]*3

    #endregion

    #region calculate individual segment spectra

    # take fft of all segments that we are keeping
    f = np.fft.fft(window*signal.detrend(segments[keep,:]))/seglength

    # Calculate the spectra, convert to wavenumber,
    # correct magnitude, and average segments together.
    # Note that because data is real, only use first half of the data
    # and double the spectra values for correct normalization

    #Calculate the spectra
    s = (np.square(np.abs(f[:,freqInd]))/dfreq).squeeze()
    if np.remainder(f.shape[-1],2)==0:
        s[:-1] = 2*s[:-1]
    else:
        s = 2*s
    
    #use low pass filtered velocities to calculate average advection speeds
    avSpeeds = np.sqrt(np.mean(uLowSegments[keep,:],1)**2+np.mean(vLowSegments[keep,:],1)**2)

    #tile advection speeds to be right shape for
    #creating unique wavenumbers for each segment
    advectVels = np.transpose(np.tile(avSpeeds,(s.shape[1],1)))

    #convert frequency to wavenumber
    waveNums = frequencies/advectVels

    #get wave correction values
    I = correctionValues(pLowStd[keep]/pMeans[keep])

    #convert to wavenumber spectrum and correct magnitude
    if correct:
        sK = ((s*advectVels).T/I).T
    else:
        sK = s*advectVels

    #get rid of untrustworthy low wavenumber values and potential signal aliasing
    if highSpectrumRemoval != 0:
        frac = 1/highSpectrumRemoval
        ikeep = np.array([waveNums[i,:]<=(waveNums[i,-1]**(-5/3)/frac)**(-3/5) for i in np.arange(waveNums.shape[0])])
    else:
        ikeep = np.ones(waveNums.shape,dtype='bool')
        
    ikeep[:,0:lowWavenumberRemoval] = False

    locs = np.where(ikeep)

    #endregion

    #region calculate averaged spectra

    #sort wavenumbers
    isort = np.argsort(waveNums[locs])

    #split wavenumbers and spectral values into bins
    numbins = int(waveNums[locs].size/binSize)
    if numbins<1:
        numbins=1
    k = np.array_split(waveNums[locs][isort],numbins)
    spectra = np.array_split(sK[locs][isort],numbins)

    #initialize final degrees of freedom, spectrum, and wavenumber arrays
    dof = np.zeros(np.shape(k)[0])
    spec = np.zeros(dof.shape)
    kActual = np.zeros(dof.shape)

    #for each bin
    for i in np.arange(dof.size):

        #calculate degrees of freedom based on actual number of points
        n = k[i].size
        dof[i] = 2*n

        #calculate mean spectral value
        spec[i] = np.mean(spectra[i])

        #calculated expected wavenumber for mean spectral value
        kActual[i] = (np.sum(k[i]**(-5/3))/n)**(-3/5)

    #endregion

    if fullOut:
        return kActual,spec,dof,k,spectra,keep,maxBad,maxConsec
    else:
        return kActual,spec,dof

#endregion

#region segmentCount
def segmentCount(u,v,orig,seconds=10,hz=16,w=None):
    #function for counting which segments were kept and what the
    #segment mean velocities were during spectrum calculation
    #without actually calculating the whole spectrum

    #use low pass filter at 1/5 hz to split advection vs. turbulent horizontal velocities
    sos = signal.butter(4,1/5,'lp',fs=hz,output='sos')
    uLow = signal.sosfiltfilt(sos,u)
    vLow = signal.sosfiltfilt(sos,v)
    uHigh = u-uLow
    vHigh = v-vLow

    if not w is None:
        uHigh = w - signal.sosfiltfilt(sos,w)

    #determine segment number and size for splitting data
    seglength = seconds*hz

    #split data into appropriate segments
    (uLowSegments,vLowSegments,uHighSegments,vHighSegments,origSegments) = \
        specSegment((uLow,vLow,uHigh,vHigh,orig),seglength)  

    #rotate segments
    (theta,pLow,sLow) = meanRotateMultiple(uLowSegments,vLowSegments,transpose=True)

    if w is None:
        (pHigh,sHigh) = rotate(theta,uHighSegments,vHighSegments,transpose=True)
    else:
        pHigh = uHighSegments

    # calculate mean primary velocity
    pMeans = np.mean(pLow,1)

    # get standard deviations of advection velocities for each segment
    pLowStd = np.std(pLow,1)
    pHighStd = np.std(pHigh,1)

    #mark segments to keep in spectrum calculation
    keep = keepSegments(pMeans,pLowStd,pHighStd,origSegments)

    return keep,pMeans

#endregion

#region specBnum
def specBnum(vec, bnum, seconds=10, fullOut=False, useW=False):
    #function for calculating wave corrected spectrum of a specific burst (bnum) of vector data (vec)

    ind = vec.bNum30 == bnum

    dataOrig = nan_interp(vec.Up.values[ind])
    u = nan_interp(vec.Primary.values[ind])
    v = nan_interp(vec.Secondary.values[ind])
    orig = vec.UpOrig.values[ind]

    hz = vec.attrs['Sampling Rate (Hz)']

    return waveCorrectedSpectrum(dataOrig, u, v, orig, seconds=seconds, hz=hz, fullOut=fullOut, useW=useW)

#endregion

#endregion

#endregion


#region semi-idealized model functions

#region idealized spectrum helper functions
def uToX(u,t,x0):
    #integrates velocity over time to determine displacemnt for model
    ux = x0 + np.concatenate(([0],np.array(np.cumsum((u[1:]+u[:-1])/2*np.diff(t)))))
    return ux

def E(kappa,ep,L):
    eta = ((10**-6)**(3)/ep)**(1/4)
    fl = ((kappa*L)/(((kappa*L)**2+4*(np.pi**2))**(1/2)))**(5/3+2)
    fn = np.exp(-5.2*(((((kappa*eta)**4)+(.4**4))**(1/4))-.4))
    return 1.5*(ep**(2/3))*(kappa**(-5/3))*fl*fn

def integrand33(kappa,k1,ep,L):
    return (E(kappa,ep,L)/kappa)*(1+((k1**2)/(kappa**2)))/2
#endregion

#region IdealSpectrum
def IdealSpectrum(epsilon, L=2, dx=0.0001, T=30*60, Umax=.5, fullOut = False):

    #total distance of our test
    D = 4*T*Umax

    #wavenumbers of our ideal spectra
    kIdeal = 2*np.pi*np.arange(1/D,1/(2*dx)+1/D,1/D) 

    #create a lower resolution k for calculating ideal spectrum
    kTemp = np.concatenate((np.linspace(kIdeal[0],\
        kIdeal[int(kIdeal.size/100)],100*L),np.linspace(\
            kIdeal[int(kIdeal.size/100)+1],kIdeal[-1],\
                10000-100)))

    #calculate ideal spectrum
    specTemp = np.array([quad(integrand33,k1,np.inf,\
        args=(k1,epsilon,L),epsabs=-1,epsrel=10**-3)[0]\
             for k1 in kTemp])

    #interpolate to full resolution spectrum
    interp = interp1d(kTemp,specTemp,kind='cubic',assume_sorted=True)
    specIdeal = interp(kIdeal)
    
    if fullOut:
        return kIdeal,specIdeal,kTemp,specTemp
    else:
        return kIdeal,specIdeal
    
#endregion

#region InertialSubrange
def InertialSubrange(epsilon, dx=0.0001, T=30*60, Umax=.5):
    
    #constants in the intertial subrange eqn
    C = (18/55)*1.5*(4/3)

    #total distance of our test
    D = 4*T*Umax

    #wavenumbers of our ideal spectra
    kIdeal = 2*np.pi*np.arange(1/D,1/(2*dx)+1/D,1/D) 

    kTemp = np.concatenate((np.linspace(kIdeal[0],\
        kIdeal[int(kIdeal.size/100)],100*L),np.linspace(\
            kIdeal[int(kIdeal.size/100)+1],kIdeal[-1],\
                10000-100)))

    #epsilons = (np.exp(b)/C)**(3/2)
    
    # y intercept of log(y) = -5/3*log(k) + b
    b = np.log((epsilon)**(2/3) * C)

    specIdeal = np.exp(((-5/3)*np.log(kIdeal))+b)
    specTemp = np.exp(((-5/3)*np.log(kTemp))+b)
    
    return kIdeal,specIdeal,kTemp,specTemp

#region IdealData:

def IdealData(kIdeal, specIdeal, dx=0.0001, T=30*60, Umax=.5):

    #add 0 frequency (0 mean)
    s = np.concatenate(([0],specIdeal))

    #divide repeated amplitudes by 2
    s[1:-1] = s[1:-1]/2
    
    #total distance of our test
    D = 4*T*Umax

    #x positions of ideal spectra
    x = np.arange(0,D,dx)

    #convert to fourier coefficients with random phases
    theta = np.random.rand(s.size)*2*np.pi
    f = (s*2*np.pi/D)**(1/2)*np.exp(1j*theta)

    #expand to have full fourier coefficients
    f = np.concatenate((f,np.conj(np.flip(f[1:-1]))))*x.size

    #Calculate spacial turbulence data
    wx = np.real(np.fft.ifft(f))

    return x,wx

#endregion

#region semiIdealData
def semiIdealData(x, wx, uReal, T=30*60, hz=16):

    #Pick a spot to make temporal observations
    loc = int(x.size/2)

    #set up timestamp with which to take observations
    t = np.arange(0,T,1/hz)

    #set up interpolation
    winterp = interp1d(x,wx)

    xObs = uToX(uReal,np.arange(uReal.size)/hz,x[loc])
    wt = winterp(xObs)

    return t,wt

#endregion

#endregion

#endregion

#region dissipation calculation functions

#region dissipationFit

def dissipationFit(freqk,speck,dof,nu=10**-6,L=1,eptest=10**-3,outFull=False, generationScaling = 1/2,\
    sizeCutoff=2.5, peakLimit=.8, debug=False, minDataPoints = 10, slopeConfidence = .95, kdeBW = 1):
    #fit a -5/3 slope and calculate a dissipation value to wavenumber spectrum
    #freqk is an array of the wavenumbers (2pi/wavelength), while speck is
    #an array of the spectrum values, and dof is an array of the degrees of
    #freedom for each spectral value.
    #nu is the kinematic viscosity of water during the time period
    #L is the expected generation length scale of the turbulence
    #eptest should be an overestimate of the expected dissipation
    #eptest is used as an initial guess for defining the beginning of the
    #dissipation subrange
    #if outFull=True, output info about the final fit (epsilon distribution and fit type, etc.) 
    #note, fitType classifies what type of fit resulted 
    # (0 = not a long enough spectrum to even try, 1 = no good fits, 
    # 2 = no fits that satisfy their own disspation derived cutoffs, 3 = good fits )
    #generationScaling is the factor of the generation length scale that a wavenumber should
    #be below for it to be possibly within the inertial subrange
    #minDataPoints is how many spectrum values must be in a segment to test if it is in the inertial subrange
    #sizeCutoff is the minimum range of wavenumbers in a segment to test if it is in the inertial subrange
    #note, sizeCutoff = 2.5 is requiring a quarter decade wavenumber span
    #slopeConfidence is the confidence interval used for determining if the segment could be the inertial subrange based off a -5/3 slope
    #PeakLimit is the prominence cutoff used for determining if there is a peak in the segment being used to fit an inertail subrange
    #kdeBW is a multiplicative factor on Scott's rule for determining the bandwidth of the KDE

    #set the kilmogorov scaling factor for where the inertial subrange ends
    #based on empirical lab experiments referenced in Pope 2000.
    kolmogorovScaling = 60

    # fit a -5/3 slope to the data
    m = -5/3

    #constants in the intertial subrange eqn.
    C = (18/55)*1.5*(4/3)
    
    #calculate an initial kolmogorov lengthscale based on eptest
    eta = (nu**3/(eptest))**(1/4)

    # Get rid of nans, and entries with no degrees of freedom, if there are any
    iNan = np.logical_and(~np.isnan(freqk),dof>0)
    freqk = freqk[iNan]
    speck = speck[iNan]
    dof = dof[iNan]

    # convert to log space, including adjustments based on chi^2 distribution
    xhat = np.log(freqk)
    yhat = np.log(speck)+np.log(dof/2)-digamma(dof/2)

    #calulate variance based on dof and gaussian
    varhat = 2/dof

    # Define Bounds within which the inertfial subrange must exist
    kmax = (2*np.pi)/(eta*kolmogorovScaling)
    kmin = (2*np.pi)/(L*generationScaling)
    freqBounds = (kmin,kmax)
    Boundsi = np.array([iNear(freqk,freqBounds[0]),iNear(freqk,freqBounds[1])])
    
    #how many sections of length 10 or more we can make inside these limits

    leng = ((np.diff(Boundsi)-(minDataPoints-2))**2 + (np.diff(Boundsi)-(minDataPoints-2)))//2

    #initialize arrays and tracking variables
    I = np.empty(leng,dtype='int')  #tracking start of segment index
    J = np.empty(leng,dtype='int')  #tracking end of segment index
    ktest1 = np.empty(leng) #tracking last wavenumber
    ktest2 = np.empty(leng) #tracking second to last wavenumber
    sectionSize = np.empty(leng)
    
    loc = 0

    if np.diff(Boundsi) >= minDataPoints:
        for i in np.arange(Boundsi[0],Boundsi[1]-(minDataPoints-2)):
            for j in np.arange(i+(minDataPoints-1),Boundsi[1]+1):

                #store beginning and end of fit
                I[loc] = i
                J[loc] = j

                #store the last 2 wavenumbers in the fit for later checks
                ktest1[loc] = freqk[j]
                ktest2[loc] = freqk[j-1]

                sectionSize[loc] = freqk[j]/freqk[i]

                #iterate
                loc += 1

        okInd = sectionSize >= sizeCutoff

        #if no fits remain, exit with fitType=0
        if np.sum(okInd) <= 1:
            fitType = 0
            if outFull:
                return np.nan,np.array([np.nan]),np.array([np.nan]),np.nan,np.nan,np.array([np.nan]),eta,fitType,0,np.array([np.nan]),np.array([np.nan])
            else:
                return np.nan,np.array([np.nan]),np.array([np.nan]),np.nan,np.nan

        I = I[okInd]
        J = J[okInd]
        ktest1 = ktest1[okInd]
        ktest2 = ktest2[okInd]

        leng = I.size

        a = np.empty(leng)              #tracking fit slope
        b = np.empty(leng)              #tracking fit constant
        slopeErr = np.empty(leng)       #tracking normalized slope err as a goodness of fit estimate
        peakProm = np.empty(leng)       #tracking how much pattern there is in the residual
        squareError = np.empty(leng)

        bestI = 0

        for i in np.arange(leng):

            #index of segment
            ind = np.arange(I[i],J[i]+1)                              

            #y data to fit to
            yt = yhat[ind]     

            #x data to fit to                                
            xt = xhat[ind]   

            #variance of data to fit to
            var = varhat[ind]

            #Fitting variables
            Afit = np.sum(xt/var)
            Bfit = np.sum(1/var)
            Cfit = np.sum(yt/var)
            Dfit = np.sum((xt**2)/var)
            Efit = np.sum((xt*yt)/var)

            #fit results

            #store slope
            a[i] = (Efit*Bfit-Cfit*Afit)/(Dfit*Bfit-Afit**2)

            #calculate y intercept based on a -5/3 slope specifically
            b[i] = (Cfit+(5/3)*Afit)/Bfit

            #calculate error on slope coefficient
            asig = np.sqrt(Bfit/(Bfit*Dfit-Afit**2))

            #calculate asig normalized error of slope prediction off -5/3 slope
            slopeErr[i] = (np.abs(a[i]-(-5/3)))/asig

            #calculate residual of fit
            yresidual = yt - ((-5/3)*xt + b[i])

            #calculate square error
            squareError[i] = np.sum(yresidual**2)

            #find prominence of most prominent peak and min
            try: 
                peaks = np.max(signal.peak_prominences(yresidual,signal.find_peaks(yresidual)[0])[0])
            except:
                peaks = 0

            try:
                mins = np.max(signal.peak_prominences(-yresidual,signal.find_peaks(-yresidual)[0])[0])
            except:
                mins = 0

            #store larges prominence
            peakProm[i] = np.max([peaks,mins])

        #check if -5/3 slope is outside of 95% confidence interval
        alpha = 1-slopeConfidence
        indslope = slopeErr<stats.norm.ppf(1-alpha/2)
        indPeak = peakProm < peakLimit
        indGood = np.logical_and(indslope,indPeak)
        indnew = indGood.copy()

        if debug:
            pdb.set_trace()

        #if no fits remain, exit with fitType=0
        if np.sum(indnew) <= 1:
            fitType = 1
            if outFull:
                return np.nan,np.array([np.nan]),np.array([np.nan]),np.nan,np.nan,np.array([np.nan]),eta,fitType,np.sum(~indPeak),np.array([np.nan]),np.array([np.nan])
            else:
                return np.nan,np.array([np.nan]),np.array([np.nan]),np.nan,np.nan

        #calculate dissipations based on y intercept
        epsilons = (np.exp(b)/C)**(3/2)

        bwFactor = kdeBW*((np.sum(indnew))**(-1./5))

        #calculate a density kernel of the remaining dissipations
        kernel = stats.gaussian_kde(np.log10(epsilons[indnew]),bw_method=bwFactor)

        #use dissipation closest to peak probability as calculated dissipation
        epCalc = epsilons[indnew][np.argmax(kernel(np.log10(epsilons[indnew])))]

        #kolmogorov scale of calculated dissipation
        etatest = (nu**3/(epCalc))**(1/4)

        #adjust maximum wavenumber
        kmax = (2*np.pi)/(etatest*kolmogorovScaling)

        #initialize counter for catching infinite loops
        nup = 0

        #assume fit works appropriately, which would return fitType=3
        fitType = 3

        #iterate through until we have a fit where the wave numbers used satisfy the kolmogorov scale 
        #predicted by that fit, or until we get stuck in an infinite loop
        #(caught by when we end up predicting a higher dissipation (go backwards) than the last at least 5 times)
        #note, first condition uses ktest2, allowing for a relaxed boundary where 1 wavenumber can go beyond the
        #kolmogorov scale based cutoff if necessary.
        while np.logical_or(np.max(ktest2[indnew])>kmax,np.logical_and(epCalc>eptest,nup<5)):
            
            #last dissipation sets our new limit for where to search for best fit
            eptest = epCalc

            #use predicted komogorov scale to identify fits that are possibly in the inertial subrange
            indold = indnew.copy()
            indnew = np.logical_and(ktest1 <= kmax,indGood)

            #If we don't find any, assume that we aren't resolving the inertial subrange sufficiently
            #in which case, return the last succesfull fit, but set fitType = 0
            if np.sum(indnew) <= 1:
                indnew = indold
                fitType = 2
                break
            
            bwFactor = kdeBW*((np.sum(indnew))**(-1./5))

            #recalculate the density kernel for new dissipation distribution
            kernel = stats.gaussian_kde(np.log10(epsilons[indnew]),bw_method=bwFactor)

            #redetermine the optimal dissipation
            epCalc = epsilons[indnew][np.argmax(kernel(np.log10(epsilons[indnew])))]

            #recalculate the kolmogorov length scale
            etatest = (nu**3/epCalc)**(1/4)

            #adjust maximum wavenumber
            kmax = (2*np.pi)/(etatest*kolmogorovScaling)

            #if we found a dissipation that was higher than before, increase counter for avoiding infinite loops
            if epCalc>eptest:
                nup += 1

        #calculate y intercept of optimal dissipation
        bfit = np.log((epCalc**(2/3))*C)

        #find range of indices that were used in this final calculation
        indfit = np.arange(np.min(I[indnew]),np.max(J[indnew])+1)

        #create -5/3 fit line to inertial subrange
        xfit = np.log10(freqk[indfit])
        yfit = m*xfit + np.log10(np.exp(bfit))

        #use standard deviation of epsilon distribution as error
        epCalcErr = (np.percentile(epsilons[indnew],97.5)-np.percentile(epsilons[indnew],2.5))/2

        #use average slope error pvalue as pvalue of dissipation fit
        pvalReturn = np.nanmean(stats.norm.sf(slopeErr[indnew]))

        if debug:
            pdb.set_trace()

        if outFull:
            return epCalc,xfit,yfit,epCalcErr,pvalReturn,epsilons[indnew],etatest,fitType,np.sum(~indPeak),squareError[indnew], J[indnew]+1-I[indnew]
        else:
            return epCalc,xfit,yfit,epCalcErr,pvalReturn

    #if not enough points in potential inertial subrange region, return nans for everything
    else:
        epsilon = np.nan
        xfit = np.array([np.nan])
        yfit = np.array([np.nan])
        epErr = np.nan
        pval = np.nan
        fitType = 0
        if outFull:
            return epsilon,xfit,yfit,epErr,pval,np.array([np.nan]),eta,fitType,0,np.array([np.nan]),np.array([np.nan])
        else:
            return epsilon,xfit,yfit,epErr, pval

#endregion

#region size
def size(list):
    return np.max([np.size(temp) for temp in list])
#endregion

#region fillArray
def fillArray(array,size):
    return np.array([np.append(temp,np.empty(size-temp.size)*np.nan) for temp in array])
#endregion

#region epCalc

def epCalc(vec,name,segSize = 10, highLowSep = 1/5, turbVarCutoff = 5, waveCutoff = 1.025, unorigionalCutoff = 0.01,\
    lowWavenumberRemoval = 2, highSpectrumRemoval = 4, binSize = 50, genScale = 1/2, minDataPoints = 10,\
        minWavenumberSpan = 2.5, slopeConfidence = .95, peakProminence = 0.8, kdeBW=1, whiteNoiseRemove = 0, correct = True):
    #Calculate dissipation values for each 30 minute burst in vec dataset
    #name is a string with the vector number for saving the data
    #segSize, highLowSep, turbVarCutoff, waveCutoff, unorigionalCutoff,lowWavenumberRemoval, highSpectrumRemoval, 
    #and binSize are variables for calculating the spectrum
    #genScale, kolScale, minDataPoints, minWavenumberSpan, slopeConfidence, PeakProminence, and kdeParams
    #are variables for fitting to the inertial subrange
    #segSize is how many seconds are used for setting an individual segment when calculating the spectrum
    #highLowSep is the frequency cutoff for determining the high and low frequency portions of the signal
    #turbVarCutoff is the cutoff used for determining if the turbulence is too big relative to the mean current for frozen turbulence
    #waveCutoff is the cutoff used for determining if the wave driven velocity variance is too big relative to the mean current
    #unorigionalCutoff is the cutoff used for determining when to stop eliminating segments of data because they have unorigional data points
    #lowWavenumberRemoval is the number of low wavenumber points removed from the spectrum to avoid bias
    #highSpectrumRemoval is the factor avove the nyquist frequency we require a spectrum value to be to be kept for avoiding aliasing bias
    #binSize is the minimum number of points used in each bin when averaging the spectrum
    #genScale is how much smaller we consider the generation lengthscale when determining where the inertial subrange can be
    #minDataPoints is how many spectrum values must be in a segment to test if it is in the inertial subrange
    #minWavenumberSpan is the minimum range of wavenumbers in a segment to test if it is in the inertial subrange
    #note, minWavenumberSpan = 2.5 is requiring a quarter decade wavenumber span
    #slopeConfidence is the confidence interval used for determining if the segment could be the inertial subrange based off a -5/3 slope
    #PeakProminence is the prominence cutoff used for determining if there is a peak in the segment being used to fit an inertail subrange
    #kdeParams are parameters fed to the kde calculation function for determining the PDF of dissipations
    #if kdeParams is None, the default parameters of the function are used.
    #whiteNoiseRemove is used for testing removing white noise levels from the spectrum before calculating dissipation
    #correct is a boolean to determine whether or not to apply the high frequency wave correction constant to each segment when calculating the spectrum

    #pull burst numbers and sampling frequency
    bnum = vec.burst30.values
    hz = vec.attrs['Sampling Rate (Hz)']

    #pull time of each burst
    t = vec.avTime30.values

    #initialize variables
    temp = np.empty(bnum.size)*np.nan
    pressure = np.empty(bnum.size)*np.nan
    nu = np.empty(bnum.size)*np.nan

    uAv = np.empty(bnum.size)*np.nan
    uStd = np.empty(bnum.size)*np.nan

    ep = np.empty(bnum.size)*np.nan
    er = np.empty(bnum.size)*np.nan
    pv = np.empty(bnum.size)*np.nan
    kept = np.empty(bnum.size)*np.nan
    fitType = np.empty(bnum.size)*np.nan
    peaksThrown = np.empty(bnum.size)*np.nan
    UrmsSW = np.empty(bnum.size)*np.nan
    UrmsIG = np.empty(bnum.size)*np.nan
    maxBad = np.empty(bnum.size)*np.nan
    maxConsec = np.empty(bnum.size)*np.nan

    k = [np.empty(0)]*bnum.size
    s = [np.empty(0)]*bnum.size
    dof = [np.empty(0)]*bnum.size
    epDist = [np.empty(0)]*bnum.size
    xfit = [np.empty(0)]*bnum.size
    yfit = [np.empty(0)]*bnum.size
    squareError = [np.empty(0)]*bnum.size
    fitLength = [np.empty(0)]*bnum.size

    #Interpolate temperature to full time frequency
    tempAll = vec.Temp.interp(time_sen=vec.time)

    #loop through each burst
    for i in np.arange(bnum.size):

        print('burst ' + str(i+1) + ' of ' + str(bnum.size) + ', v' + name)

        #pull indices for the current burst
        ind = vec.bNum30.values == bnum[i]

        #get average temperature and pressure for this burst
        temp[i] = np.nanmean(tempAll.values[ind])
        pressure[i] = np.nanmean(vec.Pressure.values[ind])

        #get average velocity and velocity variance for thi burst
        uAv[i] = np.nanmean(vec.Primary.values[ind])
        uStd[i] = np.nanstd(vec.Primary.values[ind])

        #calculate average dynamic viscosity for this burst
        nuTemp = temp[i]+273.15
        nuPress = pressure[i]/100 + 0.101325
        nu[i] = iapws95.IAPWS95_PT(nuPress,nuTemp).nu

        #eliminate nans from each velocity component
        w = nan_interp(vec.Up.values[ind])
        u = nan_interp(vec.Primary.values[ind])
        v = nan_interp(vec.Secondary.values[ind])

        #Calculate IG and SS root mean square velocities and pressures (making sure we don't have only nans first to avoid an error)
        if np.isnan(u)[0]:
            UrmsSW[i] = np.nan
            UrmsIG[i] = np.nan
        else:
            UrmsIG[i] = bandVrms(u,hz,0.04,0.004)
            UrmsSW[i] = bandVrms(u,hz,0.2,0.04)

        #store which vertical velocity points are original data for this burst
        orig = vec.UpOrig.values[ind]

        #calculate spectrum
        (k[i],s[i],dof[i],_,_,keptTemp,maxBad[i],maxConsec[i]) = waveCorrectedSpectrum(w,u,v,orig,seconds=segSize,hz=hz,fullOut=True,useW=False,\
            highLowSep=highLowSep, turbVarCutoff = turbVarCutoff, waveCutoff = waveCutoff, unorigionalCutoff = unorigionalCutoff,\
                lowWavenumberRemoval = lowWavenumberRemoval, highSpectrumRemoval = highSpectrumRemoval, binSize = binSize, correct=correct)

        #store total number of kept segments used for this spectrum
        kept[i] = np.sum(keptTemp)

        #remove noise from spectrum for testing if specified
        if whiteNoiseRemove > 0:
            snew = s[i] - whiteNoiseRemove
            whiteNoiseIndex = snew < 0
            snew[whiteNoiseIndex] = np.nan
            s[i] = snew

        #chech if spectrum is too short, then calculate dissipation from the spectrum
        if np.logical_or(np.isnan(k[i][0]),np.size(k[i])<10):
            ep[i] = np.nan
            xfit[i] = np.empty(0)
            yfit[i] = np.empty(0)
            er[i] = np.nan
            pv[i] = np.nan
            epDist[i] = np.empty(0)
            fitType[i] = 0
        else:
            (ep[i],xfit[i],yfit[i],er[i],pv[i],epDist[i],_,fitType[i],peaksThrown[i],squareError[i],fitLength[i]) = dissipationFit(k[i],s[i],dof[i],nu=nu[i],\
                L=1,eptest=10**-3,outFull=True, generationScaling = genScale, sizeCutoff=minWavenumberSpan,\
                    peakLimit = peakProminence, debug=False, minDataPoints = minDataPoints,\
                    slopeConfidence = slopeConfidence, kdeBW = kdeBW)

    #determine dimension sizes for xarray dataset
    kSize = size(k)
    xSize = size(xfit)
    epSize = size(epDist)

    #store errors
    sysEr = ep*0.152
    maxEr = np.sqrt(sysEr**2+er**2)

    #create xarray dataset and store results
    results = xr.Dataset()
    
    results.coords['time'] = t
    results.coords['specCoord'] = np.arange(kSize)
    results.coords['fitCoord'] = np.arange(xSize)
    results.coords['distCoord'] = np.arange(epSize)

    results['bnum'+name] = ('time',bnum)

    results['temp'+name] = ('time',temp)
    results['pressure'+name] = ('time',pressure)
    results['nu'+name] = ('time',nu)

    results['uMean'+name] = ('time',uAv)
    results['uStd'+name] = ('time',uStd)

    results['kept'+name] = ('time',kept)
    results['ep'+name] = ('time',ep)

    results['statEr'+name] = ('time',er)
    results['statEr'+name].attrs['description'] = 'Statistical Error caclulated from distribution of dissipations from all possible fits to inertial subrange.'
    results['sysEr'+name] = ('time',sysEr)
    results['sysEr'+name].attrs['description'] = 'Systematic Error of methods calculated from semi-idealized model tests.'
    results['maxEr'+name] = ('time',maxEr)
    results['maxEr'+name].attrs['description'] = 'Geometric Sum of Statistical Error and Systematic Error. Because those two errors are not completely independent, this is likely an overestimate of the true error'

    results['pv'+name] = ('time',pv)
    results['fitType'+name] = ('time',fitType)
    results['peaksThrown'+name] = ('time',peaksThrown)
    results['UrmsSW'+name] = ('time',UrmsSW)
    results['UrmsIG'+name] = ('time',UrmsIG)
    results['maxBad'+name] = ('time',maxBad)
    results['maxConsec'+name] = ('time',maxConsec)

    results['k'+name] = (('time','specCoord'),fillArray(k,kSize))
    results['s'+name] = (('time','specCoord'),fillArray(s,kSize))
    results['dof'+name] = (('time','specCoord'),fillArray(dof,kSize))
    results['xfit'+name] = (('time','fitCoord'),fillArray(xfit,xSize))
    results['yfit'+name] = (('time','fitCoord'),fillArray(yfit,xSize))
    results['epDist'+name] = (('time','distCoord'),fillArray(epDist,epSize))
    results['squareError'+name] = (('time','distCoord'),fillArray(squareError,epSize))
    results['fitLength'+name] = (('time','distCoord'),fillArray(fitLength,epSize))

    results.attrs['spectrum_segment_size(s)'] = segSize
    results.attrs['spectrum_highLow_separation_freq(hz)'] = highLowSep
    results.attrs['turbulence_variance_cutoff'] = turbVarCutoff
    results.attrs['wave_variance_cutoff'] = waveCutoff
    results.attrs['Fraction_unorigional_points_requirement'] = unorigionalCutoff
    results.attrs['Number_of_low_wavenumbers_removed'] = lowWavenumberRemoval
    results.attrs['spectrum_high_wavenumber_removal_cutoff'] = highSpectrumRemoval
    results.attrs['minimum_spectrum_bin_size'] = binSize
    results.attrs['generation_length_scale_modification'] = genScale
    results.attrs['min_data_points_in_inertial_subrange'] = minDataPoints
    results.attrs['min_wavenumber_span_in_inertial_subrange'] = minWavenumberSpan
    results.attrs['inertial_subrange_slope_confidence_interval'] = slopeConfidence
    results.attrs['peak_prominence_cutoff'] = 0.8
    results.attrs['kde_bandwidth_modification'] = 1

    #return results
    return results
#endregion

#endregion