import math as math
import numpy as np
import urllib
import urllib2
import os as os
import sys as sys
import warnings
import fitsio
from fitsio import FITS, FITSHDR
import subprocess
import argparse
import pdb
import pandas

from astropy import units
from astropy.coordinates import SkyCoord

try:
    os.environ['DISPLAY']
except KeyError as Err:
    warnings.warn('No display environment! Using matplotlib backend "Agg"')
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import kali.lc
except ImportError:
    print 'kali is not setup. Setup kali by sourcing bin/setup.sh'
    sys.exit(1)


@np.vectorize
def luptitude_to_flux(mag, err, band):  # Converts a Luptitude to an SDSS flux
    b_ = dict(u=1.4e-10, g=0.9e-10, r=1.2e-10, i=1.8e-10, z=7.4e-10)  # band softening parameter
    f0 = 3631.0  # Jy
    flux = math.sinh(math.log(10.0)/-2.5*mag-math.log(b_[band]))*2*b_[band]*f0
    error = err*math.log(10)/2.5*2*b_[band]*math.sqrt(1+(flux/(2*b_[band]*f0))**2)*f0
    return flux, error

def time_to_restFrame(time, z):  # Converts a cadence to the rest frame
    t = np.zeros(time.shape[0])
    if (z > 0):
    	t[1:] = np.absolute(np.cumsum((time[1:] - time[:-1])/(1 + z)))
    else:
    	print("observer frame")
        t = np.absolute(time - time[0])
    return t

    #return t


class pandaLC(kali.lc.lc):

    def _read(self, name,  path ,band):
        fileName = str(path+name)

        catalogue = pandas.read_csv("/Users/Jackster/Research/code/stripe82-class/"+'catalog.txt', sep=', ', delimiter=' ')
        labelled = catalogue[catalogue.cl != 'unknown']
 
	redshift0 = labelled.zQSO[labelled.ID == float(name)].astype(float)
	redshift = redshift0.values[0]
    	#observer-frame sanity check fit uncomment below that is all
    	#redshift = 0.
    	
	cols = ['mjd', 'band', 'mag', 'magerr']

	df = pandas.read_csv(path+"LC_"+name+".dat", sep=', ', delimiter=' ', names = ["mjd","band","mag","magerr"])
	self.band = band
        print(band)                  
        rband = df[df.band == band]
        rband = rband[rband.mag != -99].sort_values(by='mjd')
        #---------fluxes
	flux, err  = luptitude_to_flux(rband.mag.values,rband.magerr.values, band)
	flux = np.require(flux, requirements=['F', 'A', 'W', 'O', 'E'])
	#flux = (flux-np.median(flux))/np.median(flux)
	err = np.require(err, requirements=['F', 'A', 'W', 'O', 'E'])
	#err  = err/np.median(flux)
	#---------time
	MJD = rband.mjd.values
	time = time_to_restFrame(MJD.astype(float), redshift)
	time = np.require(time, requirements=['F', 'A', 'W', 'O', 'E'])
        # ---final data arrays to pass to self
        t = time
        y = flux
        yerr = err
        cadence = np.require(np.arange(0, len(time),1), requirements=['F', 'A', 'W', 'O', 'E'])
        mask = np.require(np.zeros(len(time)), requirements= ['F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        mask[:] = int(1)
        #---------------------------------------------------------
        self.startT = 1.
        self.T = time[-1]-time[0]
        dt = np.absolute(np.min(time[1:]-time[:-1]))
        self._dt = dt # Increment between epochs.
        self.mindt = dt
        print(dt, self.mindt)
        
	self._numCadences = len(time)
        self.numCadences = len(time)
        self.cadence = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.mask = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])  # Numpy array of mask values.
        self.t = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        #self.terr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.x = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.y = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        self.yerr = np.require(np.zeros(self.numCadences), requirements=['F', 'A', 'W', 'O', 'E'])
        
        # to make kali run faster delete any masked points, also this will take care of times values that are set to zero for masked fluxes
        #mask only checked for fluxes not for time values: code will hang up for nans or timestamps that are set to zero       
        self.cadence  = cadence
        self.mask = mask     
    	self.t = t
    	#self.terr = terr[w] 
    	self.y = (y[:]-np.median(y[:]))/np.median(y[:])
        self.yerr = yerr[:]/np.median(y[:])

    def read(self, name, path ,band ,**kwargs):
        self.z = kwargs.get('z', 0.0)
	fileName = name
    	if path is None:
            try:
                self.path = os.environ['DATADIR']
            except KeyError:
                raise KeyError('Environment variable "DATADIR" not set! Please set "DATADIR" to point \
                where csv data should live first...')
        else:
            self.path = path
        filePath = os.path.join(self.path, fileName)
        print("Lightcuvre is band ")
	print(band)

        self._name = str(name)  # The name of the light curve (usually the object's name).
        self._band = str(band)  # The name of the photometric band (eg. HSC-I or SDSS-g etc..).
        self._xunit = r'$t$~(MJD)'  # Unit in which time is measured (eg. s, sec, seconds etc...).
        self._yunit = r'$F$~($\mathrm{e^{-}}$)'  # Unit in which the flux is measured (eg Wm^{-2} etc...).

    	Ret = True

        if Ret == True:
             self._read(name, self.path, self.band)
        else:
            raise ValueError('csv light curve not found!')

    def write(self, name, path=None, **kwrags):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--ID', type=str, default='iizw229015_kepler_q04_q17', help=r'pass filename as -id arg')
    parser.add_argument('-z', '--z', type=float, default='0.', help=r'object redshift')
    parser.add_argument('-skiprows', '--skiprows',  type=int, default='0', help=r'how many rows to skip in csv file')
    args = parser.parse_args()

    LC = csvLC(name=args.ID,  z=args.z, skipNrows=args.skiprows)

    LC.plot()
    LC.plotacf()
    LC.plotsf()
    plt.show()