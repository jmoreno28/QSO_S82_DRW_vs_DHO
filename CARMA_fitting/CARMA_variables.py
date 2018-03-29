import numpy as np
import math as math
import cmath as cmath
import psutil as psutil
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import gridspec as gridspec
import argparse as argparse
import operator as operator
import warnings as warnings
import copy as copy
import time as time
import pdb
import os as os
import random

import kali.variables
#import kali.s82
import kali.carma
import kali.util.mcmcviz as mcmcviz
from kali.util.mpl_settings import set_plot_params
import kali.util.triangle as triangle
import pickle
import pandas


#---------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
#info for data file to be read
parser.add_argument('-id', '--id', type=str, default='iizw229015_kepler_q04_q17', help=r'pass filename as -id arg')
parser.add_argument('-z', '--z', type=float, default='0.3056', help=r'object redshift')
parser.add_argument('-band', '--band', type=str, default='i', help=r'bandpass')
parser.add_argument('-skiprows', '--skiprows',  type=int, default='0', help=r'how many rows to skip in csv file')
parser.add_argument('-N', '--N',  type=int, default='0', help=r'number of points per period')
parser.add_argument('-sampler', '--sampler',  type=str, default='0', help=r'sampler type : chop, regular , irregular, smart')
# params for CARMA fit and plotting
parser.add_argument('-libcarmaChain', '--lC', type = str, default = 'libcarmaChain', help = r'libcarma Chain Filename')
parser.add_argument('-nsteps', '--nsteps', type = int, default =1000, help = r'Number of steps per walker')
parser.add_argument('-nwalkers', '--nwalkers', type = int, default = 25*psutil.cpu_count(logical = True), help = r'Number of walkers')
parser.add_argument('-pMax', '--pMax', type = int, default = 2, help = r'Maximum C-AR order')
parser.add_argument('-pMin', '--pMin', type = int, default = 1, help = r'Minimum C-AR order')
parser.add_argument('-qMax', '--qMax', type = int, default = -1, help = r'Maximum C-MA order')
parser.add_argument('-qMin', '--qMin', type = int, default = -1, help = r'Minimum C-MA order')
parser.add_argument('--plot', dest = 'plot', action = 'store_true', help = r'Show plot?')
parser.add_argument('--no-plot', dest = 'plot', action = 'store_false', help = r'Do not show plot?')
parser.set_defaults(plot = True)
parser.add_argument('-minT', '--minTimescale', type = float, default = 2.0, help = r'Minimum allowed timescale = minTimescale*lc.dt')
parser.add_argument('-maxT', '--maxTimescale', type = float, default = 0.5, help = r'Maximum allowed timescale = maxTimescale*lc.T')
parser.add_argument('-maxS', '--maxSigma', type = float, default = 5.0, help = r'Maximum allowed sigma = maxSigma*var(lc)')
parser.add_argument('--stop', dest = 'stop', action = 'store_true', help = r'Stop at end?')
parser.add_argument('--no-stop', dest = 'stop', action = 'store_false', help = r'Do not stop at end?')
parser.set_defaults(stop = False)
parser.add_argument('--save', dest = 'save', action = 'store_true', help = r'Save files?')
parser.add_argument('--no-save', dest = 'save', action = 'store_false', help = r'Do not save files?')
parser.set_defaults(save = False)
parser.add_argument('--log10', dest = 'log10', action = 'store_true', help = r'Compute distances in log space?')
parser.add_argument('--no-log10', dest = 'log10', action = 'store_false', help = r'Do not compute distances in log space?')
parser.set_defaults(log10 = False)
parser.add_argument('--viewer', dest = 'viewer', action = 'store_true', help = r'Visualize MCMC walkers')
parser.add_argument('--no-viewer', dest = 'viewer', action = 'store_false', help = r'Do not visualize MCMC walkers')
parser.set_defaults(viewer = True)
args = parser.parse_args()


#-----------------------------------------------------------------------------

"""
Define S82 object by calling the kali class variables (a module in the kali directory)
 pandaLC  method in variables class utilizes panda package to read light curve files
 """
#name = "5671550"
name = args.id
band = args.band
s82 = kali.variables.pandaLC(name, path='/Users/Jackster/Research/code/stripe82-class/data/AllLCs/', band = band)

#-----------------------------------------
#set timescale priors, it is possible the timescales resolved might be less than the min dt in the time array, this is a naive assumption
t = s82.t
mindt = np.min(t[1:]-t[:-1])
maxdt = np.max(t[1:]-t[:-1])

#s82.minTimescale = mindt
#s82.maxTimescale = maxdt 
s82.minTimescale = 2.
s82.maxTimescale = 1000. 
s82.maxSigma = args.maxSigma

#-------------------------------------------------------------------------------
# Perform the DHO CARMA fit using method CARMATask and output the DIC for the fit
# Code can be easily generalized to do multiple orders in a for loop
taskDict = dict()
DICDict= dict()

p=2
q = p-1

nt = kali.carma.CARMATask(p, q, nwalkers = args.nwalkers, nsteps = args.nsteps )
print 'Starting libcarma fitting for p = %d and q = %d...'%(p, q)
startLCARMA = time.time()
nt.fit(s82)
stopLCARMA = time.time()
timeLCARMA = stopLCARMA - startLCARMA
print 'libcarma took %4.3f s = %4.3f min = %4.3f hrs'%(timeLCARMA, timeLCARMA/60.0, timeLCARMA/3600.0)

Deviances = copy.copy(nt.LnPosterior[:,args.nsteps/2:]).reshape((-1))
DIC = 0.5*math.pow(np.std(-2.0*Deviances),2.0) + np.mean(-2.0*Deviances)
print 'C-ARMA(%d,%d) DIC: %+4.3e'%(p, q, DIC)
DICDict['%d %d'%(p, q)] = DIC
taskDict['%d %d'%(p, q)] = nt

#-------------------------------------------------------------------------------

#Save the carma object as a pickle file
fname = str(name+"-"+'%i-%i'%(p, q)+band+"band"+'KaliObject')
output = open(fname+'.pkl', 'wb')
pickle.dump(taskDict['%i %i'%(p, q)],output)	
output.close()


sortedDICVals = sorted(DICDict.items(), key=operator.itemgetter(1))
pBest = int(sortedDICVals[0][0].split()[0])
qBest = int(sortedDICVals[0][0].split()[1])

bestTask = taskDict['%d %d'%(pBest, qBest)]
Theta = bestTask.Chain[:, np.where(bestTask.LnPosterior == np.max(bestTask.LnPosterior))[
    0][0], np.where(bestTask.LnPosterior == np.max(bestTask.LnPosterior))[1][0]]
nt = kali.carma.CARMATask(pBest, qBest)
nt.set(s82.dt, Theta)
nt.smooth(s82)

#-----------------------------Plotting Diagnositics
#sdssLC.plot(doShow = True)
A = s82.plot(doShow = False)
A.savefig(name + '_rbandLC.png')

