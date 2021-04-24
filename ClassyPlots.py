from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from scipy import stats
from MyPendulumClass import PendulumDataSet

# Rewrite code using a class that hides all the array manipulations etc.
# Code currently still needs some time offsets etc included.

NTOSKIP=104   # Number of transitions to skip at start of each PendulumDataSet
              # With this set to 104, we skip the first 13 oscillations (13*8 = 104) measurements.

# Read data from Simulation data-file into numpy array format
simfile='SimDataFile-55.dat'
sim = PendulumDataSet(simfile,NTOSKIP)

# Read data from reformatted data-file for Run 76 into numpy array format
#datafile='DataSummaryFile-Run76-Shortened.dat'
datafile='DataSummaryFile-Run76-LessShort.dat'
data = PendulumDataSet(datafile,NTOSKIP)

plt.figure(1)
errorbar(data.tvalueU(),data.tshadowU(),color='cyan',linewidth=2, label=r'Data tshadow (U)')
errorbar(sim.tvalueU(),sim.tshadowU(),color='magenta',linewidth=2, label=r'Sim tshadow (U)')
errorbar(data.tvalueD(),data.tshadowD(),color='blue',linewidth=2, label=r'Data tshadow (D)')
errorbar(sim.tvalueD(),sim.tshadowD(),color='red',linewidth=2, label=r'Sim tshadow (D)')
plt.legend()

plt.figure(2)
errorbar(data.tvalueU(),data.QvalueU(),color='cyan',linewidth=2, label=r'Data Qvalue (U)')
errorbar(sim.tvalueU(),sim.QvalueU(),color='magenta',linewidth=2, label=r'Sim Qvalue (U)')
errorbar(data.tvalueD(),data.QvalueD(),color='blue',linewidth=2, label=r'Data Qvalue (D)')
errorbar(sim.tvalueD(),sim.QvalueD(),color='red',linewidth=2, label=r'Sim Qvalue (D)')
plt.legend()

plt.figure(3)
errorbar(data.tvalueU(),data.halfperiodU(),color='cyan',linewidth=2, label=r'Data Half Period (U)')
errorbar(sim.tvalueU(),sim.halfperiodU(),color='magenta',linewidth=2, label=r'Sim Half Period (U)')
errorbar(data.tvalueD(),data.halfperiodD(),color='blue',linewidth=2, label=r'Data Half Period (D)')
errorbar(sim.tvalueD(),sim.halfperiodD(),color='red',linewidth=2, label=r'Sim Half Period (D)')
plt.legend()

plt.figure(4)
errorbar(data.tvalueD(),data.deltatDU(),color='green',linewidth=2, label=r'Data D/U Time Diff')
errorbar(sim.tvalueD(),sim.deltatDU(),color='black',linewidth=2, label=r'Sim D/U Time Diff')
plt.legend()

plt.figure(5)
errorbar(data.evalueU(),data.QvalueU(),color='cyan',linewidth=2, label=r'Data Qvalue (U)')
errorbar(data.evalueU(),data.QvalueU(),data.errQ(),fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
errorbar(sim.evalueU(),sim.QvalueU(),color='magenta',linewidth=2, label=r'Sim Qvalue (U)')
errorbar(sim.evalueU(),sim.QvalueU(),sim.errQ(),fmt="o",color='magenta',solid_capstyle='projecting',capsize=0,markersize=4)
errorbar(data.evalueD(),data.QvalueD(),color='blue',linewidth=2, label=r'Data Qvalue (D)')
errorbar(data.evalueD(),data.QvalueD(),data.errQ(),fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
errorbar(sim.evalueD(),sim.QvalueD(),color='red',linewidth=2, label=r'Sim Qvalue (D)')
errorbar(sim.evalueD(),sim.QvalueD(),sim.errQ(),fmt="o",color='red',solid_capstyle='projecting',capsize=0,markersize=4)
plt.legend()

plt.show()
