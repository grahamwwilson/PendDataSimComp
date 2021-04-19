from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from scipy import stats
from MyPendulumClass import PendulumDataSet

# Rewrite code using a class that hides all the array manipulations etc.
# Code currently still needs some time offsets etc included.

# Read data from Simulation data-file into numpy array format
simfile='SimDataFile-48.dat'
sim = PendulumDataSet(simfile)

# Read data from reformatted data-file for Run 76 into numpy array format
datafile='DataSummaryFile-Run76-Shortened.dat'
data = PendulumDataSet(datafile)

print('sim.tshadowD():',sim.tshadowD())
print('data.tshadowD():',data.tshadowD())
print('sim.tvalueD():',sim.tvalueD())
print('data.tvalueD():',data.tvalueD())
print('sim.QvalueD():',sim.QvalueD())
print('data.QvalueD():',data.QvalueD())

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

plt.show()
