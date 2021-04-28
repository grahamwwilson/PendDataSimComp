from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from MyPendulumClass import PendulumDataSet
from MyComparisonClass import Comparison

# Rewrite code using two classes. One for the datasets with methods for value computations, 
# and one for plots/chisq comparisons.
# Code currently still needs some time offsets etc included.

NTOSKIP=104   # Number of transitions to skip at start of each PendulumDataSet
              # With this set to 104, we skip the first 13 oscillations with 13*8 = 104 measurements.

# Read data from reformatted data-file for Run 76 into numpy array format
#datafile='DataSummaryFile-Run76-Shortened.dat'
datafile='DataSummaryFile-Run76-LessShort.dat'
data = PendulumDataSet(datafile,NTOSKIP)
NDATA = data.size()

# Read data from Simulation data-file into numpy array format
#simfile='SimDataFile-55.dat'
simfile='SimDataFile-101.dat.gz'
sim = PendulumDataSet(simfile,NTOSKIP,NDATA)  # Only read at most NDATA rows from simulation
#sim = PendulumDataSet(simfile,NTOSKIP)       # In this case read all sim rows available

c = Comparison(data,sim)                      # Make a comparison object with the data and sim data-sets
c.summary()

# Make tuples with quantities for chi-squared calculations
function_tuple = 'QvalueU','QvalueD','tshadowU','tshadowD', 'tdiffU', 'deltatDU'  # function names
err_tuple      =       5.0,     5.0,     2.0e-4,    2.0e-4,  2.0e-3 ,    2.0e-3   # assigned errors
print(function_tuple)
print(err_tuple)
chisqtot=0.0
ntot = 0
for idx, fname in enumerate(function_tuple):
   vals = c.chisq(fname,err_tuple[idx])
   chisqtot += vals[1]                  # 2nd element of the returned tuple
   ntot += vals[2]                      # 3rd element of the returned tuple
print('Total chisq = ',chisqtot,'Ndof = ',ntot,'chisq/dof = ',chisqtot/float(ntot))

# Plot the corresponding figures using the comparison object methods
c.plot3e(2.0e-4)  # Shadow times for U/D using event number for x-axis
c.plot4e(5.0)     # Q values for U/D using event number for x-axis
c.plot5f(2.0e-3)  # D/U time difference using event number for x-axis
c.plot6f(2.0e-3)  # Extremum time difference (U) using event number for x-axis
plt.show()
