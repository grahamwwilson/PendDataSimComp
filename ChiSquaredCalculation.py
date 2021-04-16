from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from scipy import stats
import argparse

parser = argparse.ArgumentParser(description='Chi-Squared Calculations')
parser.add_argument("-f", "--file", default='SimDataFile-48.dat', help="input data file name")
args=parser.parse_args()
print('Found argument list: ')
print(args)
simfile=args.file

# Read data from Simulation data-file into numpy array format
sim_type = genfromtxt(simfile, usecols=0, unpack=True, dtype=str)
sim_event, sim_clockticks = genfromtxt(simfile, usecols=(1,2), unpack=True)

# Read data from reformatted data-file for Run 76 into numpy array format
datafile='DataSummaryFile-Run76-Shortened.dat'
data_type = genfromtxt(datafile, usecols=0, unpack=True, dtype=str)
data_event, data_clockticks = genfromtxt(datafile, usecols=(1,2),unpack=True)

# Set a clock offset value based on the mean times of events 2 and 3 corresponding to the time at equilibrium
# Current simulation has
#D 2 1438817 0.28776264316185113
#D 3 1556555 0.31131024968446336  => mean clock-ticks = 1497686
# Current data has
#D 2 139350543
#D 3 139482284                    => mean clock-ticks = 139416513.5
# difference is 137918727.5

#TODO this should probably be recalculated and allow for a potential 
# different clock offset for the two timers.

#CLOCK_OFFSET = 137918728  # value such that the mean time of the first two D events is correct 
#sim_clockticks = sim_clockticks + CLOCK_OFFSET   (Do this later differently for U and D).

# Add some plotting customization
SMALL_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 32
plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Make subsets for U and D events for data and simulation
# Set up two lists with the indices of the respective event types for simulation
sim_ulist=[]
sim_dlist=[]
sim_err=np.empty(sim_type.size, dtype=int)
for i in range(0, sim_type.size):
   sim_err[i] = 5
# Append to the appropriate list
   if str(sim_type[i])=='U':
      sim_ulist.append(i)
   if str(sim_type[i])=='D':
      sim_dlist.append(i)
# Set up two lists with the indices of the respective event types for data
data_ulist=[]
data_dlist=[]
data_err=np.empty(data_type.size, dtype=int)
for i in range(0, data_type.size):
   data_err[i] = 5
# Append to the appropriate list
   if str(data_type[i])=='U':
      data_ulist.append(i)
   if str(data_type[i])=='D':
      data_dlist.append(i)

DTCLOCKD = 1.0/5.013290e6         # Same as simulation
DTCLOCKU = 1.00057665*DTCLOCKD    # Same as simulation
CLOCK_OFFSET_D = 137918728
DTUD = 19000 # 3.8 ms offset
CLOCK_OFFSET_U = int(CLOCK_OFFSET_D*DTCLOCKD/DTCLOCKU) - DTUD 
sim_type_U = np.delete(sim_type, sim_dlist)
sim_event_U = np.delete(sim_event, sim_dlist)
sim_clockticks_U = np.delete(sim_clockticks, sim_dlist)
sim_err_U = np.delete(sim_err, sim_dlist)
sim_type_D = np.delete(sim_type, sim_ulist)
sim_event_D = np.delete(sim_event, sim_ulist)
sim_clockticks_D = np.delete(sim_clockticks, sim_ulist)
sim_err_D = np.delete(sim_err, sim_ulist)
# Now make clock offset for simulation
for i in range(0, sim_clockticks_D.size):
    sim_clockticks_D[i] = sim_clockticks_D[i] + CLOCK_OFFSET_D
    sim_clockticks_U[i] = sim_clockticks_U[i] + CLOCK_OFFSET_U

data_type_U = np.delete(data_type, data_dlist)
data_event_U = np.delete(data_event, data_dlist)
data_clockticks_U = np.delete(data_clockticks, data_dlist)
data_err_U = np.delete(data_err, data_dlist)
data_type_D = np.delete(data_type, data_ulist)
data_event_D = np.delete(data_event, data_ulist)
data_clockticks_D = np.delete(data_clockticks, data_ulist)
data_err_D = np.delete(data_err, data_ulist)

# Make np arrays with shadow times and averaged times
data_tshadow_D=np.empty(data_type_D.size//2)
data_tvalue_D=np.empty(data_type_D.size//2)
data_tshadow_D_event=np.empty(data_type_D.size//2)
data_tshadow_U=np.empty(data_type_U.size//2)
data_tvalue_U=np.empty(data_type_U.size//2)
data_tshadow_U_event=np.empty(data_type_U.size//2)
gen_err_D=np.empty(data_type_D.size//2)
gen_err_U=np.empty(data_type_U.size//2)
for i in range(0, data_type_D.size//2):
    data_tshadow_D[i] = DTCLOCKD*(data_clockticks_D[2*i+1] - data_clockticks_D[2*i])
    data_tvalue_D[i] = 0.5*DTCLOCKD*(data_clockticks_D[2*i] + data_clockticks_D[2*i+1])
    data_tshadow_D_event[i] = 0.5*(data_event_D[2*i] + data_event_D[2*i+1])
    gen_err_D[i] = 5.0*DTCLOCKD
for i in range(0, data_type_U.size//2):
    data_tshadow_U[i] = DTCLOCKU*(data_clockticks_U[2*i+1] - data_clockticks_U[2*i])
    data_tvalue_U[i] = 0.5*DTCLOCKU*(data_clockticks_U[2*i] + data_clockticks_U[2*i+1])
    data_tshadow_U_event[i] = 0.5*(data_event_U[2*i] + data_event_U[2*i+1])
    gen_err_U[i] = 5.0*DTCLOCKU
# Now arrays with periods
data_halfperiod_D=np.empty(data_type_D.size//2)
for i in range(0, data_type_D.size//2):
    if i==0: 
       data_halfperiod_D[0] = (data_tvalue_D[1] - data_tvalue_D[0])
    else:
       data_halfperiod_D[i] = (data_tvalue_D[i] - data_tvalue_D[i-1])
# In principle the U times do not measure periods ... - but still useful probably
data_halfperiod_U=np.empty(data_type_U.size//2)
for i in range(0, data_type_U.size//2):
    if i==0: 
       data_halfperiod_U[0] = (data_tvalue_U[1] - data_tvalue_U[0])
    else:
       data_halfperiod_U[i] = (data_tvalue_U[i] - data_tvalue_U[i-1])


# Make np arrays with shadow times and averaged times
sim_tshadow_D=np.empty(sim_type_D.size//2)
sim_tvalue_D=np.empty(sim_type_D.size//2)
sim_tshadow_D_event=np.empty(sim_type_D.size//2)
sim_tshadow_U=np.empty(sim_type_U.size//2)
sim_tvalue_U=np.empty(sim_type_U.size//2)
sim_tshadow_U_event=np.empty(sim_type_U.size//2)
for i in range(0, sim_type_D.size//2):
    sim_tshadow_D[i] = DTCLOCKD*(sim_clockticks_D[2*i+1] - sim_clockticks_D[2*i])
    sim_tvalue_D[i] = 0.5*DTCLOCKD*(sim_clockticks_D[2*i] + sim_clockticks_D[2*i+1])
    sim_tshadow_D_event[i] = 0.5*(sim_event_D[2*i] + sim_event_D[2*i+1])
for i in range(0, sim_type_U.size//2):
    sim_tshadow_U[i] = DTCLOCKU*(sim_clockticks_U[2*i+1] - sim_clockticks_U[2*i])
    sim_tvalue_U[i] = 0.5*DTCLOCKU*(sim_clockticks_U[2*i] + sim_clockticks_U[2*i+1])
    sim_tshadow_U_event[i] = 0.5*(sim_event_U[2*i] + sim_event_U[2*i+1])
# Now arrays with periods
sim_halfperiod_D=np.empty(sim_type_D.size//2)
for i in range(0, sim_type_D.size//2):
    if i==0: 
       sim_halfperiod_D[0] = (sim_tvalue_D[1] - sim_tvalue_D[0])
    else:
       sim_halfperiod_D[i] = (sim_tvalue_D[i] - sim_tvalue_D[i-1])

# In principle the U times do not measure periods ... - but still useful probably
sim_halfperiod_U=np.empty(sim_type_U.size//2)
for i in range(0, sim_type_U.size//2):
    if i==0: 
       sim_halfperiod_U[0] = (sim_tvalue_U[1] - sim_tvalue_U[0])
    else:
       sim_halfperiod_U[i] = (sim_tvalue_U[i] - sim_tvalue_U[i-1])

data_period_D=np.empty(data_type_D.size//4)
data_period_U=np.empty(data_type_D.size//4)
sim_period_D=np.empty(data_type_D.size//4)
sim_period_U=np.empty(data_type_D.size//4)
data_Aperiod=np.empty(data_type_D.size//4)
sim_Aperiod=np.empty(data_type_D.size//4)
data_Ahperiod_D=np.empty(data_type_D.size//4)
sim_Ahperiod_D=np.empty(data_type_D.size//4)
data_Ahperiod_U=np.empty(data_type_D.size//4)
sim_Ahperiod_U=np.empty(data_type_D.size//4)
gen_err_period=np.empty(data_type_D.size//4)
gen_err_Aperiod=np.empty(data_type_D.size//4)
data_event_period=np.empty(data_type_D.size//4)
# Also do D/U differences
data_deltat_DU=np.empty(data_type_D.size//2)
sim_deltat_DU=np.empty(data_type_D.size//2)
data_pdeltat_DU=np.empty(data_type_D.size//4)
sim_pdeltat_DU=np.empty(data_type_D.size//4)
for i in range(0, data_type_D.size//2):
    data_deltat_DU[i] = abs(data_tvalue_D[i] - data_tvalue_U[i])
    sim_deltat_DU[i] = abs(sim_tvalue_D[i] - sim_tvalue_U[i])
for i in range(0, data_type_D.size//4):
    data_period_D[i] = data_halfperiod_D[2*i] + data_halfperiod_D[2*i+1]
    data_period_U[i] = data_halfperiod_U[2*i] + data_halfperiod_U[2*i+1]
    data_Aperiod[i] = 100.0*(data_period_U[i] - data_period_D[i])/(data_period_U[i] + data_period_D[i])
    data_Ahperiod_D[i] = 100.0*(data_halfperiod_D[2*i] - data_halfperiod_D[2*i+1])/data_period_D[i]
    data_Ahperiod_U[i] = 100.0*(data_halfperiod_U[2*i] - data_halfperiod_U[2*i+1])/data_period_U[i]
    sim_period_D[i] = sim_halfperiod_D[2*i] + sim_halfperiod_D[2*i+1]
    sim_period_U[i] = sim_halfperiod_U[2*i] + sim_halfperiod_U[2*i+1]
    sim_Aperiod[i] = 100.0*(sim_period_U[i] - sim_period_D[i])/(sim_period_U[i] + sim_period_D[i])
    sim_Ahperiod_D[i] = 100.0*(sim_halfperiod_D[2*i] - sim_halfperiod_D[2*i+1])/sim_period_D[i]
    sim_Ahperiod_U[i] = 100.0*(sim_halfperiod_U[2*i] - sim_halfperiod_U[2*i+1])/sim_period_U[i]
    gen_err_period[i] = 5.0*DTCLOCKD
    gen_err_Aperiod[i] = gen_err_period[i]
    data_event_period[i] = 0.5*(data_tshadow_U_event[2*i] + data_tshadow_U_event[2*i+1])
    data_pdeltat_DU[i] = 0.5*(data_deltat_DU[2*i] + data_deltat_DU[2*i+1])
    sim_pdeltat_DU[i] = 0.5*(sim_deltat_DU[2*i] + sim_deltat_DU[2*i+1])

#print(data_tvalue_D-data_tvalue_U)
#print(sim_tvalue_D-sim_tvalue_U)

# Let's calculate some least-squares type quantity that we want to minimize
# for both shadow times.
# Here assign uncertainties of 0.2 ms.
chisqSU = 0.0
chisqSD = 0.0
for i in range(0, data_tshadow_U.size):
    chiSU = (data_tshadow_U[i] - sim_tshadow_U[i])/2.0e-4
    chiSD = (data_tshadow_D[i] - sim_tshadow_D[i])/2.0e-4
    chisqSU += chiSU**2
    chisqSD += chiSD**2

print('chisqSU ',chisqSU)
print('chisqSD ',chisqSD)
ndof=2*data_tshadow_U.size
print('chisqt ',chisqSU+chisqSD,'ndof = ',ndof,'chisq/ndof= ',(chisqSU+chisqSD)/ndof)

# Also compare the measured average time values for each transition.
# Here assign uncertainties of 2 ms.
chisqU = 0.0
chisqD = 0.0
for i in range(0, data_tvalue_U.size):
    chiU = (data_tvalue_U[i] - sim_tvalue_U[i])/2.0e-3
    chiD = (data_tvalue_D[i] - sim_tvalue_D[i])/2.0e-3
    chisqU += chiU**2
    chisqD += chiD**2

print('chisqU ',chisqU)
print('chisqD ',chisqD)
ndof=2*data_tvalue_U.size
print('chisqt ',chisqU+chisqD,'ndof = ',ndof,'chisq/ndof= ',(chisqU+chisqD)/ndof)

# Overall total
chisqtot = chisqSU+chisqSD+chisqU+chisqD
print('Result for simfile ',simfile)
print(' ------------------------------------------------------------------------')
print('chisqTOT ',chisqtot,'chisq/ndof= ',chisqtot/(2.0*float(ndof)))
print(' ------------------------------------------------------------------------')
print(' ')

# Plot the data with assigned errors
plt.figure(1)  
errorbar(data_tshadow_U_event,data_tshadow_U,color='cyan',linewidth=2, label=r'Data tshadow (U)')
errorbar(data_tshadow_U_event,data_tshadow_U,gen_err_U,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
errorbar(sim_tshadow_U_event,sim_tshadow_U,color='magenta',linewidth=2, label=r'Sim tshadow (U)')
errorbar(sim_tshadow_U_event,sim_tshadow_U,gen_err_U,fmt="o",color='magenta',solid_capstyle='projecting',capsize=0,markersize=4)
errorbar(data_tshadow_D_event,data_tshadow_D,color='blue',linewidth=2, label=r'Data tshadow (D)')
errorbar(data_tshadow_D_event,data_tshadow_D,gen_err_D,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
errorbar(sim_tshadow_D_event,sim_tshadow_D,color='red',linewidth=2, label=r'Sim tshadow (D)')
errorbar(sim_tshadow_D_event,sim_tshadow_D,gen_err_D,fmt="o",color='red',solid_capstyle='projecting',capsize=0,markersize=4)
title('Run 76')
xlabel(r'n-th measurement')
ylabel('Shadow time (nominal s)')
plt.grid(True)
plt.legend()

#plt.show()
