import numpy as np
from matplotlib import pyplot as plt
from pylab import *
from MyPendulumClass import PendulumDataSet

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

class Comparison:
# Comparison of PendulumDataSets
    def __init__(self, datads, simds):
        self.datads = datads
        self.simds  =  simds

    def summary(self):
# Print summary information for the PendulumDataSets
        print("Size of PendulumDataSet ",self.datads.ifile,'is',self.datads.clockticks().size)
        print("Size of PendulumDataSet ",self.simds.ifile,'is',self.simds.clockticks().size)

    def chisq1(self, err):
        fname='QvalueU'
        d = self.datads.QvalueU()
        s = self.simds.QvalueU()
        chisqval = 0.0
        N = 0
        for i in range(0, min(d.size,s.size)):
            chi = (d[i] - s[i])/err
            chisqval += chi**2
            N += 1
        print(fname,'chisq',chisqval,'ndof:',N,'chisq/ndof:',chisqval/float(N))
# maybe return a pair
        return chisqval

# Write the chi-squared method more generically using the function name 
# passed as a string to invoke the method
    def chisq(self, fname, err):
# fname is passed in as a string
        d = getattr(self.datads, fname)()  # Instead of self.datads.f()
        s = getattr(self.simds, fname)()
        chisqval = 0.0
        N = 0
        for i in range(0, min(d.size,s.size)):
            chi = (d[i] - s[i])/err
            chisqval += chi**2
            N += 1
        print(fname,'err=',err,'chisq',chisqval,'ndof:',N,'chisq/ndof:',chisqval/float(N))
# return a tuple with all the relevant information
        return fname,chisqval,N,chisqval/float(N)

# Now try to put all the matplotlib stuff here too.
    def plot1(self):
        d = self.datads
        s = self.simds
        plt.figure(10)
        errorbar(d.tvalueU(),d.tshadowU(),color='cyan',linewidth=2, label=r'Data tshadow (U)')
        errorbar(s.tvalueU(),s.tshadowU(),color='magenta',linewidth=2, label=r'Sim tshadow (U)')
        errorbar(d.tvalueD(),d.tshadowD(),color='blue',linewidth=2, label=r'Data tshadow (D)')
        errorbar(s.tvalueD(),s.tshadowD(),color='red',linewidth=2, label=r'Sim tshadow (D)')
        plt.legend()

    def plot1e(self,err):
# Include errors
        d = self.datads
        s = self.simds
        tshadowU = d.tshadowU()
        N = tshadowU.size
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(101)
        errorbar(d.tvalueU(),d.tshadowU(),color='cyan',linewidth=2, label=r'Data tshadow (U)')
        errorbar(d.tvalueU(),d.tshadowU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.tvalueU(),s.tshadowU(),color='magenta',linewidth=2, label=r'Sim tshadow (U)')
        errorbar(d.tvalueD(),d.tshadowD(),color='blue',linewidth=2, label=r'Data tshadow (D)')
        errorbar(d.tvalueD(),d.tshadowD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.tvalueD(),s.tshadowD(),color='red',linewidth=2, label=r'Sim tshadow (D)')
        title('Run 76')
        xlabel('Time [s]')
        ylabel('Shadow Time [s]')
        plt.grid(True)
        plt.legend()
        print('plot1e, N=',N)

    def plot2e(self,err):
# Include errors
        d = self.datads
        s = self.simds
        QvalueU = d.QvalueU()
        N = QvalueU.size
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(102)
        errorbar(d.tvalueU(),d.QvalueU(),color='cyan',linewidth=2, label=r'Data Q-value (U)')
        errorbar(d.tvalueU(),d.QvalueU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.tvalueU(),s.QvalueU(),color='magenta',linewidth=2, label=r'Sim Q-value (U)')
        errorbar(d.tvalueD(),d.QvalueD(),color='blue',linewidth=2, label=r'Data Q-value (D)')
        errorbar(d.tvalueD(),d.QvalueD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.tvalueD(),s.QvalueD(),color='red',linewidth=2, label=r'Sim Q-value (D)')
        title('Run 76')
        xlabel('Time [s]')
        ylabel('Q-value')
        plt.grid(True)
        plt.legend()
        print('plot2e, N=',N)

    def plot3e(self,err):
# Include errors
        d = self.datads
        s = self.simds
        tshadowU = d.tshadowU()
        N = tshadowU.size
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(103)
        errorbar(d.evalueU(),d.tshadowU(),color='cyan',linewidth=2, label=r'Data tshadow (U)')
        errorbar(d.evalueU(),d.tshadowU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueU(),s.tshadowU(),color='magenta',linewidth=2, label=r'Sim tshadow (U)')
        errorbar(d.evalueD(),d.tshadowD(),color='blue',linewidth=2, label=r'Data tshadow (D)')
        errorbar(d.evalueD(),d.tshadowD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueD(),s.tshadowD(),color='red',linewidth=2, label=r'Sim tshadow (D)')
        title('Run 76')
        xlabel('Event Number')
        ylabel('Shadow Time [s]')
        plt.grid(True)
        plt.legend()
        print('plot3e, N=',N)

    def plot4e(self,err):
# Include errors
        d = self.datads
        s = self.simds
        QvalueU = d.QvalueU()
        N = QvalueU.size
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(104)
        errorbar(d.evalueU(),d.QvalueU(),color='cyan',linewidth=2, label=r'Data Q-value (U)')
        errorbar(d.evalueU(),d.QvalueU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueU(),s.QvalueU(),color='magenta',linewidth=2, label=r'Sim Q-value (U)')
        errorbar(d.evalueD(),d.QvalueD(),color='blue',linewidth=2, label=r'Data Q-value (D)')
        errorbar(d.evalueD(),d.QvalueD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueD(),s.QvalueD(),color='red',linewidth=2, label=r'Sim Q-value (D)')
        title('Run 76')
        xlabel('Event Number')
        ylabel('Q-value')
        plt.grid(True)
        plt.legend()
        print('plot4e, N=',N)

    def plot5f(self,err):
# Include errors
        d = self.datads
        s = self.simds
        N = d.deltatDU().size
        print('plot5f, N=',N)
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(205)
        errorbar(d.evalueU(),d.deltatDU(),color='green',linewidth=2, label=r'Data D/U Time Difference')
        errorbar(d.evalueU(),d.deltatDU(),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueU(),s.deltatDU(),color='black',linewidth=2, label=r'Sim D/U Time Difference')
        title('Run 76')
        xlabel('Event Number')
        ylabel('D/U Time Difference [s]')
        plt.grid(True)
        plt.legend()

    def plot6f(self,err):
# Include errors
        d = self.datads
        s = self.simds
        N = d.tdiffU().size
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(206)
        errorbar(d.evalueUU(),d.tdiffU(),color='cyan',linewidth=2, label=r'Data Extremum Time Difference (U)')
        errorbar(d.evalueUU(),d.tdiffU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueUU(),s.tdiffU(),color='magenta',linewidth=2, label=r'Sim Extremum Time Difference (U)')
        title('Run 76')
        xlabel('Event Number')
        ylabel('Extremum Time Difference [s]')
        plt.grid(True)
        plt.legend()
        print('plot6f, N=',N)
