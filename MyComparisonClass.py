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

    def plot4g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.QvalueU().size
        Ns = s.QvalueU().size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(304)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.QvalueU(),color='cyan',linewidth=2, label=r'Data Q-Value (U)')
        errorbar(d.genx(Nd),d.QvalueU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.QvalueU(),color='magenta',linewidth=2, label=r' Sim Q-Value (U)')
        errorbar(d.genx(Nd),d.QvalueD(),color='blue',linewidth=2, label=r'Data Q-Value (D)')
        errorbar(d.genx(Nd),d.QvalueD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.QvalueD(),color='red',linewidth=2, label=r' Sim Q-Value (D)')
        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Q-Value')
        plt.grid(True)
        plt.legend()
        print('plot4g, Nd=',Nd,'Ns=',Ns)

    def plot5f(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.deltatDU().size
        Ns = s.deltatDU().size

        print('plot5f, Nd=',Nd,'Ns=',Ns)
        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(205)
        errorbar(d.genx(Nd),d.deltatDU(),color='green',linewidth=2, label=r'Data D/U Time Difference')
        errorbar(d.genx(Nd),d.deltatDU(),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.deltatDU(),color='black',linewidth=2, label=r'Sim D/U Time Difference')
        title('Run 76')
        xlabel('Measurement Number')
        ylabel('D/U Time Difference [s]')
        plt.grid(True)
        plt.legend()

    def plot6g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.tdiffU().size
        Ns = s.tdiffU().size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(306)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.tdiffU(),color='cyan',linewidth=2, label=r'Data Extremum Time Difference (U)')
        errorbar(d.genx(Nd),d.tdiffU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.tdiffU(),color='magenta',linewidth=2, label=r' Sim Extremum Time Difference (U)')
        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Extremum Time Difference [s]')
        plt.grid(True)
        plt.legend()
        print('plot6g, Nd=',Nd,'Ns=',Ns)

    def plot7f(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.AhperiodU().size
        Ns = s.AhperiodU().size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(207)
        errorbar(d.genx(Nd),d.AhperiodU(),color='cyan',linewidth=2, label=r'Data Half-Period Asymmetry (U)')
        errorbar(d.genx(Nd),d.AhperiodU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.AhperiodU(),color='magenta',linewidth=2, label=r'Sim Half-Period Asymmetry (U)')
        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Half-Period Asymmetry [%]')
        plt.grid(True)
        plt.legend()
        print('plot7f, Nd=',Nd,'Ns=',Ns)

    def plot8f(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.AhperiodD().size
        Ns = s.AhperiodD().size

        errD = np.empty(Nd)
        for i in range(0,Nd):
            errD[i] = err 
        plt.figure(208)
        errorbar(d.genx(Nd),d.AhperiodD(),color='blue',linewidth=2, label=r'Data Half-Period Asymmetry (D)')
        errorbar(d.genx(Nd),d.AhperiodD(),errD,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.AhperiodD(),color='red',linewidth=2, label=r'Sim Half-Period Asymmetry (D)')
        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Half-Period Asymmetry [%]')
        plt.grid(True)
        plt.legend()
        print('plot8f, Nd=',Nd,'Ns=',Ns)

    def plot9e(self,err):
# Include errors
        d = self.datads
        s = self.simds
        periodU = d.periodU()
        N = periodU.size
        errU = np.empty(N)
        for i in range(0,N):
            errU[i] = err 
        plt.figure(109)
        errorbar(d.evalueUU(),d.periodU(),color='cyan',linewidth=2, label=r'Data Period (U)')
        errorbar(d.evalueUU(),d.periodU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueUU(),s.periodU(),color='magenta',linewidth=2, label=r' Sim Period (U)')
        errorbar(d.evalueDD(),d.periodD(),color='blue',linewidth=2, label=r'Data Period (D)')
        errorbar(d.evalueDD(),d.periodD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.evalueDD(),s.periodD(),color='red',linewidth=2, label=r' Sim Period (D)')
        title('Run 76')
        xlabel('Event Number')
        ylabel('Period [s]')
        plt.grid(True)
        plt.legend()
        print('plot9e, N=',N)

    def plot9g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.periodU().size
        Ns = s.periodU().size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(309)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.periodU(),color='cyan',linewidth=2, label=r'Data Period (U)')
        errorbar(d.genx(Nd),d.periodU(),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.periodU(),color='magenta',linewidth=2, label=r' Sim Period (U)')
        errorbar(d.genx(Nd),d.periodD(),color='blue',linewidth=2, label=r'Data Period (D)')
        errorbar(d.genx(Nd),d.periodD(),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.periodD(),color='red',linewidth=2, label=r' Sim Period (D)')
        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Period [s]')
        plt.grid(True)
        plt.legend()
        print('plot9g, Nd=',Nd,'Ns=',Ns)

    def plot10g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.PeriodU(0).size
        Ns = s.PeriodU(0).size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(310)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.PeriodU(0),color='cyan',linewidth=2, label=r'Data Period (U0)')
        errorbar(d.genx(Nd),d.PeriodU(0),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodU(0),color='magenta',linewidth=2, label=r' Sim Period (U0)')

        errorbar(d.genx(Nd),d.PeriodU(1),color='blue',linewidth=2, label=r'Data Period (U1)')
        errorbar(d.genx(Nd),d.PeriodU(1),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodU(1),color='red',linewidth=2, label=r' Sim Period (U1)')

        errorbar(d.genx(Nd),d.PeriodU(2),color='green',linewidth=2, label=r'Data Period (U2)')
        errorbar(d.genx(Nd),d.PeriodU(2),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodU(2),color='black',linewidth=2, label=r' Sim Period (U2)')

        errorbar(d.genx(Nd),d.PeriodU(3),color='orange',linewidth=2, label=r'Data Period (U3)')
        errorbar(d.genx(Nd),d.PeriodU(3),errU,fmt="o",color='orange',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodU(3),color='grey',linewidth=2, label=r' Sim Period (U3)')

        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Period [s]')
        plt.grid(True)
        plt.legend()
        print('plot10g, Nd=',Nd,'Ns=',Ns)

    def plot11g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.PeriodD(0).size
        Ns = s.PeriodD(0).size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(311)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.PeriodD(0),color='cyan',linewidth=2, label=r'Data Period (D0)')
        errorbar(d.genx(Nd),d.PeriodD(0),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodD(0),color='magenta',linewidth=2, label=r' Sim Period (D0)')

        errorbar(d.genx(Nd),d.PeriodD(1),color='blue',linewidth=2, label=r'Data Period (D1)')
        errorbar(d.genx(Nd),d.PeriodD(1),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodD(1),color='red',linewidth=2, label=r' Sim Period (D1)')

        errorbar(d.genx(Nd),d.PeriodD(2),color='green',linewidth=2, label=r'Data Period (D2)')
        errorbar(d.genx(Nd),d.PeriodD(2),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodD(2),color='black',linewidth=2, label=r' Sim Period (D2)')

        errorbar(d.genx(Nd),d.PeriodD(3),color='orange',linewidth=2, label=r'Data Period (D3)')
        errorbar(d.genx(Nd),d.PeriodD(3),errU,fmt="o",color='orange',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PeriodD(3),color='grey',linewidth=2, label=r' Sim Period (D3)')

        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Period [s]')
        plt.grid(True)
        plt.legend()
        print('plot11g, Nd=',Nd,'Ns=',Ns)

    def plot12g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.PPeriodD(0).size
        Ns = s.PPeriodD(0).size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(312)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.PPeriodD(0),color='cyan',linewidth=2, label=r'Data PPeriod (D0)')
        errorbar(d.genx(Nd),d.PPeriodD(0),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PPeriodD(0),color='magenta',linewidth=2, label=r' Sim PPeriod (D0)')

        errorbar(d.genx(Nd),d.PPeriodD(1),color='blue',linewidth=2, label=r'Data PPeriod (D1)')
        errorbar(d.genx(Nd),d.PPeriodD(1),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PPeriodD(1),color='red',linewidth=2, label=r' Sim PPeriod (D1)')

        errorbar(d.genx(Nd),d.PPeriodU(0),color='green',linewidth=2, label=r'Data PPeriod (U0)')
        errorbar(d.genx(Nd),d.PPeriodU(0),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PPeriodU(0),color='black',linewidth=2, label=r' Sim PPeriod (U0)')

        errorbar(d.genx(Nd),d.PPeriodU(1),color='orange',linewidth=2, label=r'Data PPeriod (U1)')
        errorbar(d.genx(Nd),d.PPeriodU(1),errU,fmt="o",color='orange',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.PPeriodU(1),color='grey',linewidth=2, label=r' Sim PPeriod (U1)')

        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Period [s]')
        plt.grid(True)
        plt.legend()
        print('plot12g, Nd=',Nd,'Ns=',Ns)

    def plot13h(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.APeriodU(0,1).size
        Ns = s.APeriodU(0,1).size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(313)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.APeriodU(0,1),color='cyan',linewidth=2, label=r'Data APeriod (U-01)')
        errorbar(d.genx(Nd),d.APeriodU(0,1),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodU(0,1),color='magenta',linewidth=2, label=r' Sim APeriod (U-01)')

        errorbar(d.genx(Nd),d.APeriodU(0,2),color='blue',linewidth=2, label=r'Data APeriod (U-02)')
        errorbar(d.genx(Nd),d.APeriodU(0,2),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodU(0,2),color='red',linewidth=2, label=r' Sim APeriod (U-02)')

        errorbar(d.genx(Nd),d.APeriodU(0,3),color='green',linewidth=2, label=r'Data APeriod (U-03)')
        errorbar(d.genx(Nd),d.APeriodU(0,3),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodU(0,3),color='black',linewidth=2, label=r' Sim APeriod (U-03)')

        errorbar(d.genx(Nd),d.APeriodU(1,2),color='orange',linewidth=2, label=r'Data APeriod (U-12)')
        errorbar(d.genx(Nd),d.APeriodU(1,2),errU,fmt="o",color='orange',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodU(1,2),color='grey',linewidth=2, label=r' Sim APeriod (U-12)')

        errorbar(d.genx(Nd),d.APeriodU(1,3),color='olive',linewidth=2, label=r'Data APeriod (U-13)')
        errorbar(d.genx(Nd),d.APeriodU(1,3),errU,fmt="o",color='olive',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodU(1,3),color='yellow',linewidth=2, label=r' Sim APeriod (U-13)')

        errorbar(d.genx(Nd),d.APeriodU(2,3),color='brown',linewidth=2, label=r'Data APeriod (U-23)')
        errorbar(d.genx(Nd),d.APeriodU(2,3),errU,fmt="o",color='brown',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodU(2,3),color='purple',linewidth=2, label=r' Sim APeriod (U-23)')

        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Period Asymmetry [%]')
        plt.grid(True)
        plt.legend()
        print('plot13h, Nd=',Nd,'Ns=',Ns)

    def plot14h(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.APeriodD(0,1).size
        Ns = s.APeriodD(0,1).size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(314)
# use generic x-axes based on size of array
        errorbar(d.genx(Nd),d.APeriodD(0,1),color='cyan',linewidth=2, label=r'Data APeriod (D-01)')
        errorbar(d.genx(Nd),d.APeriodD(0,1),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodD(0,1),color='magenta',linewidth=2, label=r' Sim APeriod (D-01)')

        errorbar(d.genx(Nd),d.APeriodD(0,2),color='blue',linewidth=2, label=r'Data APeriod (D-02)')
        errorbar(d.genx(Nd),d.APeriodD(0,2),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodD(0,2),color='red',linewidth=2, label=r' Sim APeriod (D-02)')

        errorbar(d.genx(Nd),d.APeriodD(0,3),color='green',linewidth=2, label=r'Data APeriod (D-03)')
        errorbar(d.genx(Nd),d.APeriodD(0,3),errU,fmt="o",color='green',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodD(0,3),color='black',linewidth=2, label=r' Sim APeriod (D-03)')

        errorbar(d.genx(Nd),d.APeriodD(1,2),color='orange',linewidth=2, label=r'Data APeriod (D-12)')
        errorbar(d.genx(Nd),d.APeriodD(1,2),errU,fmt="o",color='orange',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodD(1,2),color='grey',linewidth=2, label=r' Sim APeriod (D-12)')

        errorbar(d.genx(Nd),d.APeriodD(1,3),color='olive',linewidth=2, label=r'Data APeriod (D-13)')
        errorbar(d.genx(Nd),d.APeriodD(1,3),errU,fmt="o",color='olive',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodD(1,3),color='yellow',linewidth=2, label=r' Sim APeriod (D-13)')

        errorbar(d.genx(Nd),d.APeriodD(2,3),color='brown',linewidth=2, label=r'Data APeriod (D-23)')
        errorbar(d.genx(Nd),d.APeriodD(2,3),errU,fmt="o",color='brown',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodD(2,3),color='purple',linewidth=2, label=r' Sim APeriod (D-23)')

        title('Run 76')
        xlabel('Measurement Number')
        ylabel('Period Asymmetry [%]')
        plt.grid(True)
        plt.legend()
        print('plot14h, Nd=',Nd,'Ns=',Ns)

    def plot15g(self,err):
# Include errors
        d = self.datads
        s = self.simds
        Nd = d.APeriodDU(0).size
        Ns = s.APeriodDU(0).size

        errU = np.empty(Nd)
        for i in range(0,Nd):
            errU[i] = err 
        plt.figure(315)
# use generic x-axes based on size of array
 
        errorbar(d.genx(Nd),d.APeriodDU(1),color='cyan',linewidth=2, label=r'Data D/U Period Asymmetry 1')
        errorbar(d.genx(Nd),d.APeriodDU(1),errU,fmt="o",color='cyan',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodDU(1),color='magenta',linewidth=2, label=r' Sim D/U Period Asymmetry 1')
        errorbar(d.genx(Nd),d.APeriodDU(0),color='blue',linewidth=2, label=r'Data D/U Period Asymmetry 0')
        errorbar(d.genx(Nd),d.APeriodDU(0),errU,fmt="o",color='blue',solid_capstyle='projecting',capsize=0,markersize=4)
        errorbar(s.genx(Ns),s.APeriodDU(0),color='red',linewidth=2, label=r' Sim D/U Period Asymmetry 0')

        title('Run 76')
        xlabel('Measurement Number')
        ylabel('D/U Period Asymmetry [%]')
        plt.grid(True)
        plt.legend()
        print('plot15g, Nd=',Nd,'Ns=',Ns)

