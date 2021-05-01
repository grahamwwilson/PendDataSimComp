import numpy as np

DTCLOCKD = 1.0/5.013290e6         # Same as simulation
#DTCLOCKU = 1.00057665*DTCLOCKD    # Same as simulation
DTCLOCKU = 1.0005862*DTCLOCKD    # Same as simulation
#DTCLOCKU=1.0*DTCLOCKD
CLOCK_OFFSET_D = 137918728
#DTUD = 19000 # 3.8 ms offset
#DTUD = 27400
DTUD = 20770
CLOCK_OFFSET_U = int(CLOCK_OFFSET_D*DTCLOCKD/DTCLOCKU) - DTUD

# Notes. Tidy up time definitions to all be based on tD and tU
# where we can put all frequency and offset corrections

class PendulumDataSet:
# PendulumDataSet class defined using the input file
    def __init__(self, ifile, datatype, ntoskip=0, max_rows=None):
        self.ifile = ifile
        self.datatype = datatype   # Specify whether 'data' or 'sim'
        self.ntoskip = ntoskip     # Number of lines to skip at the start of the file (to avoid noisy data)
        self.max_rows = max_rows   # Maximum number of rows to read (after skipping the number specified)

    def ttype(self):
# transition type (U/D)
        return np.genfromtxt(self.ifile, usecols=0, skip_header=self.ntoskip+1, max_rows=self.max_rows, unpack=True, dtype=str)

    def event(self):
# event number
        return np.genfromtxt(self.ifile, usecols=1, skip_header=self.ntoskip+1, max_rows=self.max_rows, unpack=True, dtype=int)

    def clockticks(self):
# clock ticks
        return np.genfromtxt(self.ifile, usecols=2, skip_header=self.ntoskip+1, max_rows=self.max_rows, unpack=True, dtype=int)

    def summary(self):
# Print summary information for this PendulumDataSet
        print("Size of PendulumDataSet ",self.ifile,'is',self.clockticks().size)

    def size(self):
        return int(self.clockticks().size)

# Set up list with indices of the 'U' event types
    def ulist(self):
        list=[]
        ttype = self.ttype()
        for i in range(0, ttype.size):
# Append to the appropriate list
            if str(ttype[i])=='U':
               list.append(i)
        return list

# Set up lists with indices of the 'D' event types
    def dlist(self):
        list=[]
        ttype = self.ttype()
        for i in range(0, ttype.size):
# Append to the appropriate list
            if str(ttype[i])=='D':
               list.append(i)
        return list

# Make U, D versions by removing the unwanted ones from each
    def ttypeU(self):
        return np.delete(self.ttype(),self.dlist())

    def ttypeD(self):
        return np.delete(self.ttype(),self.ulist())

    def eventU(self):
        return np.delete(self.event(),self.dlist())

    def eventD(self):
        return np.delete(self.event(),self.ulist())

    def clockticksU(self):
        return np.delete(self.clockticks(),self.dlist())

    def clockticksD(self):
        return np.delete(self.clockticks(),self.ulist())

# Now finally we can define the methods that return the arrays of interest
    def tD(self):
        N = self.ttypeD().size
        tD = np.empty(N)
        clockticksD = self.clockticksD()
#        for i in range(0,N):
#            if self.datatype=='sim':
#               tD[i] = DTCLOCKD*(clockticksD[i] + CLOCK_OFFSET_D)
#            else:
#               tD[i] = DTCLOCKD*clockticksD[i]
        for i in range(0,N):
            if self.datatype=='data':
               tD[i] = DTCLOCKD*(clockticksD[i] - CLOCK_OFFSET_D)
            else:
               tD[i] = DTCLOCKD*clockticksD[i]
        return tD

    def tU(self):
        N = self.ttypeU().size
        tU = np.empty(N)
        clockticksU = self.clockticksU()
#        for i in range(0,N):
#            if self.datatype=='sim':
#               tU[i] = DTCLOCKU*(clockticksU[i] + CLOCK_OFFSET_U)
#            else:
#               tU[i] = DTCLOCKU*clockticksU[i]
        for i in range(0,N):
            if self.datatype=='data':
               tU[i] = DTCLOCKU*(clockticksU[i] - CLOCK_OFFSET_U)
            else:
               tU[i] = DTCLOCKU*clockticksU[i]
        return tU

    def tvalueD(self):
        N = self.ttypeD().size//2
        tvalueD = np.empty(N)
        tD = self.tD()
        for i in range(0,N):
            tvalueD[i] = 0.5*(tD[2*i+1] + tD[2*i])
        return tvalueD

    def tvalueU(self):
        N = self.ttypeU().size//2
        tvalueU = np.empty(N)
        tU = self.tU()
        for i in range(0,N):
            tvalueU[i] = 0.5*(tU[2*i+1] + tU[2*i])
        return tvalueU

    def tshadowD(self):
        N = self.ttypeD().size//2
        tshadowD = np.empty(N)
        tD = self.tD()
        for i in range(0,N):
            tshadowD[i] = tD[2*i+1] - tD[2*i]
        return tshadowD

    def tshadowU(self):
        N = self.ttypeU().size//2
        tshadowU = np.empty(N)
        tU = self.tU()
        for i in range(0,N):
            tshadowU[i] = tU[2*i+1] - tU[2*i]
        return tshadowU

    def PeriodD(self,iphase):
# Need iphase to be 0, 1, 2 or 3.
        n = self.tD().size//4
        tD = self.tD()
        PeriodD = np.empty(n-1)
        for i in range(0,n-1):
            PeriodD[i] = tD[4*(i+1)+iphase] - tD[4*i+iphase]
        return PeriodD

    def PeriodU(self,iphase):
# Need iphase to be 0, 1, 2 or 3.
        n = self.tU().size//4
        tU = self.tU()
        PeriodU = np.empty(n-1)
        for i in range(0,n-1):
            PeriodU[i] = tU[4*(i+1)+iphase] - tU[4*i+iphase]
        return PeriodU

    def APeriodU(self,ip,jp):
# Period Asymmetry (%) for phase ip wrt jp
        N = self.PeriodU(0).size
        PeriodUi = self.PeriodU(ip)
        PeriodUj = self.PeriodU(jp)
        APeriodU = np.empty(N)
        for i in range(0,N):
            APeriodU[i] = 100.0*(PeriodUi[i] - PeriodUj[i])/(PeriodUi[i] + PeriodUj[i])
        return APeriodU

    def APeriodD(self,ip,jp):
# Period Asymmetry (%) for phase ip wrt jp
        N = self.PeriodD(0).size
        PeriodDi = self.PeriodD(ip)
        PeriodDj = self.PeriodD(jp)
        APeriodD = np.empty(N)
        for i in range(0,N):
            APeriodD[i] = 100.0*(PeriodDi[i] - PeriodDj[i])/(PeriodDi[i] + PeriodDj[i])
        return APeriodD

#NB Can likely define a period-based Q value analog
#   or maybe a logarithmic decrement type quantity.

    def PPeriodU(self,i):
        if i==0:
           i0 = 1
           i1 = 2
        else:
           i0 = 0
           i1 = 3
        P0 = self.PeriodU(i0)
        P1 = self.PeriodU(i1)
        N = self.PeriodU(0).size
        PPeriodU = np.empty(N)
        for i in range(0,N):
            PPeriodU[i] = 0.5*(P0[i] + P1[i])
        return PPeriodU

    def PPeriodD(self,i):
        if i==0:
           i0 = 1
           i1 = 2
        else:
           i0 = 0
           i1 = 3
        P0 = self.PeriodD(i0)
        P1 = self.PeriodD(i1)
        N = self.PeriodD(0).size
        PPeriodD = np.empty(N)
        for i in range(0,N):
            PPeriodD[i] = 0.5*(P0[i] + P1[i])
        return PPeriodD

    def APeriodDU(self,i):
        TU = self.PPeriodU(i)
        TD = self.PPeriodD(i)
        N = TU.size
        APeriodDU = np.empty(N)
        for i in range(0,N):
            APeriodDU[i] = 100.0*(TU[i] - TD[i])/(TU[i] + TD[i])
        return APeriodDU

    def evalueD(self):
        N = self.ttypeD().size//2
        evalueD = np.empty(N)
        eventD = self.eventD()
        for i in range(0,N):
            evalueD[i] = 0.5*(eventD[2*i+1] + eventD[2*i])
        return evalueD

    def evalueDD(self):
        N = self.ttypeD().size//4
        evalueDD = np.empty(N)
        eventD = self.eventD()
        for i in range(0,N):
            evalueDD[i] = 0.25*(eventD[2*i+1] + eventD[2*i]+eventD[2*i+3] + eventD[2*i+2])
        return evalueDD

    def evalueU(self):
        N = self.ttypeU().size//2
        evalueU = np.empty(N)
        eventU = self.eventU()
        for i in range(0,N):
            evalueU[i] = 0.5*(eventU[2*i+1] + eventU[2*i])
        return evalueU

    def evalueUU(self):
        N = self.ttypeU().size//4
        evalueUU = np.empty(N)
        eventU = self.eventU()
        for i in range(0,N):
            evalueUU[i] = 0.25*(eventU[2*i+1] + eventU[2*i]+eventU[2*i+3] + eventU[2*i+2])
        return evalueUU

    def errQ(self):
        N = self.ttypeU().size//2
        errQ = np.empty(N)
        for i in range(0,N):
            errQ[i] = 5.0
        return errQ

    def QvalueD(self):  
        tshadowD = self.tshadowD()
        N = tshadowD.size
        QvalueD = np.empty(N-2)
        for i in range(0, N-2):
            E0 = 1.0/tshadowD[i]**2
            E1 = 1.0/tshadowD[i+1]**2
            E2 = 1.0/tshadowD[i+2]**2
            QvalueD[i] = 2.0*np.pi*E1/(E0-E2)
        return QvalueD

    def QvalueU(self):
        tshadowU = self.tshadowU()
        N = tshadowU.size
        QvalueU = np.empty(N-2)
        for i in range(0, N-2):
            E0 = 1.0/tshadowU[i]**2
            E1 = 1.0/tshadowU[i+1]**2
            E2 = 1.0/tshadowU[i+2]**2
            QvalueU[i] = 2.0*np.pi*E1/(E0-E2)
        return QvalueU

    def halfperiodD(self):
        tvalueD = self.tvalueD()
        N = tvalueD.size - 1
        halfperiodD = np.empty(N)
        for i in range(0,N):
            halfperiodD[i] = tvalueD[i+1] - tvalueD[i]
        return halfperiodD

    def tdiffU(self):         # (t4 - t3)U
# This should tend towards zero when the upstream laser passage 
# is about to be incomplete 
        N = (self.ttypeU().size//4) - 1
        tdiffU = np.empty(N)
        tU = self.tU()
        for i in range(0,N):
            tdiffU[i] = tU[4*i+4] - tU[4*i+3]
        return tdiffU

    def halfperiodU(self):
        tvalueU = self.tvalueU()
        N = tvalueU.size - 1
        halfperiodU = np.empty(N)
        for i in range(0,N):
            halfperiodU[i] = tvalueU[i+1] - tvalueU[i]
        return halfperiodU

    def deltatDU(self):
        tvalueU = self.tvalueU()
        tvalueD = self.tvalueD()
        N = tvalueD.size
        deltatDU = np.empty(N)
        for i in range(0,N):
            deltatDU[i] = abs(tvalueD[i] - tvalueU[i])
#            deltatDU[i] = tvalueD[i] - tvalueU[i]
        return deltatDU

#Casey  ###
    def periodU(self):
        halfperiodU = self.halfperiodU()
        N = halfperiodU.size//2
        periodU = np.empty(N)
        for i in range (0,N):
            periodU[i] = halfperiodU[2*i] + halfperiodU[2*i+1]
        return periodU

    def AhperiodU(self):
        halfperiodU = self.halfperiodU()
        periodU = self.periodU()
        N = halfperiodU.size//2
        AhperiodU = np.empty(N)
        for i in range (0,N):
            AhperiodU[i] = 100.0*(halfperiodU[2*i+1] - halfperiodU[2*i])/(halfperiodU[2*i+1] + halfperiodU[2*i])
        return AhperiodU

    def periodD(self):
        halfperiodD = self.halfperiodD()
        N = halfperiodD.size//2
        periodD = np.empty(N)
        for i in range (0,N):
            periodD[i] = halfperiodD[2*i] + halfperiodD[2*i+1]
        return periodD

    def AhperiodD(self):
        halfperiodD = self.halfperiodD()
        periodD = self.periodD()
        N = halfperiodD.size//2
        AhperiodD = np.empty(N)
        for i in range (0,N):
            AhperiodD[i] = 100.0*(halfperiodD[2*i+1] - halfperiodD[2*i])/(halfperiodD[2*i+1] + halfperiodD[2*i])
        return AhperiodD
#

    def checksizes(self):
# Print summary information of the sizes of all arrays
         print('Size of event is ',self.event().size)
         print('Size of halfperiodU is ',self.halfperiodU().size)
         print('Size of periodU is ',self.periodU().size)
         print('Size of AhperiodU is ',self.AhperiodU().size)
         print('Size of evalueUU is ',self.evalueUU().size)
         print('Size of periodD is ',self.periodD().size)
         print('Size of AhperiodD is ',self.AhperiodD().size)
         print('Size of evalueDD is ',self.evalueDD().size)
   
    def genx(self,N):
# Make generic numpy array for x-axes ranging from [0, N-1] for use in plots
        genx = np.empty(N)
        for i in range (0,N):
            genx[i] = i
        return genx

