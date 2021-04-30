import numpy as np

DTCLOCKD = 1.0/5.013290e6         # Same as simulation
DTCLOCKU = 1.00057665*DTCLOCKD    # Same as simulation
CLOCK_OFFSET_D = 137918728
DTUD = 19000 # 3.8 ms offset
CLOCK_OFFSET_U = int(CLOCK_OFFSET_D*DTCLOCKD/DTCLOCKU) - DTUD 

class PendulumDataSet:
# PendulumDataSet class defined using the input file
    def __init__(self, ifile, ntoskip=0, max_rows=None):
        self.ifile = ifile
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

# Now finally we can define the methods that return the arrays that 
# we're more interested in
    def tshadowD(self):
        N = self.ttypeD().size//2
        tshadowD = np.empty(N)
        clockticksD = self.clockticksD()
        for i in range(0,N):
            tshadowD[i] = DTCLOCKD*(clockticksD[2*i+1] - clockticksD[2*i])
        return tshadowD

    def tvalueD(self):
        N = self.ttypeD().size//2
        tvalueD = np.empty(N)
        clockticksD = self.clockticksD()
        for i in range(0,N):
            tvalueD[i] = 0.5*DTCLOCKD*(clockticksD[2*i+1] + clockticksD[2*i])
        return tvalueD

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
        QvalueD = np.empty(N)
        QvalueD[0] = 400.0
        QvalueD[N-1] = 400.0
        for i in range(1, N-1):
            E0 = 1.0/tshadowD[i-1]**2
            E1 = 1.0/tshadowD[i]**2
            E2 = 1.0/tshadowD[i+1]**2
            QvalueD[i] = 2.0*np.pi*E1/(E0-E2)
        return QvalueD

    def halfperiodD(self):
        tvalueD = self.tvalueD()
        N = tvalueD.size
        halfperiodD = np.empty(N)
        halfperiodD[0] = tvalueD[1]-tvalueD[0]
        for i in range(1,N):
            halfperiodD[i] = tvalueD[i] - tvalueD[i-1]
        return halfperiodD

    def tshadowU(self):
        N = self.ttypeU().size//2
        tshadowU = np.empty(N)
        clockticksU = self.clockticksU()
        for i in range(0,N):
            tshadowU[i] = DTCLOCKU*(clockticksU[2*i+1] - clockticksU[2*i])
        return tshadowU

    def tdiffU(self):               # time between
        N = self.ttypeU().size//4
        tdiffU = np.empty(N)
        clockticksU = self.clockticksU()
        for i in range(0,N-1):
            tdiffU[i] = DTCLOCKU*(clockticksU[4*i+4] - clockticksU[4*i+3])
        tdiffU[N-1] = 0.0
        return tdiffU

    def tvalueU(self):
        N = self.ttypeU().size//2
        tvalueU = np.empty(N)
        clockticksU = self.clockticksU()
        for i in range(0,N):
            tvalueU[i] = 0.5*DTCLOCKU*(clockticksU[2*i+1] + clockticksU[2*i])
        return tvalueU

    def QvalueU(self):
        tshadowU = self.tshadowU()
        N = tshadowU.size
        QvalueU = np.empty(N)
        QvalueU[0] = 400.0
        QvalueU[N-1] = 400.0
        for i in range(1, N-1):
            E0 = 1.0/tshadowU[i-1]**2
            E1 = 1.0/tshadowU[i]**2
            E2 = 1.0/tshadowU[i+1]**2
            QvalueU[i] = 2.0*np.pi*E1/(E0-E2)
        return QvalueU

    def halfperiodU(self):
        tvalueU = self.tvalueU()
        N = tvalueU.size
        halfperiodU = np.empty(N)
        halfperiodU[0] = tvalueU[1]-tvalueU[0]
        for i in range(1,N):
            halfperiodU[i] = tvalueU[i] - tvalueU[i-1]
        return halfperiodU

    def deltatDU(self):
        tvalueU = self.tvalueU()
        tvalueD = self.tvalueD()
        N = tvalueD.size
        deltatDU = np.empty(N)
        for i in range(0,N):
            deltatDU[i] = abs(tvalueD[i] - tvalueU[i])
        return deltatDU

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

    def checksizes(self):
# Print summary information of the sizes of all arrays
         print('Size of event is ',self.event().size)
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



