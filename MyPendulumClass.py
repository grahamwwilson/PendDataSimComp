import numpy as np

DTCLOCKD = 1.0/5.013290e6         # Same as simulation
DTCLOCKU = 1.00057665*DTCLOCKD    # Same as simulation
CLOCK_OFFSET_D = 137918728
DTUD = 19000 # 3.8 ms offset
CLOCK_OFFSET_U = int(CLOCK_OFFSET_D*DTCLOCKD/DTCLOCKU) - DTUD 

class PendulumDataSet:
# PendulumDataSet class defined using the input file
    def __init__(self, ifile):
        self.ifile = ifile

    def ttype(self):
# transition type (U/D)
        return np.genfromtxt(self.ifile, usecols=0, unpack=True, dtype=str)

    def event(self):
# event number
        return np.genfromtxt(self.ifile, usecols=1, unpack=True, dtype=int)

    def clockticks(self):
# clock ticks
        return np.genfromtxt(self.ifile, usecols=2, unpack=True, dtype=int)

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

