from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from NewPendulumClass import PendulumDataSet
#from MyComparisonClass import Comparison

NTOSKIP=30000-2   # Number of transitions to skip at start of each PendulumDataSet

datafile='/home/graham/516_Fall2020/GolfBallTimer/DataTaking/Run201/datafile.datS'
data = PendulumDataSet(datafile,'data',NTOSKIP,2000)
NDATA = data.size()

print(data.byte3())
print(data.byte2())
print(data.byte1())
print(data.byte0())

print(data.ibyte3())
print(data.ibyte2())
print(data.ibyte1())
print(data.ibyte0())

print(data.clockticks())

print(data.ttype())

print(data.PeriodD(0))
print(data.PeriodD(1))
print(data.PeriodD(2))
print(data.PeriodD(3))

print(data.event())

data.stats(0)
data.stats(1)
data.stats(2)
data.stats(3)

data.stats2(0)
data.stats2(1)
data.stats2(2)
data.stats2(3)

