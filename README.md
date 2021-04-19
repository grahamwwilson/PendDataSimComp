# PendDataSimComp

Contains 3 programs.

python ClassyPlots.py               Recommended starting point

python DataSimCompPlots.py          Previous more procedural code

python ChiSquaredCalculation.py     Previous initial implementation

Makes a number of plots comparing a particular simulation 
data-set with the Run 76 data.

Also now includes a chi-squared calculation, comparing the measurements 
with simulation.

Chi-squared implementation 0.
  1) Shadow times (t(i+1) - t(i)) with errors of 0.2 ms
  2) Mean time during shadow (1/2 (t(i+1) + t(i)) ) with errors of 2 ms.
Each is implemented separately for up-stream and down-stream positions.

Also included a shell script (runall.sh) to run the chi-squared 
calculation on all files in current directory named Simulation-*.dat

Refactored 4-18-2021 to hide all the methods in MyPendulumClass.py 
and use a more object oriented way to do the plots. Still a work in 
progress, but illustrative code is in ClassyPlots.py.
Plan to incorporate the chi-squared calculations in a similar way.
