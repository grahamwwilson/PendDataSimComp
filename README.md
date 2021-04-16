# PendDataSimComp

python DataSimCompPlots.py

Makes a number of plots comparing a particular simulation 
data-set with the Run 76 data.

Also now includes a chi-squared calculation, 
python ChiSquaredCalculation.py comparing the measurements 
with simulation.

Implementation 0.
  1) Shadow times (t(i+1) - t(i)) with errors of 0.2 ms
  2) Mean time during shadow (1/2 (t(i+1) + t(i)) ) with errors of 2 ms.
Each is implemented separately for up-stream and down-stream positions.

Also included a shell script (runall.sh) to run the chi-squared 
calculation on all files in current directory named Simulation-*.dat
