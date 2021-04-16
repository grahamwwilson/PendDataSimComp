#!/bin/sh
# runall.sh
#
# Goal run chi-squared calculation on all SimDataFile-*.dat files
# to get an idea of which one is the best so far, using this 
# particular metric.
#

listfile=MYLISTFILE.dat

ls -1 SimDataFile-*.dat >${listfile}
cat ${listfile}

for filename in $(cat ${listfile})
do
   python ChiSquaredCalculation.py --file ${filename}
done

rm ${listfile}

exit
