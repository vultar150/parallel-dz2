#!/bin/bash

mkdir -p output 
mkdir -p err

for i in 2 4 8 16 60
do
for eps in 3.0e-5 5.0e-6 1.5e-6
do
mpisubmit.pl -p $i -w 00:15 --stdout output/mainout$i-$eps.out --stderr err/mainerr$i-$eps.err a.out -- $eps
done
done
