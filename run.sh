#!/bin/bash

mkdir -p output 
mkdir -p err

for ((i = 2; i <= 32; i*=2))
do
for eps in 3.0e-6 5.0e-7 1.5e-7
do
mpisubmit.pl -p $i -w 00:15 --stdout output/mainout$i-$eps.out --stderr err/mainerr$i-$eps.err a.out -- $eps
done
done
