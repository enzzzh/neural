#!/bin/sh
cd ../ 
gcc -o neural neural.c -lm 
./neural 0.14159265 0.000001
# please change value depending on how beefy your computer is, multithreading and parallelism is still running like shit so your pc might crash
#
# NOTE TO SELF: make program memory safe

