#!/bin/bash
#SBATCH --job-name=matmatgpu
#SBATCH --output=prova.out
#SBATCH --error=prova.err
#SBATCH --time=00:60:00
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1



nvcc  -o matmatgpu main.cu -Xcompiler -fopenmp  -lmpi -I/usr/mpi/gcc/openmpi-4.1.0rc5/include/ -L/usr/mpi/gcc/openmpi-4.1.0rc5/lib64 -O3

mpirun ./matmatgpu

