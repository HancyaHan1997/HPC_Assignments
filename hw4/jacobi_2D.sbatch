#!/bin/bash 
#SBATCH --nodes=4                        # requests 4 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=0:20:00
#SBATCH --mem=2GB
#SBATCH --job-name=jacobi_2D
#SBATCH --mail-type=END
#SBATCH --mail-user=xh2223@nyu.edu
#SBATCH --output=jacobi_2D.out

module purge
module load openmpi/gcc/4.0.5 
mpirun -np 4 ./jacobi_2D 16 10000