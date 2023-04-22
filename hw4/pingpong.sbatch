#!/bin/bash 
#SBATCH --nodes=2                        # requests 2 compute servers
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=pingpong
#SBATCH --mail-type=END
#SBATCH --mail-user=xh2223@nyu.edu
#SBATCH --output=pingpong.out

module purge
module load openmpi/gcc/4.0.5 
mpirun -np 2 ./pingpong 0 1