#!/bin/bash
#### this a bash script which we write out and then submit to the batch queue
# This script is intepreted by the Bourne Shell, sh
#
#SBATCH --account=csd562
#SBATCH --job-name=apf-strong-scale
#SBATCH --output="apf-large-scale_15.out"
#SBATCH --partition="compute"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu
#SBATCH -t 00:02:00
#### 96, 192, 240, 384 and 480
# Commands go here
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 3 -y 8
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 2 -y 12
#ibrun -np 24 ./apf -n 1800 -i 2000 -x 8 -y 3
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 2 -y 24
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 3 -y 16
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 4 -y 12
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 12 -y 4
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 24 -y 2
#ibrun -np 48 ./apf -n 1800 -i 2000 -x 24 -y 2
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 4 -y 24
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 2 -y 48
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 1 -y 96
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 6 -y 16
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 4 -y 24
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 1 -y 96
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 2 -y 48
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 3 -y 32
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 4 -y 24
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 6 -y 16
#ibrun -np 96 ./apf -n 1800 -i 2000 -x 4 -y 24
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 8 -y 12
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 12 -y 8
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 16 -y 6
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 8 -y 12
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 6 -y 16
#ibrun -np 96 ./apf -n 8000 -i 2000 -x 4 -y 24
ibrun -np 192 ./apf -n 8000 -i 2000 -x 32 -y 6
ibrun -np 192 ./apf -n 8000 -i 2000 -x 6 -y 32
#ibrun -np 240 ./apf -n 8000 -i 2000 -x 6 -y 40
#ibrun -np 384 ./apf -n 8000 -i 2000 -x 6 -y 32
#ibrun -np 480 ./apf -n 8000 -i 2000 -x 16 -y 30
#ibrun -np 480 ./apf -n 8000 -i 2000 -x 20 -y 24
#ibrun -np 480 ./apf -n 8000 -i 2000 -x 8 -y 60
