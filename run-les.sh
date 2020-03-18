#!/bin/bash                                                                                                                    

#SBATCH --job-name=cb32-1.0-imex
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --tasks-per-node=2
#SBATCH --time=18:00:00

#SBATCH --mail-user=asridhar@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/central/scratch/asridhar/BOMEX-MRRK-C18-50x20/output.jl

set -euo pipefail # kill the job if anything fails

module load julia
module load openmpi/4.0.1_cuda-10.0 cmake/3.10.2 cuda/10.0

julia --project=/home/asridhar/CLIMA/ -e 'using Pkg; Pkg.instantiate(); Pkg.API.precompile()'

mpirun julia --project=/home/asridhar/CLIMA /home/asridhar/CLIMA/experiments/AtmosLES/bomex.jl --enable-vtk --vtk-interval=2000 --enable-diagnostics --diagnostics-interval=2000 --output-dir=/central/scratch/asridhar/BOMEX-MRRK-C18-50x20
