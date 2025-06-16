#!/bin/bash
#SBATCH --job-name=job1         # Job name
#SBATCH --output=train_output.txt    # Output file
#SBATCH --error=train_error.txt      # Error file
#SBATCH --time=24:00:00             # Maximum run time (HH:MM:SS)
#SBATCH --partition=compute
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=5                # Request 5 CPU
#SBATCH --mail-user=baumannlu@ethz.ch  # Send email notifications to this address
#SBATCH --mail-type=ALL             # Notify on job start, end, and failure

# Load python module
module load stack/2024-06
module load gcc/12.2.0
module load py-six/1.16.0-fhpdxtx
module load python/3.10.13  

# Add user-installed packages to PYTHONPATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH

# Run the script
python models.py
