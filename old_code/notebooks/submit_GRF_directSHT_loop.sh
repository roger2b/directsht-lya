#!/bin/bash

# Define the ranges for each parameter
chi_shift=(5000 8000 10000 12000)
ell_max=(500)
lambda_max=(1000)
L_max=(2000)
num_sim=(50)
num_qso=(10000)

# Loop through each combination of parameters
for param1 in "${chi_shift[@]}"; do
  for param2 in "${ell_max[@]}"; do
    for param3 in "${lambda_max[@]}"; do
      for param4 in "${L_max[@]}"; do
        for param5 in "${num_sim[@]}"; do
          # Construct the command
          cmd="python lya_GRFs_directSHT_loop_26062024.py $param1 $param2 $param3 $param4 $param5 $num_qso"      
          # Submit the job
          echo "Submitting job with parameters: $param1 $param2 $param3 $param4 $param5 $num_qso"
          $cmd
        done
      done
    done
  done
done