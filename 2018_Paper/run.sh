#!/bin/bash

# List of different flag combinations to pass to the script
flag_lists=(
#    "--config_path=experiment_configs/allencahn_brownian.json --log_dir=log_allencahn --exp_name=brownian"
#    "--config_path=experiment_configs/allencahn_hadamard.json --log_dir=log_allencahn --exp_name=hadamard"
#  "--config_path=experiment_configs/pricing_default_risk_brownian.json --log_dir=log_risk --exp_name=brownian"
#  "--config_path=experiment_configs/pricing_default_risk_hadamard.json --log_dir=log_risk --exp_name=hadamard"
    "--config_path=experiment_configs/hjb_lq_brownian.json --log_dir=log_hjb --exp_name=brownian"
    "--config_path=experiment_configs/hjb_lq_hadamard.json --log_dir=log_hjb --exp_name=hadamard"
)

# Make sure the Python script is executable
chmod +x main.py

# Loop through each flag combination and call the Python script
for flags in "${flag_lists[@]}"; do
    echo "Running: python main.py $flags"
    python main.py $flags
done
