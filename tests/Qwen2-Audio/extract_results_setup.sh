#!/bin/bash

declare -a experiments=('set_evalsampling_10_epochs_clotho_maxlength_480000' 'set_evalsampling_1_epochs_audiocaps_maxlength_160000')

DIR_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MY_HOME=${DIR_SCRIPT}

# Create the results directory if it does not exist
mkdir -p ${MY_HOME}/results
# Go in the results directory
cd ${MY_HOME}/results

if [ -d "saved_csv" ]; then
    rm -r saved_csv
fi

mkdir saved_csv

DIR_QWEN2_AUDIO=${MY_HOME}/../../Qwen2-Audio

cd ${DIR_QWEN2_AUDIO}

for experiment_name in "${experiments[@]}"
do  
    python ${DIR_QWEN2_AUDIO}/scripts_download_csv.py --experiment_name=${experiment_name} --save_dir=${MY_HOME}/results/saved_csv --mlflow_uri=${DIR_QWEN2_AUDIO}/logs/mlflow/mlruns
    echo "done for ${experiment_name}"
done