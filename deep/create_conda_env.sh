#!/bin/bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /tmp/ 
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b 
eval "$(/root/miniconda3/bin/conda shell.bash hook)" 
conda init 
conda update -n base -c defaults conda 
conda config --add channels conda-forge 
conda env create -f deep_environment.yml
conda activate deeplearning 