#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt update

# Install Java
echo "Installing Java..."
sudo apt install default-jre -y

# Build C++ compiler
sudo apt-get install -y build-essential 

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
conda create -n peft_mcl python=3.10.15 -y
source activate peft_mcl

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt

echo "Setup completed successfully!" 