PROJECT_DIR="$(pwd)"

echo "Creating conda environment..."
conda create -p $PROJECT_DIR/env python=3.10.15 -y

echo "Activating conda environment..."
source activate $PROJECT_DIR/env

echo "Installing requirements..."
pip install -r requirements.txt

echo "Installing build-essential..."
sudo apt-get update 
sudo apt-get install -y build-essential 

echo "Installing default-jre (Java)..."
sudo apt install default-jre -y

echo "Setting up environment variables..."
export JAVA_PATH=java
export COCO_CAPTION_PATH=$PROJECT_DIR/utils
export CONNETTE_PATH=$PROJECT_DIR/utils

echo "Setup completed successfully! Activate the environment with 'source activate $PROJECT_DIR/env'" 
