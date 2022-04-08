# Developed by Kevin A. Spiekermann
# This script does the following tasks:
# 	- creates the conda
# 	- prompts user for desired CUDA version
# 	- installs PyTorch with specified CUDA version in the environment
# 	- installs torch torch-geometric in the environment


# get OS type
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=MacOS;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "Running ${machine}..."


# request user to select one of the supported CUDA versions
# source: https://pytorch.org/get-started/locally/
CUDA="cudatoolkit=11.1"
CUDA_VERSION="cu111"

echo "Creating conda environment..."
echo "Running: conda env create -f environment.yml"
conda env create -f environment.yml

# activate the environment to install torch-geometric
source activate chiral_gnn

echo "Installing PyTorch with requested CUDA version..."
echo "Running: conda install pytorch torchvision $CUDA -c pytorch"
conda install pytorch torchvision $CUDA -c pytorch

echo "Installing torch-geometric..."
echo "Using CUDA version: $CUDA_VERSION"
# get PyTorch version
TORCH_VERSION=1.8.0
echo "Using PyTorch version: $TORCH_VERSION"

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
pip install torch-geometric
