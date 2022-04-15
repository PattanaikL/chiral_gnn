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

echo "Creating conda environment..."
echo "Running: conda env create -f environment.yml"
conda env create -f environment.yml

# activate the environment to install torch-geometric
source activate ml_prop

conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

pip install torch_cluster-1.5.9-cp37-cp37m-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp37-cp37m-linux_x86_64.whl
pip install torch_sparse-0.6.10-cp37-cp37m-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp37-cp37m-linux_x86_64.whl
pip install torch-geometric
