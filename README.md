# chiral gnn
Custom aggregation functions for molecules with tetrahedral chirality ([arXiv](https://arxiv.org/abs/2012.00094))

## Requirements
* python (version=3.7)
* pytorch (version=1.14)
* rdkit (version=2020.03.2)

## Installation
`git clone https://github.com/PattanaikL/chiral_gnn`

## Usage
To train the model with the best-performing parameters, call the `train.py` script with the following parameters defined:

`python train.py --data_path data/d4_docking/d4_docking.csv --split_path data/d4_docking/full/split0.npy --log_dir ./test_run --gnn_type dmpnn --message tetra_permute_concat --global_chiral_features --chiral_features`
