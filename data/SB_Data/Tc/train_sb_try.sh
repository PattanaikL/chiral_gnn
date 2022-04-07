#!/bin/bash
#SBATCH -J dmpnn_tpc_bcf
#SBATCH -o dmpnn_tpc_bcf.log
#SBATCH -t 1000:00:00
#SBATCH --gres=gpu:volta:1
#SBATCH -c 7





gnn_type=dmpnn
split=0

CHIRAL_GNN='/home/Sayandeep/git_repo/chiral_gnn'
log_dir='/home/Sayandeep/git_repo/chiral_gnn/data/SB_Data/Tc'
data_path="$CHIRAL_GNN/data/SB_Data/Tc/crit_all_Tc_fold0_scaled.csv"

message="sum"


echo "Start time: $(date '+%Y-%m-%d_%H:%M:%S')"
source activate chiral_gnn
split_path="$CHIRAL_GNN/data/SB_Data/Tc/Tc_split0.npy"

python -u $CHIRAL_GNN/train.py --data_path $data_path --split_path $split_path --gnn_type $gnn_type --log_dir "$log_dir/split${split}/${gnn_type}_${message}_bcf" --message $message --num_workers 20 --n_epochs 80

echo "End time: $(date '+%Y-%m-%d_%H:%M:%S')"
