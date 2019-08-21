#!/bin/sh

EXP_DIR=exp_train_triplet
mkdir -p ${EXP_DIR}
now=$(date +"%Y%m%d_%H%M%S")

ROOT=./
export PYTHONPATH=$ROOT:$PYTHONPATH

srun -p $1 -n 2 --gres=gpu:2 --ntasks-per-node=2 \
    --job-name=cexp4 --kill-on-bad-exit=1 \
python -W ignore -u eval_triplet.py \
  --data_root=/mnt/lustre/pangchao/datasetmake/planet/europe_201801/data_20000_end \
  --train_list=../../data/csv/train_1.csv \
  --val_list=../../data/csv/val_1.csv \
  --config=config.json \
  --syncbn=1 --bn_group=8 \
  --dist=1 \
  --port=22000 \
  --gpu 0 1 --base_lr=1e-3 --epochs=1 --save_step=1 --start_epoch=1 \
  --batch_size=4 \
  --batch_size_val=1\
  --evaluate=1 \
  --weighted=1 \
  --print_freq=5 \
  --save_path=${EXP_DIR} \
  --root_be='../../data/beforechange'\
  --root_af='../../data/afterchange'\
  --root_mask='../../data/label_change_new'\
  --pretrain='/mnt/lustre/pangchao/cd_paper/code_v2.0/cexp4/exp_train_triplet/model_20.pkl'\
  2>&1 | tee ${EXP_DIR}/train-$now.log

