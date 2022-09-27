#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

if [ ! -d "/tmp/wmt14" ]; then
    echo "Downloading dataset"
    hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/duanrenchong/datasets/en-fr/onefile_databin /tmp/wmt14
fi

lightseq-train /tmp/wmt14/ \
    --arch ls_transformer --task translation \
    --save-dir int4_enfr \
    --finetune-from-model fp16_enfr/checkpoint_best.pt \
    --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --warmup-updates 4000 --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler polynomial_decay \
    --lr 5e-4 --total-num-update 300000 --end-learning-rate 1e-6 \
    --save-interval-updates 3000 --keep-interval-updates 1 \
    --max-tokens 8192 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --fp16 \
    --use-torch-layer \
    --enable-quant \
    --quant-mode qat --quant-bits 4 \
    --max-epoch 160 --keep-last-epochs 1 \
    --n-gpus-intk 1 --smooth-avg-update 200 \
    --max-epoch 25
