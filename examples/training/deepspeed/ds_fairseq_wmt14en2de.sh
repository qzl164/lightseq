#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

if [ ! -d "wmt14_en_de" ]; then
    echo "Downloading dataset"
    wget http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/lightseq/wmt_data/databin_wmt14_en_de.tar.gz -P /tmp
    tar -zxvf /tmp/databin_wmt14_en_de.tar.gz && rm /tmp/databin_wmt14_en_de.tar.gz
fi

deepspeed ${THIS_DIR}/ds_fairseq.py \
    /tmp/wmt14_en_de/ \
    --user-dir  ${THIS_DIR}/../../../lightseq/training/fs_modules \
    --arch ls_transformer_wmt_en_de_big_t2t --share-decoder-input-output-embed \
    --optimizer ls_adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0001 \
    --criterion ls_label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 8192 \
    --log-interval 200 \
    --validate-interval-updates 2000 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --fp16 \
    --deepspeed_config ${THIS_DIR}/deepspeed_config.json
