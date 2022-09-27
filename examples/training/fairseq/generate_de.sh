#!/usr/bin/env bash
set -ex
THIS_DIR=$(dirname $(readlink -f $0))
cd $THIS_DIR/../../..

model=int4_ende

lightseq-generate /tmp/wmt14_en_de/  \
        --path $model/checkpoint_best.pt --gen-subset test --quiet \
        --beam 4 --max-tokens 8192 --remove-bpe --lenpen 0.6 --fp16
