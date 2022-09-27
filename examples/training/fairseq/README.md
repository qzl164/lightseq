# LightSeq for Fairseq
This repo contains examples for how to use LightSeq to accerate the training of translation task in [Fairseq](https://github.com/pytorch/fairseq).

First you should install these requirements.
```shell
$ pip install fairseq sacremoses
$ cd lightseq
$ pip install -e ./
```

## Train
First you need to train a full precision model on wmt14 en2de dataset using LightSeq by running the following script:
```shell
$ sh examples/training/fairseq/ls_torch_fairseq_wmt14en2de.sh
```

You can then use `--quant-bits 4`, `--enable-quant` and `--quant-mode qat` to fine-tune the full precision model to run quantization aware training, like the following script:
```shell
$ sh examples/training/fairseq/ls_torch_fairseq_quant_wmt14en2de.sh
```

## Generate
Then you can evaluate on wmt14 en2de dataset by running the following command:
```shell
$ lightseq-generate /tmp/wmt14_en_de/  \
        --path int4_ende/checkpoint_best.pt --gen-subset test \
        --beam 4 --max-tokens 8192 --remove-bpe --lenpen 0.6 \
        --quiet --fp16
```
