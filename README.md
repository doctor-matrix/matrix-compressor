# Matrix Compressor

This project is an implementation of the LPLR framework for applying a joint low rank quantization technique to compress matrices, with applications in domains with large data matrices. Popular applications include Deep Learning, Graph (Neural) Networks, Computational Biology, Large scale data retrieval amongst others.

## Setup

The repository uses `conda` for setup, other python dependency management frameworks may be utilized as appropriate by importing `environment.yml`.

```bash
conda import -f environment.yml
```

## Common operations

### Shepp Logan Image Generation

```
mkdir -p artifacts/paper
python scripts/phantom/low_precision_low_rank.py
```

### MobileNetV3 Embeddings for CIFAR10

#### Generation

```fish
#!/usr/bin/env fish
set -x EXECUTOR python
set -x SCRIPT scripts/mnetv3/save_quantized_splits.py
set -x SKETCH Gaussian
set -x CUDA_VISIBLE_DEVICES 2
set -x INPUT_DATA artifacts/custom_data/knn/mnetv3/cifar10/cifar10-train-embeddings.pt
set -x OUTPUT_DIR artifacts/custom_data/knn/mnetv3/cifar10-quantized-full/train

for b0 in 1 2 4 8
$EXECUTOR $SCRIPT --input-data-location $INPUT_DATA --output-data-location $OUTPUT_DIR --map-location cpu --b1 8 --b2 8 --b-nq $b0 --cr 1 --force
end
```

#### Evaluation

```fish

set -x EXECUTOR python
set -x SCRIPT scripts/mnetv3/evaluated_preloaded_embeddings.py
set -x CUDA_VISIBLE_DEVICES 2
set -x TRAIN_DIR artifacts/custom_data/knn/mnetv3/cifar10
set -x BASE_QUANT_DIR artifacts/custom_data/knn/mnetv3/cifar10-quantized-full/train
set -x LOG_DIR misc/knn/embedding-evaluation-metrics/cifar10

set -x B1 8
set -x B2 8

for BNQ in 1 2 4 8
$EXECUTOR $SCRIPT --quant-directory $BASE_QUANT_DIR/b1_$B1-b2_$B2-bnq_$BNQ --b1 $B1 --b2 $B2 --b_nq $BNQ --train-dir $TRAIN_DIR --eval-train 2>&1 | tee -ia $LOG_DIR/evaluation-(date -u +%s).log
end
```


### LlaMa Layer Analysis

Replace the MODEL_DIRECTORY by the location of the unquantized model, and adjust parameters `b1`, `b2` and `cr`.

```bash
export OUTPUT_DIRECTORY="./artifacts/llama-quantized"
export MODEL_DIRECTORY="./artifacts/llama-7b-hf/"
export LOGURU_LEVEL=INFO 
stdbuf -oL python scripts/llama/per_layer_naive_quantization_comparison/lplr_vanilla.py --model-directory $MODEL_DIRECTORY --output-directory $OUTPUT_DIRECTORY --b1 8 --b2 8 --cr 1 --map-location "cuda:1" 2>&1 | stdbuf -oL tee -i $OUTPUT_DIRECTORY/quantization-$(date +%m%d%H%M%S).log
```

### Quantization and Evaluation

#### Prequisites

```bash
mkdir repos
cd repos
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
```

#### Evaluation

```bash
export RF=0.8
export B1=16
export B2=16
export CUDA_VISIBLE_DEVICES=1
export LOGURU_LEVEL=TRACE
export INPUT_DIR=artifacts/llama-7b-hf
export OUTPUT_DIR=artifacts/llama-quantized-svd-r{$RF}-{$B1}-{$B2}

stdbuf --output=L python scripts/llama/batch_quantizers/quantize_llama_svd.py --in-path $INPUT_DIR --out-path $OUTPUT_DIR --map-device 'cuda:0' --rank-fraction $RF --b1 $B1 --b2 $B2 2>&1 | stdbuf --output=L tee $OUTPUT_DIR/quantization.log

cp artifacts/llama-7b-hf/tokenizer.model artifacts/llama-7b-hf/*.json $INPUT_DIR

stdbuf --output=L python repos/lm-evaluation-harness/main.py --model hf-causal --model_args pretrained=./$INPUT_DIR --tasks boolq,hellaswag,piqa 2>&1 | stdbuf --output=L tee $INPUT_DIR/evaluation.log
```