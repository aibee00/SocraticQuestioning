#!/bin/bash

MODEL_BASE=vicuna-7b-v1.5
MODEL_QLORA_BASE=llava-v1.5-7b-lora
MODEL_SUFFIX=$MODEL_QLORA_BASE

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/${MODEL_QLORA_BASE} \
    --model-base ./checkpoints/${MODEL_BASE} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.5-7b-${MODEL_SUFFIX}.json

