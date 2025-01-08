#!/bin/bash

MODEL_BASE=vicuna-7b-v1.5
MODEL_QLORA_BASE=llava-v1.5-7b-lora
MODEL_SUFFIX=$MODEL_QLORA_BASE-3turn


python -m llava.eval.model_vqa_science_3turn \
    --model-path ./checkpoints/${MODEL_QLORA_BASE} \
    --model-base ./checkpoints/${MODEL_BASE} \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_output-${MODEL_SUFFIX}.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_result-${MODEL_SUFFIX}.json
