#!/bin/bash

MODEL_BASE=vicuna-7b-v1.5
MODEL_QLORA_BASE=llava-v1.5-7b-lora
MODEL_SUFFIX=$MODEL_QLORA_BASE

python -m llava.eval.model_vqa_loader_3turn \
    --model-path ./checkpoints/${MODEL_QLORA_BASE} \
    --model-base ./checkpoints/${MODEL_BASE} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1


python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl
