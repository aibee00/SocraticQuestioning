#!/bin/bash

MODEL_BASE=vicuna-7b-v1.5
MODEL_QLORA_BASE=llava-v1.5-7b-lora

MODEL_SUFFIX=$MODEL_QLORA_BASE-3turn

python -m llava.eval.model_vqa_smg \
    --model-path ./checkpoints/${MODEL_QLORA_BASE} \
    --model-base ./checkpoints/${MODEL_BASE} \
    --question-file ./playground/data/eval/capqa/questions.jsonl \
    --image-folder ./playground/data/eval/capqa/images \
    --answers-file ./playground/data/eval/capqa/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/capqa/reviews

OPENAI_API_KEY="your openai-api-key" python llava/eval/eval_gpt_review_capqa.py \
    --question playground/data/eval/capqa/questions.jsonl \
    --context playground/data/eval/capqa/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/capqa/answers_gpt4.jsonl \
        playground/data/eval/capqa/answers/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl \
    --output \
        playground/data/eval/capqa/reviews/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/capqa/reviews/llava-v1.5-7b-${MODEL_SUFFIX}.jsonl
