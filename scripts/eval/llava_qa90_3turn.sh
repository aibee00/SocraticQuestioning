#!/bin/bash
# MMHal-Bench Evaluation
MODEL_BASE=vicuna-7b-v1.5
MODEL_QLORA_BASE=llava-v1.5-7b-lora
MODEL_SUFFIX=$MODEL_QLORA_BASE-3turn

[[ -e "playground/data/eval/llava_qa90" ]] || mkdir -p playground/data/eval/llava_qa90

python -m llava.eval.model_vqa_3turn \
    --model-path ./checkpoints/${MODEL_QLORA_BASE} \
    --model-base ./checkpoints/${MODEL_BASE} \
    --question-file \
    playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    playground/data/coco2014_val \
    --answers-file \
    playground/data/eval/llava_qa90/answer-file-${MODEL_SUFFIX}.jsonl
    

OPENAI_API_KEY="your openai-api-key" python llava/eval/eval_gpt_review_visual.py \
    --question playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --context llava/eval/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    playground/data/coco2014_val_qa_eval/qa90_gpt4_answer.jsonl \
    playground/data/eval/llava_qa90/answer-file-${MODEL_SUFFIX}.jsonl \
    --rule llava/eval/table/rule.json \
    --output playground/data/eval/llava_qa90/review-file-${MODEL_SUFFIX}.jsonl

python -m llava.eval.summarize_gpt_review \
    --files playground/data/eval/llava_qa90/review-file-${MODEL_SUFFIX}.jsonl


