import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    SMG_PROMPT_TEMPLATE,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    get_rationale_mmhal,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    get_input_ids_from_query,
    get_image_tensor_from_image_file,
    get_answer_mmhal
)
from llava.model import *
from PIL import Image
import math
from peft import PeftModel

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = 'LLaVa-RLHF_' + get_model_name_from_path(model_path)
    compute_dtype = torch.float16
    if args.use_qlora:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        bits = 16
        dtype = torch.bfloat16
        compute_dtype = torch.bfloat16

        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            device_map={"": "cuda:0"},
            torch_dtype=dtype,
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["mm_projector", "lm_head"],
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = PeftModel.from_pretrained(
            model,
            args.qlora_path,
        )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=compute_dtype)
        image_processor = vision_tower.image_processor
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit
        )

    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    print("Loading mmhal dataset ...")
    # dataset = load_dataset("Shengcao1006/MMHal-Bench")['test']
    dataset = load_dataset("./playground/data/eval/mmhal/MMHal-Bench")['test']
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    record = []
    for line in tqdm(dataset):
        # use the pre-downloaded images
        cur_prompt = line["question"]
        image_file = line["image_path"]

        # get image_tensor
        image_tensor = get_image_tensor_from_image_file(args.image_aspect_ratio, image_file, image_processor)

        # Get rationale
        prompt_smg = SMG_PROMPT_TEMPLATE.format(cur_prompt)
        rationale = get_rationale_mmhal(args, model, tokenizer, prompt_smg, image_tensor, compute_dtype, stop_str, keywords, debug=args.debug)

        # Ask model to generate final reasoning result.
        print("-"*40 + f"Generating Final Reasoning Result ..." + "-"*40) if args.debug else None
        cur_prompt = f"Please answer the quesion according to the context. \n\n[Context]: '{rationale}';\n\n[Question]: {cur_prompt}"
        input_ids = get_input_ids_from_query(args.conv_mode, cur_prompt, tokenizer, model.config)
        
        # Forward Inference
        outputs = get_answer_mmhal(args, model, tokenizer, input_ids, image_tensor, compute_dtype, stop_str, keywords)
        print(f"Current question: {cur_prompt}, \n\nOutput: {outputs}\n\n") if args.debug else None

        line["model_answer"] = outputs
        record.append(line)

    json.dump(record, ans_file, indent=2)
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")
    parser.add_argument("--short_eval", type=bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    parser.add_argument("--test-prompt", type=str, default='\nAnswer the question using a single word or phrase.')
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.answers_file):
        print(f"{args.answers_file} already exists. Please delete it first.")
        exit(1)
    eval_model(args)