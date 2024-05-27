import argparse
import os
import json
from tqdm import tqdm
import shortuuid

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_answer_vqa, get_input_ids_from_query, get_rationale_vqa, get_smg_prompt, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, load_8bit=args.load_8bit, load_4bit=args.load_4bit)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # crate data_loader of smg question
    prompts_smg = get_smg_prompt(questions)  # List, len is len(questions)

    for prompt_smg, line in tqdm(zip(prompts_smg, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs

        # Get image_tensor from image_file
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # Ask model to generate questions
        rationale = get_rationale_vqa(args, model, tokenizer, prompt_smg, image_tensor, debug=args.debug)

        # Ask model to generate final reasoning result.
        print("-"*40 + f"Generating Final Reasoning Result of sample {idx} ..." + "-"*40) if args.debug else None
        cur_prompt = f"Please answer the quesion according to the context. \n\n[Context]: '{rationale}';\n\n[Question]: {cur_prompt}"
        input_ids = get_input_ids_from_query(args.conv_mode, cur_prompt, tokenizer, model.config)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        outputs = get_answer_vqa(args, model, tokenizer, input_ids, image_tensor)
        print(f"Current question: {cur_prompt}, \n\nOutput: {outputs}\n\n") if args.debug else None

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": qs,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    args = parser.parse_args()

    eval_model(args)
