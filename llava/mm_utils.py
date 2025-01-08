from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria

from copy import deepcopy
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SMG_PROMPT_TEMPLATE
from llava.conversation import conv_templates, SeparatorStyle


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):  # concat prompt and seq([image_token_index] * (offset + 1)) 
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def get_input_ids_from_query(conv_mode, questions, tokenizer, model_config):
    """
    Args:
        conv_mode: str, mode of conversation
        questions: List[Dict[keys['question_id', 'image', 'text', 'category']] or str]
    """
    if not isinstance(questions, list):
        questions = [questions]
    
    input_ids = []
    for question in questions:
        qs = question["text"] if isinstance(question, dict) else question
        if model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        cur_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids.append(cur_input_ids)
    input_ids = torch.stack(input_ids, dim=0)
    return input_ids


def get_input_ids_from_query_sqa(args, line, question, tokenizer, model_config):
    qs = question
    
    if 'image' in line:
        if getattr(model_config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if args.single_pred_prompt:
        qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    return input_ids


def get_smg_prompt(questions):
    """ Get SMG prompt from questions.
    Args:
        questions: List[{keys['question_id', 'image', 'text', 'category']}]
        e.g.: {'question_id': 20003000, 'image': 'COCO_val2014_000000277289.jpg', 'text': 'Is there a chair in the image?\nAnswer the question using a single word or phrase.', 'category': 'popular'}}
    """
    questions_smg = deepcopy(questions)
    for q in tqdm(questions_smg):
        assert "text" in q or "conversations" in q, "text or conversations not found in question"
        
        if "text" in q:
            cur_prompt = q["text"]
        else:
            cur_prompt = q['conversations'][0]
            cur_prompt = cur_prompt['value'].replace('<image>', '').strip()
        
        smg_question = SMG_PROMPT_TEMPLATE.format(cur_prompt)
        q["text"] = smg_question
    return questions_smg


def get_answer(args, model, tokenizer, input_ids, image_tensor, temperature=None):
    """ Get answer of one question.
    Args:
        input_ids: torch.Tensor, shape [b, seq_len]
        image_tensor: torch.Tensor, shape [b, c, h, w]
    """
    temperature = args.temperature if temperature is None else temperature
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    # Ask model to generate questions
    with torch.inference_mode():
        output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
    
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs


def get_answer_mmhal(args, model, tokenizer, input_ids, image_tensor, compute_dtype, stop_str, keywords, temperature=None):
    """ Get answer of one question, used for forward inference of MMHal-Bench evaluation.
    Args:
        input_ids: torch.Tensor, shape [b, seq_len]
    """
    # get stopping criteria
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    temperature = args.temperature if temperature is None else temperature

    input_ids = input_ids.to(device='cuda', non_blocking=True)
    
    model.config.use_cache = True
    model.config.cache_shape = (2048,)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=compute_dtype).cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=64 if args.short_eval else 1024,
            # stopping_criteria=[stopping_criteria],
            use_cache=True,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (
        (input_ids != output_ids[:, :input_token_len]).sum().item()
    )
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def get_answer_vqa(args, model, tokenizer, input_ids, image_tensor, temperature=None):
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    temperature = args.temperature if temperature is None else temperature

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def get_answer_sqa(args, model, tokenizer, input_ids, images, temperature=None):
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)] if conv.version == "v0" else None
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    temperature = args.temperature if temperature is None else temperature
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def get_rationale(args, model, tokenizer, prompt_smg, image_tensor, debug=False):
    """
    Before get the final answer, we first ask model to generate questions.
    Then, we ask the model to answer these questions.
    We can view this questions and answers as rationale to aid LLM to reasoning.

    Return: 
        quesions and answers, str
    """
    def combine(questions, answers):
        rationale = "{}; \n{}".format(questions, answers)
        return rationale
    
    ## ------------------- Step 1. Ask model to generate questions ------------------- ##
    print("-"*40 + "Generating Questions ..." + "-"*40) if debug else None
    # Gen input_ids for prompt
    print(f"prompt_smg: {prompt_smg['text']}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt_smg, tokenizer, model.config)
    questions = get_answer(args, model, tokenizer, input_ids, image_tensor, temperature=0.7)
    print(f"questions: {questions}\n") if debug else None

    ## ------------------- Step 2. Ask model to generate answers of all quesion ------------------- ##
    print("-"*40 + "Generating Answers ..." + "-"*40) if debug else None
    prompt = f"Please answer all the questions one by one directly, answers are split by line break.\n[Questions]:{questions}\n[Answers]:"
    print(f"prompt: {prompt}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt, tokenizer, model.config)
    answers = get_answer(args, model, tokenizer, input_ids, image_tensor)
    print(f"answers: {answers}\n") if debug else None

    # Combine quesions and answers as rationale
    rationale = combine(questions, answers) if 'Q1' not in answers else answers
    return rationale


def get_rationale_mmhal(args, model, tokenizer, prompt_smg, image_tensor, compute_dtype, stop_str, keywords, debug=False):
    """
    Before get the final answer, we first ask model to generate questions.
    Then, we ask the model to answer these questions.
    We can view this questions and answers as rationale to aid LLM to reasoning.

    Return: 
        quesions and answers, str
    """
    def combine(questions, answers):
        rationale = "{}; \n{}".format(questions, answers)
        return rationale
    
    ## ------------------- Step 1. Ask model to generate questions ------------------- ##
    print("-"*40 + "Generating Questions ..." + "-"*40) if debug else None
    # Gen input_ids for prompt
    print(f"prompt_smg: {prompt_smg['text']}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt_smg, tokenizer, model.config)
    questions = get_answer_mmhal(
        args, model, tokenizer, input_ids, image_tensor, compute_dtype, stop_str, keywords, 0.7)
    print(f"questions: {questions}\n") if debug else None

    ## ------------------- Step 2. Ask model to generate answers of all quesion ------------------- ##
    print("-"*40 + "Generating Answers ..." + "-"*40) if debug else None
    prompt = f"Please answer all the questions one by one directly, answers are split by line break.\n[Questions]:{questions}\n[Answers]:"
    print(f"prompt: {prompt}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt, tokenizer, model.config)
    answers = get_answer_mmhal(args, model, tokenizer, input_ids, image_tensor, compute_dtype, stop_str, keywords)
    print(f"answers: {answers}\n") if debug else None

    # Combine quesions and answers as rationale
    rationale = combine(questions, answers) if 'Q1' not in answers else answers
    return rationale


def get_rationale_vqa(args, model, tokenizer, prompt_smg, image_tensor, debug=False):
    """  Used for llava_qa90
    Before get the final answer, we first ask model to generate questions.
    Then, we ask the model to answer these questions.
    We can view this questions and answers as rationale to aid LLM to reasoning.

    Return: 
        quesions and answers, str
    """
    def combine(questions, answers):
        rationale = "{}; \n{}".format(questions, answers)
        return rationale
    
    ## ------------------- Step 1. Ask model to generate questions ------------------- ##
    print("-"*40 + "Generating Questions ..." + "-"*40) if debug else None
    # Gen input_ids for prompt
    print(f"prompt_smg: {prompt_smg['text']}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt_smg, tokenizer, model.config)
    questions = get_answer_vqa(args, model, tokenizer, input_ids, image_tensor, temperature=0.7)
    print(f"questions: {questions}\n") if debug else None

    ## ------------------- Step 2. Ask model to generate answers of all quesion ------------------- ##
    print("-"*40 + "Generating Answers ..." + "-"*40) if debug else None
    prompt = f"Please answer all the questions one by one directly, answers are split by line break.\n[Questions]:{questions}\n[Answers]:"
    print(f"prompt: {prompt}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt, tokenizer, model.config)
    answers = get_answer_vqa(args, model, tokenizer, input_ids, image_tensor)
    print(f"answers: {answers}\n") if debug else None

    # Combine quesions and answers as rationale
    rationale = combine(questions, answers) if 'Q1' not in answers else answers
    return rationale


def get_rationale_sqa(args, model, tokenizer, prompt_smg, images, debug=False):
    """  Used for llava_qa90
    Before get the final answer, we first ask model to generate questions.
    Then, we ask the model to answer these questions.
    We can view this questions and answers as rationale to aid LLM to reasoning.

    Return: 
        quesions and answers, str
    """
    def combine(questions, answers):
        rationale = "{}; \n{}".format(questions, answers)
        return rationale
    
    ## ------------------- Step 1. Ask model to generate questions ------------------- ##
    print("-"*40 + "Generating Questions ..." + "-"*40) if debug else None
    # Gen input_ids for prompt
    print(f"prompt_smg: {prompt_smg['text']}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt_smg, tokenizer, model.config)
    questions = get_answer_sqa(args, model, tokenizer, input_ids, images, temperature=0.7)
    print(f"questions: {questions}\n") if debug else None

    ## ------------------- Step 2. Ask model to generate answers of all quesion ------------------- ##
    print("-"*40 + "Generating Answers ..." + "-"*40) if debug else None
    prompt = f"Please answer all the questions one by one directly, answers are split by line break.\n[Questions]:{questions}\n[Answers]:"
    print(f"prompt: {prompt}\n") if debug else None
    input_ids = get_input_ids_from_query(args.conv_mode, prompt, tokenizer, model.config)
    answers = get_answer_sqa(args, model, tokenizer, input_ids, images)
    print(f"answers: {answers}\n") if debug else None

    # Combine quesions and answers as rationale
    rationale = combine(questions, answers) if 'Q1' not in answers else answers
    return rationale


def get_image_tensor_from_image_file(image_aspect_ratio, image_file, image_processor):
    """
    Args:
        image_aspect_ratio: str, pad or square
        image_file: str, path of image file
        image_processor: transformers.ImageProcessor
    """
    image = Image.open(image_file)
    if image_aspect_ratio == 'pad':
        image = image.convert('RGB')
        def expand2square(pil_img, background_color):
            # print(background_color)
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]
    return image_tensor
