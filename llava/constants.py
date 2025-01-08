CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

INSTRUCTION = "请你对图片中活动相关的细节提出5～8个问题，如果某些问题不确定，就再将问题细化，再针对不确定的问题提出5个需要的问题，最多不超过20个提问。中文回答。"

PROMPT = f"Below is an instruction that describes a task, paired with an image that provides further context. " \
    f"Write a response that appropriately completes the request.\n\n" \
    f"Instruction: {INSTRUCTION}\n\n"

# SMG_PROMPT_TEMPLATE = "Please formulate 5 to 8 questions related to the activity details in the image. If some questions are uncertain, further refine them, and pose an additional 5 questions specifically targeting these uncertainties, with a total not exceeding 20 questions. The purpose of these questions should be to assist the model in complete the answer to this question: '{}'. Aim to ask questions that can be definitively answered and avoid questions that do not have clear answers."

SMG_PROMPT_TEMPLATE = "Please formulate 4 to 6 questions related to the activity details in the image. The purpose of these questions should be to assist the model in complete the answer to this question: '{}'. Aim to ask questions that can be definitively answered and avoid questions that do not have clear answers."
