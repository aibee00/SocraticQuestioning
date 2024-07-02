# Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild
---

This is the official completion of the paper: "Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild." The code will be coming soon.

Our dataset is available at [CapQA](https://huggingface.co/datasets/wphu/CapQA).


## How to use

Before running the evaluation script in this repository, you need to copy it to the official [LLAVA](https://github.com/haotian-liu/LLaVA) repository to replace the original ones.

Please note that you need to replace the original `llava/mm_utils` with the new `llava/mm_utils` from this repository, as there are several newly added functions implementing our "Socratic Questioning" methodology.

Please download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing) and unzip it under './playground/data' .
