import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import sys
import re

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def eval_model(args):
    # Model
    disable_torch_init()
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from PIL import Image
    import requests
    import torch

    model_id = "google/paligemma-3b-ft-textcaps-224"

    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id)

    answers=[]
    questions=[]
    with open(args.answers_file, 'r') as f:
        for line in f:
            answers.append(json.loads(line.strip()))

    with open(args.question_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    num_hyps=args.num_beams



    def update_outputs(outputs,outputs_dict,num_hyps):
        for i in range(num_hyps):
            outputs[f"hypothesis_{i}"]["cands"].append(outputs_dict["cands"][i])
            outputs[f"hypothesis_{i}"]["mrefs"].append(outputs_dict["mrefs"][i])
            outputs[f"hypothesis_{i}"]["fname"].append(outputs_dict["fname"][i])
            outputs[f"hypothesis_{i}"]["subset"].append('val')
            outputs[f"hypothesis_{i}"]["index"].append(outputs_dict["index"][i])
            outputs[f"hypothesis_{i}"]["dataset"].append('textCaps')
            outputs[f"hypothesis_{i}"]["preds"].append(None)
            outputs[f"hypothesis_{i}"]["losses"].append(None)

    answersList=[]
    GTs=[]
    threshs=[1.0,0.8,0.5]
    for thresh in threshs:
        outputsHyps = {}
        for i in range(num_hyps):
            outputsHyps[f"hypothesis_{i}"] = {}
            outputsHyps[f"hypothesis_{i}"]["cands"] = []
            outputsHyps[f"hypothesis_{i}"]["mrefs"] = []
            outputsHyps[f"hypothesis_{i}"]["fname"] = []
            outputsHyps[f"hypothesis_{i}"]["subset"] = []
            outputsHyps[f"hypothesis_{i}"]["index"] = []
            outputsHyps[f"hypothesis_{i}"]["dataset"] = []
            outputsHyps[f"hypothesis_{i}"]["preds"] = []
            outputsHyps[f"hypothesis_{i}"]["losses"] = []        
        for j,line in enumerate(tqdm(questions)):
            
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            answer = answers[j]["text"]

            cur_prompt = qs
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')

            prompt = "caption en"
            model_inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda')
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                outputs = []

                generation = model.generate(**model_inputs, max_new_tokens=128, do_sample=False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams*args.num_boost,
                    num_beam_groups=args.num_beams,
                    diversity_penalty=thresh,
                    num_return_sequences=args.num_beams,
                    repetition_penalty=1.1,
                    use_cache=False)
                decoded_predictions=[]

                for k in range(generation.shape[0]):
                    generationCrop = generation[k][input_len:]

                    decoded = processor.decode(generationCrop, skip_special_tokens=True)
                
                    decoded_predictions.append(decoded)
                
                print(decoded_predictions)

            outputs_dict = {}
            outputs_dict["cands"] = []
            outputs_dict["mrefs"] = []
            outputs_dict["fname"] = []
            outputs_dict["index"] = []

            for i in range(args.num_beams):
                outputs_dict["cands"].append(decoded_predictions[i])
                outputs_dict["mrefs"].append(answer)
                outputs_dict["fname"].append(idx)
                outputs_dict["index"].append(i)
            update_outputs(outputsHyps,outputs_dict,num_hyps)

        print('computing metrics')
        import pickle 
        try:
            with open('./checkpoints/'+'paliGemma_'+str(thresh)+'RepPenBoosted'+str(args.num_boost)+'SINGLEHYP.pkl', 'wb') as f:
                pickle.dump(outputsHyps, f)
        except:
            breakpoint()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--lora_weights", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_hyps", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--num_boost", type=int, default=1)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    eval_model(args)
