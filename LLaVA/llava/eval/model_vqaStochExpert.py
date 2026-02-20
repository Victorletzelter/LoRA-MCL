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
sys.path.append("./metrics_code")

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
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    from llava.model import LlavaLlamaForCausalLM_MCL,LlavaLlamaForCausalLM
    model = LlavaLlamaForCausalLM_MCL.from_pretrained(
        model_path,
        num_hyps=args.num_hyps,
        wta_training_mode='relaxed-wta',
        wta_params_epsilon=0.1,
    )
    num_hyps = model.num_hyps
  
    # Load LoRA weights if specified
    if hasattr(args, 'lora_weights') and args.lora_weights:
        from peft import PeftModel
        print(f"Loading LoRA weights from {args.lora_weights}")

        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model),
            task_type="CAUSAL_LM",
        )
        lora_config.use_group_lora = False #True
        lora_config.use_moe_lora = True  
        lora_config.num_experts = 3

        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config, adapter_name="lora0")


        def debug_moe_weights(model):
            for name, module in model.named_modules():
                if 'MoELoraLinear' in str(type(module)):
                    print(f"\nFound MoELoraLinear: {name}")
                    print(f"lora_A keys: {list(module.lora_A.keys()) if hasattr(module, 'lora_A') else 'No lora_A'}")
                    print(f"lora_B keys: {list(module.lora_B.keys()) if hasattr(module, 'lora_B') else 'No lora_B'}")
                    if hasattr(module, 'router'):
                        print(f"Router weight shape: {module.router.weight.shape}")
                        print(f"Router weight sum: {module.router.weight.sum()}")

        from safetensors.torch import load_file, save_file
        if  args.lora_weights is not None: # For checkpoint loading if needed
            for i in range(args.num_hyps):
                model.load_adapter(os.path.join( args.lora_weights, f"lora{i}"), adapter_name=f"lora{i}", is_trainable=False, torch_device=f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
        model.to(torch.float16)
        model.to("cuda")
        # Merge weights for faster inference if supported
        if hasattr(model, 'merge_and_unload'):
            try:
                model = model.merge_and_unload()
                print("LoRA weights merged with base model for faster inference")
            except Exception as e:
                print(f"Failed to merge LoRA weights: {e}")
                print("Continuing with adapter weights")

    answers=[]
    questions=[]
    with open(args.answers_file, 'r') as f:
        for line in f:
            answers.append(json.loads(line.strip()))

    with open(args.question_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line.strip()))
    num_hyps=args.num_beams

    import torch.nn as nn

    def set_random_expert(model,exp):
            for name, module in model.named_modules():
                # Check if this is a LoRA layer (commonly LoRALinear or customized Linear)
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    # Set expert_specific if it exists or can be set
                    if hasattr(module, 'random_expert'):
                        module.random_expert=True
                    else:
                        setattr(module, 'random_expert', True)
                        print(f"Set random_expert=True for: {name}")


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
    threshs=[0.5]#,0.8,1.0]
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
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            #image_tensor = process_images([image], image_processor, model.config)[0]

            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0)
            #print('image',image_tensor.shape)
            '''gen_config = GenerationConfig(
                num_beams=gen_config_dict.get("beam_size"),
                num_return_sequences=gen_config_dict.get("num_return_sequences"),
                do_sample=gen_config_dict.get("do_sample"),
                temperature=gen_config_dict.get("temperature"),
                top_k=gen_config_dict.get("top_k"),
                top_p=gen_config_dict.get("top_p"),
                diversity_penalty=gen_config_dict.get("diversity_penalty"),
                num_beam_groups=gen_config_dict.get("num_beam_groups"),
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                repetition_penalty=gen_config_dict.get("repetition_penalty", model.generation_config.repetition_penalty),
                chat_format="chatml",
                output_scores=gen_config_dict.get("output_scores"),
                return_dict_in_generate=gen_config_dict.get("return_dict_in_generate"),
                max_new_tokens=gen_config_dict.get("max_new_tokens", None),
                penalty_alpha=gen_config_dict.get("penalty_alpha", None),
            ) '''
            with torch.inference_mode():
                outputs = []
                #print('thresh',thresh)
                #breakpoint()
                for exp in range(args.num_beams):
                    set_random_expert(model,exp)
                    #breakpoint()
                    output_ids = model.generate(
                        input_ids,
                        hypothesis_idx=0,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=False,
                        top_p=args.top_p,
                        num_beams=1,#args.num_beams*args.num_boost,
                        max_new_tokens=128,
                        repetition_penalty=1.1,
                        use_cache=True)
                    decoded_predictions=[tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()]
                    # Remove newlines and backslashes, and Chinese characters
                    decoded_predictions=[re.sub(r'[\u4e00-\u9fff]', '', s.replace('\n', '').replace('\\', '').replace('_','')) for s in decoded_predictions]
                    outputs.append(decoded_predictions[0])
                    print(decoded_predictions[0])
                breakpoint()

            outputs_dict = {}
            outputs_dict["cands"] = []
            outputs_dict["mrefs"] = []
            outputs_dict["fname"] = []
            outputs_dict["index"] = []

            for i in range(args.num_beams):
                outputs_dict["cands"].append(outputs[i]) #decoded_predictions[i])
                outputs_dict["mrefs"].append(answer)
                outputs_dict["fname"].append(idx)
                outputs_dict["index"].append(i)

            update_outputs(outputsHyps,outputs_dict,num_hyps)

        print('computing metrics')
        import pickle 
        with open('./checkpoints/'+args.lora_weights.split('/')[-1]+'/'+args.lora_weights.split('/')[-1]+'_'+str(thresh)+'RANDOMEXPRepPenBoosted'+str(args.num_boost)+'.pkl', 'wb') as f:
            pickle.dump(outputsHyps, f)

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
