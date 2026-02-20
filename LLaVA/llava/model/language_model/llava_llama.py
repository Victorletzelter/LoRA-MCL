# This file is part of LLaVA (https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py)

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import os
import rootutils
rootutils.setup_root(__file__, indicator=".project-root")
ROOT = os.environ.get("PROJECT_ROOT")
sys.path.append(os.path.join(ROOT, 'LLaVA', 'peft_mcl'))
import peft as peft
import transformers
sys.modules["peft"] = peft
sys.modules["transformers"] = transformers

from typing import List, Optional, Tuple, Union

import torch
import copy
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import wandb

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Default values for WTA (Winner-Take-All) parameters
        self.wta_training_mode = kwargs.pop("wta_training_mode", "wta")
        self.wta_params_epsilon = kwargs.pop("wta_params_epsilon", 0.1)
        self.wta_params_ini_temp = kwargs.pop("wta_params_ini_temp", 1.0)
        self.wta_params_fin_temp = kwargs.pop("wta_params_fin_temp", 0.01)
        self.wta_params_decay_rate = kwargs.pop("wta_params_decay_rate", 0.999)
        self.wta_params_schedule_mode = kwargs.pop("wta_params_schedule_mode", "global_step")
        self.num_hyps = kwargs.pop("num_hyps", 1)
        self.orig_layer_names_one_hyp = kwargs.pop("orig_layer_names_one_hyp", True)
        self.native_group_lora_enabled = kwargs.pop("native_group_lora_enabled", True)
        self.try_new_version_adapter_setting = kwargs.pop("try_new_version_adapter_setting", False)


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_hyps = 1
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.FloatTensor] = None, #FIXME: MAKE SURE IT DOES NOT CHANGE THE OUTPUT OTHERWISE DOWNGRADE TRANSFORMERS
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)



class LlavaLlamaForCausalLM_MCL(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.wta_training_mode = config.wta_training_mode
        self.wta_params_epsilon = config.wta_params_epsilon

        #here we hardcoded the annealed-wta parameters for now
        if self.wta_training_mode=='annealed-wta':
            self.wta_params_ini_temp = 1.0
            self.wta_params_fin_temp = 1e-6
            self.wta_params_decay_rate = 0.999
            self.wta_params_schedule_mode = 'global_step'
            self.temperature = config.wta_params_ini_temp
            self.global_step=0
        else:
            self.wta_params_ini_temp = config.wta_params_ini_temp
            self.wta_params_fin_temp = config.wta_params_fin_temp
            self.wta_params_decay_rate = config.wta_params_decay_rate
            self.wta_params_schedule_mode = config.wta_params_schedule_mode
            self.temperature = config.wta_params_ini_temp
        self.num_hyps = config.num_hyps 
        self.orig_layer_names_one_hyp = config.orig_layer_names_one_hyp      
        self.native_group_lora_enabled = config.native_group_lora_enabled
        self.try_new_version_adapter_setting = getattr(config, 'try_new_version_adapter_setting', False)
        # Initialize weights and apply final processing
        self.post_init()
        self.model.gradient_checkpointing_disable()
        self.gradient_checkpointing_disable()
        self.model.config.gradient_checkpointing = False


    def get_model(self):
        return self.model
    
    def model_temperature(self, global_step, epoch_number):
        if self.wta_params_schedule_mode == 'global_step':
            temperature = self.wta_params_ini_temp * self.wta_params_decay_rate ** global_step
        elif self.wta_params_schedule_mode == 'epoch_number':
            temperature = self.wta_params_ini_temp * self.wta_params_decay_rate ** epoch_number
        else:
            raise ValueError(f"Invalid wta_params_schedule_mode: {self.wta_params_schedule_mode}")
        return max(temperature, self.wta_params_fin_temp)
    
    def _set_adapter(self, adapter_name):
        #if hasattr(self, "enable_adapters"): #and self.enable_adapters():
        # if not self._hf_peft_config_loaded:
        #     self._hf_peft_config_loaded = True # Avoid error since we are using a custom version of PEFT

        # # Patch the MIN_PEFT_VERSION temporarily to bypass the version check
        # import transformers.integrations.peft as peft_integration
        # peft_integration.MIN_PEFT_VERSION = "0.0.0"
        # peft_integration.is_peft_available = lambda: True
        
        # self.set_adapter(adapter_name)

        if isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # For backward compatbility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def forward(
        self,
        hypothesis_idx: int = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        return_all_hyps: Optional[bool] = False,
        return_all_losses: Optional[bool] = False,
        global_step: Optional[int] = None,
        epoch_number: Optional[int] = None,
        wta_training_mode: Optional[str] = None,
        num_items_in_batch: Optional[int] = None,
        cache_position: Optional[torch.FloatTensor] = None, #FIXME: MAKE SURE IT DOES NOT CHANGE THE OUTPUT OTHERWISE DOWNGRADE TRANSFORMERS
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        if wta_training_mode is None:
            wta_training_mode = self.wta_training_mode

        if hypothesis_idx is None:
            list_hyps = range(self.num_hyps)
        else:
            list_hyps = [hypothesis_idx]
        if self.native_group_lora_enabled and len(list_hyps) > 1:
            original_batch_size = input_ids.shape[0]
        else:
            try:
                original_batch_size = inputs_embeds.shape[0]
            except:
                original_batch_size = input_ids.shape[0]

        logits_dict = {}
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        outs=[]

        for hyp in list_hyps:
            if len(list_hyps) > 1:
                self._set_adapter('lora'+str(hyp))

            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            outs.append(outputs)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_hyps, -1)
            attention_mask = attention_mask.reshape(-1, *attention_mask.shape[2:]) # shape [batch*adapter_count, in_features,seq]'

        # Handle past_key_values as a tuple
        if outs[0].past_key_values is not None:
            past_key_values = outs[0].past_key_values
            # If past_key_values is a tuple, we need to stack each element separately
            pass

        else:
            past_key_values = None

        hidden_states = None
        if outs[0].attentions is not None:
            attentions = torch.stack([outs[h].attentions for h in range(len(list_hyps))]).flatten(0,1)
        else:
            attentions = None
        logits = torch.stack([outs[h].logits for h in range(len(list_hyps))]).flatten(0,1)

        if self.native_group_lora_enabled and len(list_hyps) > 1:
            # outputs.logits shape [batch*adapter_count, seq, vocab_size]
            for hypothesis_idx in list_hyps:
                logits_dict[hypothesis_idx] = logits.reshape(original_batch_size, self.num_hyps, outputs[1].shape[-2], outputs[1].shape[-1])[:,hypothesis_idx,:,:]
                # shape [batch, seq, vocab_size]

            attention_mask = attention_mask.reshape(original_batch_size, self.num_hyps, attention_mask.shape[-1])[:,0,:]
        else :
            logits_dict[list_hyps[0]] = logits

        
        loss = None
        try:
            batch_size = logits_dict[list_hyps[0]].shape[0]
        except:
            breakpoint()


        if labels is not None:
            losses = []
            for h in range(len(list_hyps)):
                shift_logits = outs[h].logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten
                batch, seq_len, vocab_size = shift_logits.shape
                flattened_logits = shift_logits.view(-1, vocab_size)
                flattened_labels = shift_labels.view(-1)
                
                # Calculate loss for this hypothesis
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
                loss = loss_fct(flattened_logits, flattened_labels)
                loss = loss.view(batch_size, -1)
                loss = loss.float().sum(dim=1)/(loss!=0).float().sum(dim=1)
                losses.append(loss)

            # Stack the losses
            loss = torch.stack(losses)  # shape: [num_hyps, batch_size]
            total_loss=loss
            if wta_training_mode == 'wta':
                wta_loss = torch.min(loss, dim=0).values
                loss=wta_loss.mean()
                
            elif wta_training_mode == 'relaxed-wta':
                wta_loss = torch.min(loss, dim=0).values # shape (batch_size)
                epsilon = self.wta_params_epsilon
                loss = (1 - epsilon - epsilon/(self.num_hyps - 1)) * wta_loss + (epsilon/(self.num_hyps - 1)) * loss.sum(dim = 0) # shape (batch_size)
                loss = loss.mean()
            elif wta_training_mode == 'annealed-wta':
                temperature = self.model_temperature(global_step=self.global_step, epoch_number=epoch_number)
                if temperature <= self.wta_params_fin_temp:
                    # Go back to the original loss in this case
                    loss = torch.min(loss, dim=0).values.mean()#loss = loss.mean()
                else :
                    weights = torch.softmax(-loss / temperature, dim=0)
                    weights = weights.detach()
                    loss = (loss * weights).sum(dim = 0)
                    loss = loss.mean()
 
                self.global_step+=1
                wta_loss=loss

            else:
                raise ValueError(f"Invalid wta_training_mode: {self.wta_training_mode}")

        # TODO: Handle inference
        if return_dict:
            # Concatenate the logits_dict into the axis 0
            output = (logits_dict[list_hyps[-1]],) + outputs[1:]
            if loss is not None:
                return (loss,) + output
            else: 
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits_dict[list_hyps[-1]],
                    past_key_values=past_key_values,
                    hidden_states=hidden_states,
                    attentions=attentions
                )


        if return_all_hyps is True:
            logits = torch.stack([logits_dict[hypothesis_idx] for hypothesis_idx in list_hyps], dim=1)
        else:
            logits = logits_dict[list_hyps[-1]]
        
        r=CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions
        )
        r.total_loss = total_loss
        r.wta_loss=wta_loss
        return r

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        hypothesis_idx: Optional[int] = 0,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
            
        # Set the appropriate adapter for generation
        if self.native_group_lora_enabled:
            # Save current adapter
            old_adapter = self.model.layers[0].self_attn.q_proj.active_adapter
            adapter_name = f"lora{hypothesis_idx}"
            oldHyps = copy.deepcopy(self.num_hyps)
            self.num_hyps = 1
            self._set_adapter(adapter_name)
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        out= super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        self.num_hyps = oldHyps
        # Restore original adapter if it was changed
        self._set_adapter(old_adapter)
        return out

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM_MCL)
