#%%
import pandas as pd
import yaml
import os
import numpy as np
import warnings
import argparse
import pandas as pd
import re
import math
from matplotlib.lines import Line2D
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

results_dir = os.environ['PROJECT_ROOT']

global dataset_name
global cross_dataset_mode
global recompute_nll
global compute_mean_score_nll

#%%

def get_marker_and_color(N, config_type, markers, method_colors, num_hyp):
    # Determine marker based on hyp count and method type
    marker = None
    color = None
    alpha = 0.8

    # Check if it's 1-hyp or 5-hyp
    is_one_hyp = 'lora-mle' in config_type.lower()
    is_n_hyp = 'lora-mcl' in config_type.lower()
    is_search = 'bs' in config_type.lower()
    is_diverse_beam = 'dbs' in config_type.lower()
    is_tta = 'tta' in config_type.lower()
    is_moe = 'moe' in config_type.lower()
    is_random = 'random' in config_type.lower()
    
    # Assign marker
    if is_moe:
        if 'stochastic router bs' in config_type.lower():
            marker = markers['MoE-stochastic-router']
        elif 'expert specific bs' in config_type.lower():
            marker = markers['MoE-expert-specific']
        elif 'dbs' in config_type.lower():
            marker = markers['MoE-diverse-beam-search']
        elif 'bs' in config_type.lower():
            marker = markers['MoE-beam-search']
    elif is_one_hyp:
        if is_search:
            if 'r=8' in config_type:
                marker = markers['1_hyp_search_rank8']
            elif f'r={N*8}' in config_type:
                marker = markers['1_hyp_search_rank40']
            else:
                marker = markers['1_hyp_search_rank8']  # default to rank 8 marker
        elif is_tta:
            marker = markers['1_hyp_search_tta']
        else:
            marker = markers['1_hyp_sampling']
    elif is_random:
        marker = markers['random']
    else:  # 5-hyp
        marker = markers['5_hyp_sampling_eps'] if not is_search else markers['5_hyp_search_eps']

    if 'temperature sampling' in config_type.lower():
        color = method_colors['temperature']
    elif 'tta' in config_type.lower():
        color = 'darkorange'
    elif 'top-k sampling' in config_type.lower():
        color = method_colors['top_k']
    elif 'nucleus sampling' in config_type.lower():
        color = method_colors['nucleus']
    elif 'dbs' in config_type.lower():
        color = method_colors['diverse_beam']
    elif 'bs' in config_type.lower():
        if 'epsilon' in config_type.lower():
            eps_val = extract_epsilon(config_type)
            if eps_val < 0.:
                color = method_colors['annealed']
            else:
                color = method_colors['beam_search']
        else:
            if 'stochastic router bs' in config_type.lower():
                color = method_colors['stochastic_router']
            elif 'expert specific bs' in config_type.lower():
                color = method_colors['expert_specific']
            else:
                color = method_colors['beam_search']
    elif 'rephraser' in config_type.lower():
        color = method_colors['rephraser']

    def sizes(beam_size):
        return 100 + beam_size * 30

    beam_size = extract_params(config_type, 'beam')

    # Calculate size based on beam size
    if beam_size is not None:
        # Base size of 100, scale up by 20 for each beam size unit
        size = sizes(beam_size)
    else:
        size = 120

    # Adjust alpha based on parameters
    if 'moe' in config_type.lower():
        alpha = 0.5
        if 'stochastic router' in config_type.lower() or 'expert specific' in config_type.lower() :
            size = sizes(beam_size*num_hyp)
        else:
            size = sizes(beam_size)
    elif 'tta' in config_type.lower():
        freq = extract_params(config_type, 'freq')
        time = extract_params(config_type, 'time')
        num_hyp = num_hyp
        mean = (freq + time)/2
        # alpha = 0.4 + (mean / 0.99) * 0.6
        alpha = 0.5
        size = sizes(beam_size*num_hyp)
    elif 'gauss' in config_type:
        white_noise = extract_params(config_type, 'white_noise')
        num_hyp = num_hyp
        alpha = 0.4 + (white_noise / 0.95) * 0.6
        size = sizes(beam_size*num_hyp)
    elif 'beam=' in config_type and 'lambda=' not in config_type and 'epsilon=' not in config_type:
        alpha = 0.5
    elif 'lambda=' in config_type:
        lambda_val = extract_params(config_type, 'lambda')
        # alpha = 0.4 + (lambda_val / 3.0) * 0.6  # Scale alpha with lambda
        alpha = 0.5
    elif 'epsilon' in config_type:
        # num_hyp = extract_num_hypotheses(config_type)
        num_hyp = num_hyp
        eps_val = extract_epsilon(config_type)
        if eps_val >= 0.:
            alpha = 0.5
            # alpha = 0.4+(eps_val / 0.1)*0.6  # Scale alpha with epsilon
        else:
            # alpha = 0.4
            alpha = 0.5
        size = sizes(beam_size*num_hyp)

    # Add fillstyle based on epsilon or rank
    fillstyle = 'full'  # default
    
    if 'eps=0.05' in config_type:
        fillstyle = 'left'
    elif 'eps=0.1' in config_type:
        fillstyle = 'right'
    elif 'eps=0.01' in config_type:
        fillstyle = 'bottom'
    elif 'eps=0.02' in config_type:
        fillstyle = 'top'
    elif 'eps=0.2' in config_type:
        fillstyle = 'bottom'
    elif 'rank 8' in config_type:
        fillstyle = 'full'
    elif f'rank {N*8}' in config_type:
        fillstyle = 'full'
    
    # Create marker with fillstyle
    marker = MarkerStyle(marker, fillstyle=fillstyle)

    return marker, color, alpha, size

def extract_lambda(config_type):
    match = re.search(r'\\lambda=(\d+\.?\d*)', config_type)
    if match:
        return float(match.group(1))
    return None

def extract_epsilon(config_type):
    """
    Extract epsilon value from a config type string.
    
    Args:
        config_type (str): String like '5 hyp $\\varepsilon=0.01$ (beam search, beam=2)'
        
    Returns:
        float or None: The epsilon value if found, None otherwise
    """
    # Match pattern: \varepsilon=X where X is a number (integer or float)
    # match = re.search(r'\\varepsilon=(\d+\.?\d*)', config_type)
    match = re.search(r'\\varepsilon=([+-]?\d+\.?\d*)', config_type)
    if match:
        return float(match.group(1))
    return None

def extract_rank(config_type):
    """
    Extract rank value from a config type string.
    
    Args:
        config_type (str): String like 'LoRA-MLE ($r = 8$, dbs, $\\lambda=0.8$)'
        
    Returns:
        float or None: The rank value if found, None otherwise
    """
    # Match pattern: $r = X$ where X is a number
    match = re.search(r'\$r\s*=\s*(\d+\.?\d*)\$', config_type)
    if match:
        return float(match.group(1))
    return None

def extract_training_method(config_type):
    # First match the base method (LoRA-MLE or LoRA-MCL)
    base_match = re.search(r'LoRA-MLE|LoRA-MCL|LoRA-MoE|LoRA-Random', config_type)
    if not base_match:
        return None
        
    base_method = base_match.group(0)
    
    # For LoRA-MLE, extract the rank
    if base_method == 'LoRA-MLE':
        rank_match = re.search(r'\$r\s*=\s*(\d+)\$', config_type)
        if rank_match:
            rank = rank_match.group(1)
            return f"\\texttt{{{base_method}}} ($r = {rank}$)"
    
    # For LoRA-MCL, extract the epsilon
    elif base_method == 'LoRA-MCL':
        eps_match = re.search(r'\\varepsilon=([+-]?\d+\.?\d*)', config_type)
        if eps_match:
            eps = eps_match.group(1)
            if float(eps) < 0:
                return f"\\texttt{{{base_method}}} (annealed)"
            return f"\\texttt{{{base_method}}} ($\\varepsilon={eps}$)"

    elif base_method == 'LoRA-MoE':
        return f"\\texttt{{{base_method}}}"

    elif base_method == 'LoRA-Random':
        eps_match = re.search(r'\\varepsilon=([\d.]+)', config_type)
        if eps_match:
            eps = eps_match.group(1)
            return f"\\texttt{{{base_method}}} ($\\varepsilon={eps}$)"
    
    return base_method

def extract_decoding_method(config_type):
    # First check for diverse beam search
    if 'dbs' in config_type:
        # Extract lambda value
        lambda_match = re.search(r'lambda=(\d+\.?\d*)', config_type)
        if lambda_match:
            lambda_val = lambda_match.group(1)
            return f'DBS ($\\lambda={lambda_val}$)'
        return 'DBS'
    # Then check for regular beam search
    elif 'bs' in config_type:
        if 'tta' in config_type:
            time_mask_percentage_match = re.search(r'\\tau=(\d+\.?\d*)', config_type)
            freq_mask_percentage_match = re.search(r'f=(\d+\.?\d*)', config_type)
            if time_mask_percentage_match and freq_mask_percentage_match:
                time_mask_percentage = time_mask_percentage_match.group(1)
                freq_mask_percentage = freq_mask_percentage_match.group(1)
                if time_mask_percentage == '0.2' and freq_mask_percentage == '0.3':
                    return f'TTA BS ($\\phi_{1}$)'
                elif time_mask_percentage == '0.4' and freq_mask_percentage == '0.6':
                    return f'TTA BS ($\\phi_{2}$)'
                elif time_mask_percentage == '0.6' and freq_mask_percentage == '0.9':
                    return f'TTA BS ($\\phi_{3}$)'
                elif time_mask_percentage == '0.05' and freq_mask_percentage == '0.075':
                    return None
                # else:
                return None
                # return f'TTA BS ($\\tau={time_mask_percentage}, f={freq_mask_percentage}$)'
            else:
                return 'TTA BS'
        elif 'stochastic' in config_type:
            return 'SR BS'
        elif 'expert' in config_type:
            return 'ES BS'
        else:
            return 'BS'
    elif 'stochastic' in config_type:
        return 'Stochastic Router BS'
    elif 'expert' in config_type:
        return 'Expert Specific BS'
    elif 'tta' in config_type:
        return 'TTA'
    elif 'top-k' in config_type:
        return 'Top-$k$ sampling'
    elif 'nucleus' in config_type:
        return 'Nucleus (Top-$p$) sampling'
    elif 'typical' in config_type:
        return 'Typical $p$ sampling'
    # If neither is found, return None
    else:
        return None

def extract_params(config_type, param_name):
    # Match 'param_name=X' where X is a number (integer or float), followed by either ',' or ')'
    match = re.search(f'{param_name}=(\d+\.?\d*)(?:,|\))', config_type)
    if match:
        return float(match.group(1))
    return None

def extract_num_hypotheses(config_type):
    """
    Extract number of hypotheses from a config type string.
    
    Args:
        config_type (str): String like '5 hyp $\\varepsilon=0.05$ (beam search, beam=1)'
        
    Returns:
        int or None: The number of hypotheses if found, None otherwise
    """
    # Match pattern: X hyp where X is a number
    match = re.search(r'(\d+)\s*hyp', config_type)
    if match:
        return int(match.group(1))
    return None

def compute_mean_NLL_scoreV2(scores):
    # Loop over hypotheses_0, ..., hypotheses_N
    num_hypotheses = 0
    while f'hypothesis_{num_hypotheses}_0' in scores[list(scores.keys())[0]].keys():
        num_hypotheses += 1
    num_examples = 0
    while f'hypothesis_0_{num_examples}' in scores[list(scores.keys())[0]].keys():
        num_examples += 1
    
    lprobs_list = []
    for hypothesis_idx in range(num_hypotheses):
        hypothesis_key = f"hypothesis_{hypothesis_idx}"
        lprobs = []
        for key in scores.keys():
            lprobs.append([scores[key][f'hypothesis_{hypothesis_idx}_{i}'] for i in range(num_examples)])
        # stack them to get a tensor of shape (num_examples, num_captions_per_example)
        lprobs = torch.tensor(lprobs)  # shape: (num_keys, num_examples)
        lprobs_list.append(lprobs)

    stacked_lprobs = torch.stack(lprobs_list) # shape: (num_hypotheses, num_examples, num_captions_per_example)
    num_captions_per_example = stacked_lprobs.shape[2]
    oracle_nll = stacked_lprobs.min(dim=0).values.mean()
    return oracle_nll

def get_oracle_mean_nll(name, dataset_name, list_folders):
        if "/lustre/" in name:
            name = name.split("/lustre/")[0]
        good_name = [e for e in list_folders if name in e]
        if len(good_name) == 0:
            return None
        path_folder = good_name[-1]
        if compute_mean_score_nll is False:
            if dataset_name == 'clotho':
                yaml_file = os.path.join(path_folder, 'metrics_coco_oracle_clotho_eval.yaml')
            else:
                yaml_file = os.path.join(path_folder, 'metrics_coco_oracle_audiocaps_test.yaml')
            if not os.path.exists(yaml_file):
                return None
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
            return yaml_content['oracle_mean_nll']
        else: 
            if dataset_name == 'clotho':
                yaml_file = os.path.join(path_folder, 'metrics_coco_full_scores_oracle_clotho_eval.yaml')
            else:
                yaml_file = os.path.join(path_folder, 'metrics_coco_full_scores_oracle_audiocaps_test.yaml')
            if not os.path.exists(yaml_file):
                return None
            with open(yaml_file, 'r') as f:
                yaml_content = yaml.safe_load(f)
            return compute_mean_NLL_scoreV2(yaml_content['oracle_mean_nll_full_scores'])

def bold_extreme_values(dframe, mode="max", precision=3):
    """
    Return a DataFrame where the max or min value in each column is bolded.
    
    Parameters:
    - df: pd.DataFrame of numeric values
    - mode: "max" or "min"
    - precision: number of decimals for float formatting
    """
    df_fmt = dframe.copy()
    for col in dframe.columns:
        if not np.issubdtype(dframe[col].dtype, np.number):
            continue  # skip non-numeric columns

        if "mBLEU" in col or "NLL" in col:
            mode = "min"
        else:
            mode = "max"

        # Find the extreme (max or min)
        extreme = dframe[col].max() if mode == "max" else dframe[col].min()

        if mode == "max":
            second_extreme = dframe[col].sort_values(ascending=False).iloc[1]
        else:
            second_extreme = dframe[col].sort_values(ascending=True).iloc[1]

        # Format with bold
        def format_val(x):
            val_str = f"{x:.{precision}f}"
            if x == extreme:
                return f"\\textbf{{{val_str}}}"
            elif x == second_extreme:
                return f"\\underline{{{val_str}}}"
            else:
                return val_str

        df_fmt[col] = dframe[col].apply(format_val)
    
    return df_fmt

def display_latex_table(latex_table, caption):    
    # Add necessary LaTeX packages and formatting
    full_latex = """
    \\begin{table}
    \\caption{%s}
    \\begin{center}
    \\resizebox{\columnwidth}{!}{
        %s
    }
    \\end{center}
    \\end{table}
    """ % (caption, latex_table)
    
    return full_latex

def determine_config_type(row, N, list_epsilon, show_other_moe=False, show_wta_random=False, show_tta=False):
    # Extract folder name from artifact_uri
    params = row.to_dict()
    
    # Default patterns to look for in params
    is_one_hyp = False
    is_n_hyp = False
    beam_size = params.get('model.generation_config.beam_size', 0)
    repetition_penalty = params.get('model.generation_config.repetition_penalty', None)

    # Check for 1-hyp vs N-hyp
    if 'model.num_hyps' in params:
        num_hyps = params.get('model.num_hyps')
        if num_hyps == 1:
            is_one_hyp = True
        elif num_hyps == N:
            is_n_hyp = True

    if 'model.use_moe_lora' in params and params['model.use_moe_lora'] is True:
        is_moe= True
    else:
        is_moe = False

    if is_moe:
        beam_size = params.get('model.generation_config.beam_size')
        beam_size = int(beam_size) if beam_size is not None else None
        stochastic_router = str(params.get('model.stochastic_router'))
        expert_specific = str(params.get('model.expert_specific'))
        lambda_dbs = params.get('model.generation_config.diversity_penalty', 0)
        num_beam_groups = params.get('model.generation_config.num_beam_groups', 1)
        init_zero_router = str(params.get('model.init_zero_router'))

        if init_zero_router.lower() == 'false':
            return None, beam_size

        if show_other_moe:
            if stochastic_router == 'True' and expert_specific == 'False':
                return f"N-hyp-MoE-stochastic-router-beam{beam_size}", beam_size
            elif stochastic_router == 'False' and expert_specific == 'True':
                return None, beam_size
                # return f"N-hyp-MoE-expert-specific-beam{beam_size}", beam_size
            elif stochastic_router == 'False' and expert_specific == 'False':
                if lambda_dbs > 0 and num_beam_groups == N:
                    if lambda_dbs == 0.5:
                        return None, beam_size
                    return f"1-hyp-MoE-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}", beam_size
                else:
                    return f"1-hyp-MoE-beam-search-beam{beam_size}", beam_size
        else:
            if stochastic_router == 'False' and expert_specific == 'False':
                if lambda_dbs > 0 and num_beam_groups == N:
                    if lambda_dbs == 0.5:
                        return None, beam_size
                    return f"1-hyp-MoE-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}", beam_size
                else:
                    return f"1-hyp-MoE-beam-search-beam{beam_size}", beam_size

    # For 1-hyp configurations
    if is_one_hyp:
        if 'model.generation_config.num_beam_groups' in params and params['model.generation_config.num_beam_groups'] == 25:
            return None, beam_size
        
        if 'model.generation_config.diversity_penalty' in params and params['model.generation_config.diversity_penalty'] > 1.0:
            return None, beam_size

        if 'ckpt_path' in params and params['ckpt_path'] is not None:
            rank = None
            for possible_rank in [8,16,24,32,40]:
                if 'ckpt_path' in params and type(params['ckpt_path']) == str and f'rank-{possible_rank}' in params['ckpt_path']:
                    rank = possible_rank
                    break
        else :
            rank = params.get('native_lora_r')
        tta_enabled = params.get('model.tta_enabled', False)
        do_sample = params.get('model.generation_config.do_sample')
        beam_size = params.get('model.generation_config.beam_size')
        beam_size = int(beam_size) if beam_size is not None else None
        num_beam_groups = params.get('model.generation_config.num_beam_groups', 1)
        num_return_sequences = params.get('model.generation_config.num_return_sequences')
        lambda_dbs = params.get('model.generation_config.diversity_penalty', 0)
        alpha = params.get('model.generation_config.penalty_alpha', 0.)
        top_k = params.get('model.generation_config.top_k', 50)
        top_p = params.get('model.generation_config.top_p', 1.0)
        typical_p = params.get('model.generation_config.typical_p', 1.0)
        tta_enabled = params.get('model.tta_enabled', False)

        if math.isnan(alpha):
            alpha = 0.

        if tta_enabled is True :
            if show_tta is True:
                tta_mode = params.get('model.tta_mode')
                greedy_sh_tta = params.get('model.greedy_sh_tta', None)
                if greedy_sh_tta=='Single':
                    beam_size = 1
                if beam_size == 5:
                    a = 1
                if tta_mode == "spec_aug" or tta_mode == "spec_augment":
                    freq_mask_percentage = params.get('model.freq_mask_percentage', None)
                    time_mask_percentage = params.get('model.time_mask_percentage', None)
                    if (str(time_mask_percentage) != '0.4' or str(freq_mask_percentage) != '0.6') : #or (beam_size != 5):
                        return None, None
                    if do_sample == False:
                        return f"1-hyp-rank{rank}-tta-specaugment-beam{beam_size}-freq{freq_mask_percentage}-time{time_mask_percentage}", beam_size
                elif tta_mode == "gauss":
                    return None, None
                else:
                    # Raise error
                    raise ValueError(f"Unknown TTA mode: {tta_mode}")

            else:
                return None, None
                
        # 1-hyp with rank 8
        elif rank == 8:
            if do_sample and num_return_sequences == N and alpha == 0.:
                if top_k == 0 and top_p == 1.0 and typical_p < 1.0:  # Typical p sampling
                    return f"1-hyp-rank8-typical-p-sampling", beam_size
                elif top_k == 0 and top_p < 1.0 and typical_p == 1.0:  # Nucleus sampling
                    return f"1-hyp-rank8-nucleus-sampling", beam_size
                elif top_k == 50:  # Temperature sampling
                    return f"1-hyp-rank8-top-k-sampling", beam_size
                else:  # Temperature sampling
                    return f"1-hyp-rank8-temp-sampling", beam_size
            elif not do_sample and beam_size is not None and num_beam_groups == 1 and num_return_sequences == N:
                return f"1-hyp-rank8-beam-search-beam{beam_size}", beam_size
            elif not do_sample and beam_size is not None and num_beam_groups == N and num_return_sequences == N and lambda_dbs > 0:
                return f"1-hyp-rank8-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}", beam_size

        # 1-hyp with rank 8*N
        elif rank == 8 * N:
            if do_sample and num_return_sequences == N and alpha == 0.:
                if top_k == 0 and top_p == 1.0 and typical_p < 1.0:  # Typical p sampling
                    return f"1-hyp-rank8N-typical-p-sampling", beam_size
                elif top_k == 0 and top_p < 1.0 and typical_p == 1.0:  # Nucleus sampling
                    return f"1-hyp-rank8N-nucleus-sampling", beam_size
                else:  # Temperature sampling
                    return f"1-hyp-rank8N-temp-sampling", beam_size
            elif not do_sample and beam_size is not None and num_beam_groups == 1 and num_return_sequences == N:
                return f"1-hyp-rank8N-beam-search-beam{beam_size}", beam_size
            elif not do_sample and beam_size is not None and num_beam_groups == N and num_return_sequences == N and lambda_dbs > 0:
                return f"1-hyp-rank8N-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}", beam_size

    # For N-hyp configurations
    if is_n_hyp:
        rank = params.get('model.native_lora_r')
        ckpt_path = params.get('ckpt_path')
        lambda_dbs = params.get('model.generation_config.diversity_penalty', 0)
        alpha = params.get('model.generation_config.penalty_alpha', 0)
        top_k = params.get('model.generation_config.top_k', 50)
        top_p = params.get('model.generation_config.top_p', 1.0)
        typical_p = params.get('model.generation_config.typical_p', 1.0)

        epsilon = None

        if params.get('ckpt_path') is not None and type(params.get('ckpt_path')) == str and params.get('ckpt_path')!='None':
            if 'epsilon' in params.get('ckpt_path') and 'annealed' not in params.get('ckpt_path'):
                epsilon = float(params.get('ckpt_path').split('epsilon-')[1].split('_')[0])
            elif 'annealed' in params.get('ckpt_path'):
                epsilon = -1.0
        elif np.isnan(params.get('ckpt_path')) or params.get('ckpt_path') is None:
            to_return = [None, beam_size]

        if epsilon not in list_epsilon:
            to_return = [None, beam_size]

        do_sample = params.get('model.generation_config.do_sample')
        beam_size = params.get('model.generation_config.beam_size')
        beam_size = int(beam_size) if beam_size is not None else None

        if math.isnan(alpha):
            alpha = 0.
        
        if rank == 8:
            if do_sample and alpha == 0:
                if top_k == 0 and top_p == 1.0 and typical_p < 1.0:  # Typical p sampling
                    to_return=[f"N-hyp-eps{epsilon}-typical-p-sampling", beam_size]
                elif top_k == 0 and top_p < 1.0 and typical_p == 1.0:  # Nucleus sampling
                    to_return=[f"N-hyp-eps{epsilon}-nucleus-sampling", beam_size]
                elif top_k == 50:  # Temperature sampling
                    to_return=[f"N-hyp-eps{epsilon}-top-k-sampling", beam_size]
            elif not do_sample and beam_size is not None:
                to_return=[f"N-hyp-eps{epsilon}-beam-search-beam{beam_size}", beam_size]

        if type(params.get('ckpt_path')) == str and 'random' in params.get('ckpt_path'):
            if show_wta_random is True:
                to_return[0] += '-random'
            else:
                return None, beam_size
        return (to_return[0], to_return[1])

    return None, beam_size

def display_latex_table_for_dataset(dataset_name, show_sampling=True, num_hyps=5, list_epsilon=[0.0005, 0.05, 0.0], use_long_clotho=True, recompute_nll=False):
    """
    Plot results for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset ('audiocaps' or 'clotho')
        ax (matplotlib.axes.Axes): Axes to plot on
        plot (bool): Whether to create a plot or just return the table
        show_sampling (bool): Whether to include sampling results in the table
    """

    for N in [num_hyps]:

        def determine_config_type(row, N, list_epsilon, show_tta=True):
            # Extract folder name from artifact_uri
            params = row.to_dict()

            # Default patterns to look for in params
            is_one_hyp = False
            is_n_hyp = False
            beam_size = params.get('model.generation_config.beam_size', 0)

            repetition_penalty = params.get('model.generation_config.repetition_penalty', None)

            if repetition_penalty != 1.1:
                return None, beam_size
            
            # Check for 1-hyp vs N-hyp
            if 'model.num_hyps' in params:
                num_hyps = params.get('model.num_hyps')
                if num_hyps == 1:
                    is_one_hyp = True
                elif num_hyps == N:
                    is_n_hyp = True
            
            # For 1-hyp configurations
            if is_one_hyp:   
                if 'model.generation_config.num_beam_groups' in params and params['model.generation_config.num_beam_groups'] == 25:
                    return None, beam_size
                
                if 'model.generation_config.diversity_penalty' in params and params['model.generation_config.diversity_penalty'] > 1.0:
                    return None, beam_size

                if 'ckpt_path' in params and params['ckpt_path'] is not None:
                    rank = None
                    for possible_rank in [8,16,24,32,40]:
                        if 'ckpt_path' in params and type(params['ckpt_path']) == str and f'rank-{possible_rank}' in params['ckpt_path']:
                            rank = possible_rank
                            break
                else :
                    rank = params.get('native_lora_r')
                do_sample = params.get('model.generation_config.do_sample')
                beam_size = params.get('model.generation_config.beam_size')
                beam_size = int(beam_size) if beam_size is not None else None
                num_beam_groups = params.get('model.generation_config.num_beam_groups', 1)
                num_return_sequences = params.get('model.generation_config.num_return_sequences')
                lambda_dbs = params.get('model.generation_config.diversity_penalty', 0)
                alpha = params.get('model.generation_config.penalty_alpha', 0.)
                top_k = params.get('model.generation_config.top_k', 50)
                top_p = params.get('model.generation_config.top_p', 1.0)
                typical_p = params.get('model.generation_config.typical_p', 1.0)
                tta_enabled = params.get('model.tta_enabled', False)

                if math.isnan(alpha):
                    alpha = 0.

                if tta_enabled is True :
                    if show_tta is True:
                        tta_mode = params.get('model.tta_mode')
                        greedy_sh_tta = params.get('model.greedy_sh_tta', None)
                        if greedy_sh_tta=='Single':
                            beam_size = 1
                        if beam_size == 5:
                            a = 1
                        if tta_mode == "spec_aug" or tta_mode == "spec_augment":
                            freq_mask_percentage = params.get('model.freq_mask_percentage', None)
                            time_mask_percentage = params.get('model.time_mask_percentage', None)
                            if (str(time_mask_percentage) != '0.4' or str(freq_mask_percentage) != '0.6') : #or (beam_size != 5):
                                return None, None
                            if do_sample == False:
                                return f"1-hyp-rank{rank}-tta-specaugment-beam{beam_size}-freq{freq_mask_percentage}-time{time_mask_percentage}", beam_size
                        elif tta_mode == "gauss":
                            return None, None
                        else:
                            # Raise error
                            raise ValueError(f"Unknown TTA mode: {tta_mode}")

                    else:
                        return None, None

                # 1-hyp with rank 8
                elif rank == 8:
                    if do_sample and num_return_sequences == N and alpha == 0.:
                        if top_k == 0 and top_p == 1.0 and typical_p < 1.0:  # Typical p sampling
                            return f"1-hyp-rank8-typical-p-sampling", beam_size
                        elif top_k == 0 and top_p < 1.0 and typical_p == 1.0:  # Nucleus sampling
                            return f"1-hyp-rank8-nucleus-sampling", beam_size
                        elif top_k == 50:  # Temperature sampling
                            return f"1-hyp-rank8-top-k-sampling", beam_size
                        else:  # Temperature sampling
                            return f"1-hyp-rank8-temp-sampling", beam_size
                    elif not do_sample and beam_size is not None and num_beam_groups == 1 and num_return_sequences == N:
                        return f"1-hyp-rank8-beam-search-beam{beam_size}", beam_size
                    elif not do_sample and beam_size is not None and num_beam_groups == N and num_return_sequences == N and lambda_dbs > 0:
                        return f"1-hyp-rank8-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}", beam_size

                # 1-hyp with rank 8*N
                elif rank == 8 * N:
                    if do_sample and num_return_sequences == N and alpha == 0.:
                        if top_k == 0 and top_p == 1.0 and typical_p < 1.0:  # Typical p sampling
                            return f"1-hyp-rank8N-typical-p-sampling", beam_size
                        elif top_k == 0 and top_p < 1.0 and typical_p == 1.0:  # Nucleus sampling
                            return f"1-hyp-rank8N-nucleus-sampling", beam_size
                        else:  # Temperature sampling
                            return f"1-hyp-rank8N-temp-sampling", beam_size
                    elif not do_sample and beam_size is not None and num_beam_groups == 1 and num_return_sequences == N:
                        return f"1-hyp-rank8N-beam-search-beam{beam_size}", beam_size
                    elif not do_sample and beam_size is not None and num_beam_groups == N and num_return_sequences == N and lambda_dbs > 0:
                        return f"1-hyp-rank8N-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}", beam_size

            # For N-hyp configurations
            if is_n_hyp:
                rank = params.get('model.native_lora_r')
                ckpt_path = params.get('ckpt_path')
                lambda_dbs = params.get('model.generation_config.diversity_penalty', 0)
                alpha = params.get('model.generation_config.penalty_alpha', 0)
                top_k = params.get('model.generation_config.top_k', 50)
                top_p = params.get('model.generation_config.top_p', 1.0)
                typical_p = params.get('model.generation_config.typical_p', 1.0)

                if params.get('ckpt_path') is not None and type(params.get('ckpt_path')) == str and params.get('ckpt_path')!='None':
                    epsilon = float(params.get('ckpt_path').split('epsilon-')[1].split('_')[0])
                    if 'annealed' in params.get('ckpt_path'):
                        epsilon = -1.0
                else : 
                    epsilon = params.get('model.wta_params_epsilon')

                if epsilon not in list_epsilon:
                    return None, beam_size

                do_sample = params.get('model.generation_config.do_sample')
                beam_size = params.get('model.generation_config.beam_size')
                beam_size = int(beam_size) if beam_size is not None else None

                if math.isnan(alpha):
                    alpha = 0.
                
                if rank == 8:
                    if do_sample and alpha == 0:
                        if top_k == 0 and top_p == 1.0 and typical_p < 1.0:  # Typical p sampling
                            return f"N-hyp-eps{epsilon}-typical-p-sampling", beam_size
                        elif top_k == 0 and top_p < 1.0 and typical_p == 1.0:  # Nucleus sampling
                            return f"N-hyp-eps{epsilon}-nucleus-sampling", beam_size
                        elif top_k == 50:  # Temperature sampling
                            return f"N-hyp-eps{epsilon}-top-k-sampling", beam_size
                    elif not do_sample and beam_size is not None:
                        return f"N-hyp-eps{epsilon}-beam-search-beam{beam_size}", beam_size
            return None, beam_size

        if dataset_name == 'clotho':
            csv_file_dir = os.path.join(results_dir, "results", "saved_csv")
            csv_file = [os.path.join(results_dir, "results", "saved_csv", e) for e in os.listdir(f"{results_dir}/results/saved_csv/") if "clotho" in e and ".csv" in e][-1]
        elif dataset_name == 'audiocaps':
            csv_file_dir = os.path.join(results_dir, "results", "saved_csv")
            csv_file = [os.path.join(results_dir, "results", "saved_csv", e) for e in os.listdir(f"{results_dir}/results/saved_csv/") if "audiocaps" in e and ".csv" in e][-1]
        else:
            raise ValueError(f"Dataset name {dataset_name} not supported")

        folder_path = os.path.join(csv_file_dir, csv_file.split('/')[-1].split('_id_')[0])
        print(csv_file)
        df = pd.read_csv(csv_file)
        num_hyps = N
        output_dir = f"{results_dir}/results/saved_csv/latex"

        df = df[df['_status'] == 'FINISHED']
        df = df[(df['model.num_hyps'] == N) | (df['model.generation_config.num_return_sequences'] == N)]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Determine config type for each row
        results = {}
        for _, row in df.iterrows():
            config_type, beam_size = determine_config_type(row, N, list_epsilon)
            if config_type is None:
                continue
            if config_type not in results:
                results[config_type] = []
            row_dict = row.to_dict()
            row_dict['beam_size'] = beam_size  # Add beam_size to the row
            results[config_type].append(row_dict)

        for config_type in results.keys():
            import numpy as np
            idx_most_recent = np.argmax([e['_start_time'] for e in results[config_type]])
            results[config_type] = [results[config_type][idx_most_recent]]

        ### Create a new dataframe with only the relevant raws (ie the ones that belong to a config_type),
        # with a new column that contains the config_type
        results_df = pd.DataFrame()
        for config_type, rows in results.items():
            to_add = pd.DataFrame(rows)
            to_add['config_type'] = config_type

            if recompute_nll:
                to_add['oracle_mean_nll'] = to_add['Name'].apply(lambda x: get_oracle_mean_nll(x, dataset_name, list_folders))
            
            # Extract beam size from rows
            to_add['beam_size'] = to_add['model.generation_config.beam_size'].fillna(0).astype(int)

            # Filter out rows where oracle_mean_nll is None
            if recompute_nll:
                to_add = to_add.dropna(subset=['oracle_mean_nll'])

            results_df = pd.concat([results_df, to_add])

        # Only keep the most recent run for each config_type

        # Order the dataframe with the config_type in this order: (for the raws that exist)
        config_types_order = [
            "1-hyp-rank8-temp-sampling",
            "1-hyp-rank8-top-k-sampling",
            "1-hyp-rank8-beam-search",
            "1-hyp-rank8-diverse-beam-search",
            "1-hyp-rank8N-temp-sampling",
            "1-hyp-rank8N-top-k-sampling",
            "1-hyp-rank8N-beam-search",
            "1-hyp-rank8N-diverse-beam-search"]

        for epsilon in list_epsilon:
            config_types_order.append(f"N-hyp-eps{epsilon}-temp-sampling")
            config_types_order.append(f"N-hyp-eps{epsilon}-top-k-sampling")
            config_types_order.append(f"N-hyp-eps{epsilon}-beam-search")

            # Random versions
            config_types_order.append(f"N-hyp-eps{epsilon}-temp-sampling-random")
            config_types_order.append(f"N-hyp-eps{epsilon}-top-k-sampling-random")
            config_types_order.append(f"N-hyp-eps{epsilon}-beam-search-random")

        for beam_size in [1, 2, 5]:
            config_types_order.append(f"N-hyp-MoE-stochastic-router-beam{beam_size}")
            config_types_order.append(f"N-hyp-MoE-expert-specific-beam{beam_size}")

        for beam_size in [5,10,25]:
            config_types_order.append(f"1-hyp-MoE-beam-search-beam{beam_size}")
            for lambda_dbs in [0.5, 0.8, 1.0]:
                config_types_order.append(f"1-hyp-MoE-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}")

        results_df = results_df.sort_values(by='config_type', key=lambda x: pd.Categorical(x, categories=config_types_order, ordered=True))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            results_df['config_name'] = results_df['Name'].apply(lambda x: x.split('/')[-1])

        # Select the relevant columns

        if dataset_name == 'clotho':
            columns_to_select = ['config_type', 'beam_size', 'clotho_eval_mbleu_4', 'clotho_eval_aac_oracle_spider']
            # columns_to_select = ['config_type', 'beam_size', 'clotho_eval_mbleu_4', 'clotho_eval_aac_oracle_spider', 'clotho_eval_aac_oracle_fense']      
            # columns_to_selectV2 = ['config_type', 'beam_size', 'clotho_eval_mbleu_4', 'clotho_eval_aac_oracle_spider', 'clotho_eval_aac_oracle_fense']
        
        elif dataset_name == 'audiocaps':
            columns_to_select = ['config_type', 'beam_size', 'audiocaps_test_mbleu_4', 'audiocaps_test_aac_oracle_spider']
            # columns_to_select = ['config_type', 'beam_size', 'audiocaps_test_mbleu_4', 'audiocaps_test_aac_oracle_spider', 'audiocaps_test_aac_oracle_fense']      
            # columns_to_selectV2 = ['config_type', 'beam_size', 'audiocaps_test_mbleu_4', 'audiocaps_test_aac_oracle_spider', 'audiocaps_test_aac_oracle_fense']

        if dataset_name == 'clotho':  
            if recompute_nll is False:
                results_df = results_df[[col for col in columns_to_select]]
            else: 
                results_df = results_df[[col for col in columns_to_selectV2]]
        elif dataset_name == 'audiocaps':
            if recompute_nll is False:
                results_df = results_df[[col for col in columns_to_select]]
            else: 
                results_df = results_df[[col for col in columns_to_selectV2]]

        columns_to_render_dict = {
            'diversity_1': 'Div1',
            'diversity_2': 'Div2',
            'mbleu_4': 'mBLEU-4',
            'vocabulary_size': 'Vocab Size',
            'aac_oracle_bert_score.precision': 'BERTScore-P',
            'aac_oracle_bert_score.recall': 'BERTScore-R',
            'aac_oracle_bert_score.f1': 'BERTScore-F1',
            'aac_oracle_bleu_4': 'BLEU-4',
            'aac_oracle_bleu_1': 'BLEU-1',
            'aac_oracle_bleu_2': 'BLEU-2',
            'aac_oracle_bleu_3': 'BLEU-3',
            'aac_oracle_cider_d': 'CIDEr-D',
            'aac_oracle_sbert_sim': 'SBERT-Sim',
            'aac_oracle_fense': 'FENSE',
            'aac_oracle_meteor': 'METEOR',
            'aac_oracle_rouge_l': 'ROUGE-L',
            'aac_oracle_spice': 'SPICE',
            'aac_oracle_spider': 'SPIDEr',
            'acc_oracle_fer': 'FER',
            'oracle_mean_nll': 'Mean NLL'
        }

        new_columns = []

        for col in results_df.columns:
            if 'clotho' in col and col.split('clotho_eval_')[-1] in columns_to_render_dict.keys():
                new_columns.append(columns_to_render_dict[col.split('clotho_eval_')[-1]])
            elif 'audiocaps' in col and col.split('audiocaps_test_')[-1] in columns_to_render_dict.keys():
                new_columns.append(columns_to_render_dict[col.split('audiocaps_test_')[-1]])
            else:
                if col in columns_to_render_dict.keys():
                    new_columns.append(columns_to_render_dict[col])
                else:
                    new_columns.append(col)

        results_df.columns = new_columns

        # alpha = 0.6
        top_k = 8.0
        # lambda_dbs = 1.0

        # Change the name of the raws
        rows_to_render = {
            "1-hyp-rank8-temp-sampling": "LoRA-MLE ($r = 8$, temperature sampling)",
            "1-hyp-rank8-top-k-sampling": "LoRA-MLE ($r = 8$, top-k sampling)",
            "1-hyp-rank8-nucleus-sampling": "LoRA-MLE ($r = 8$, nucleus sampling)",
            "1-hyp-rank8-typical-p-sampling": "LoRA-MLE ($r = 8$, typical p sampling)",
            "1-hyp-rank8N-temp-sampling": f"LoRA-MLE ($r = {8*num_hyps}$, temperature sampling)",
            "1-hyp-rank8N-top-k-sampling": f"LoRA-MLE ($r = {8*num_hyps}$, top-k sampling)",
            "1-hyp-rank8N-nucleus-sampling": f"LoRA-MLE ($r = {8*num_hyps}$, nucleus sampling)",
            "1-hyp-rank8N-typical-p-sampling": f"LoRA-MLE ($r = {8*num_hyps}$, typical p sampling)",
        }

        Time_mask = [0.2, 0.4, 0.6]
        Freq_mask =[0.3, 0.6, 0.9]

        COUPLES = [(Time_mask[i], Freq_mask[i]) for i in range(len(Time_mask))]

        for time_mask_percentage, freq_mask_percentage in COUPLES:
            for beam_size in [1, 2, 5]:
                rows_to_render[f"1-hyp-rank8-tta-specaugment-beam{beam_size}-freq{freq_mask_percentage}-time{time_mask_percentage}"] = f"LoRA-MLE ($r=8$) (specaugment tta search, bs, beam={beam_size}, freq={freq_mask_percentage}, time={time_mask_percentage}, num_return={num_hyps})"
                rows_to_render[f"1-hyp-rank40-tta-specaugment-beam{beam_size}-freq{freq_mask_percentage}-time{time_mask_percentage}"] = f"LoRA-MLE ($r=40$) (specaugment tta search, bs, beam={beam_size}, freq={freq_mask_percentage}, time={time_mask_percentage}, num_return={num_hyps})"
        
        for epsilon in list_epsilon:
            rows_to_render[f"N-hyp-eps{epsilon}-temp-sampling"] = f"LoRA-MCL $\\varepsilon={epsilon}$ (temperature sampling)"
            rows_to_render[f"N-hyp-eps{epsilon}-top-k-sampling"] = f"LoRA-MCL $\\varepsilon={epsilon}$ (top-k sampling)"
            rows_to_render[f"N-hyp-eps{epsilon}-nucleus-sampling"] = f"LoRA-MCL $\\varepsilon={epsilon}$ (nucleus sampling)"
            rows_to_render[f"N-hyp-eps{epsilon}-typical-p-sampling"] = f"LoRA-MCL $\\varepsilon={epsilon}$ (typical p sampling)"

        # Add beam search entries dynamically for all possible beam sizes
        for beam_size in range(1, 26):  # Assuming beam sizes 1-10
            rows_to_render[f"N-hyp-MoE-stochastic-router-beam{beam_size}"] = f"LoRA-MoE stochastic router bs beam={beam_size}"
            rows_to_render[f"N-hyp-MoE-expert-specific-beam{beam_size}"] = f"LoRA-MoE expert specific bs beam={beam_size}"
            rows_to_render[f"1-hyp-MoE-beam-search-beam{beam_size}"] = f"LoRA-MoE bs beam={beam_size}"
            rows_to_render[f"1-hyp-rank8-beam-search-beam{beam_size}"] = f"LoRA-MLE ($r=8$) (bs, beam={beam_size})"
            rows_to_render[f"1-hyp-rank8-rephraser-sampling"] = f"LoRA-MLE ($r=8$) (rephraser sampling)"
            rows_to_render[f"1-hyp-rank8N-beam-search-beam{beam_size}"] = f"LoRA-MLE ($r={8*num_hyps}$ (bs, beam={beam_size})"
            for epsilon in list_epsilon:
                rows_to_render[f"N-hyp-eps{epsilon}-beam-search-beam{beam_size}"] = f"LoRA-MCL $\\varepsilon={epsilon}$ (bs, beam={beam_size})"
           
                # Random versions
                rows_to_render[f"N-hyp-eps{epsilon}-beam-search-beam{beam_size}-random"] = f"LoRA-Random $\\varepsilon={epsilon}$ (bs, beam={beam_size}, random)"

            for lambda_dbs in [0.2, 0.5, 0.8, 1.0, 2.0, 2.5, 3.0]:
                rows_to_render[f"1-hyp-MoE-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}"] = f"LoRA-MoE dbs, beam={beam_size}, lambda={lambda_dbs}"
                rows_to_render[f"1-hyp-rank8-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}"] = f"LoRA-MLE ($r=8$) (dbs, beam={beam_size}, lambda={lambda_dbs})"
                rows_to_render[f"1-hyp-rank8N-diverse-beam-search-beam{beam_size}-lambda{lambda_dbs}"] = f"LoRA-MLE ($r={8*num_hyps}$ (dbs, beam={beam_size}, lambda={lambda_dbs})"

        # Drop rows with NaN values in config_type before categorization
        results_df = results_df.dropna(subset=['config_type'])

        # Use simple lookup for config_type to display name
        results_df['config_type'] = results_df['config_type'].apply(lambda x: rows_to_render.get(x, x))

        # Create a category for sorting
        results_df['sort_category'] = 'unknown'
        # First sort by LoRA-MLE vs LoRA-MCL
        results_df.loc[results_df['config_type'].str.contains('LoRA-MLE', case=False, na=False), 'sort_category'] = '1_LoRA-MLE'
        results_df.loc[results_df['config_type'].str.contains('LoRA-MoE', case=False, na=False), 'sort_category'] = '2_LoRA-MoE'
        results_df.loc[results_df['config_type'].str.contains('LoRA-Random', case=False, na=False), 'sort_category'] = '3_LoRA-Random'
        results_df.loc[results_df['config_type'].str.contains('LoRA-MCL', case=False, na=False), 'sort_category'] = '4_LoRA-MCL'
        
        # Then by sampling vs beam search within each category
        results_df.loc[(results_df['sort_category'] == '1_LoRA-MLE') & 
                    (results_df['config_type'].str.contains('sampling', case=False, na=False)), 'sort_category'] = '1_LoRA-MLE_sampling'
        results_df.loc[(results_df['sort_category'] == '1_LoRA-MLE') & 
                    (results_df['config_type'].str.contains('dbs', case=False, na=False)) &
                    (~results_df['config_type'].str.contains('tta', case=False, na=False)), 'sort_category'] = '1_LoRA-MLE_dbs'
        results_df.loc[(results_df['sort_category'] == '1_LoRA-MLE') & 
                    (results_df['config_type'].str.contains('bs', case=False, na=False)) & 
                    (~results_df['config_type'].str.contains('dbs', case=False, na=False)) &
                    (~results_df['config_type'].str.contains('tta', case=False, na=False)), 'sort_category'] = '1_LoRA-MLE_bs'
        results_df.loc[(results_df['sort_category'] == '1_LoRA-MLE') & 
                    (results_df['config_type'].str.contains('tta', case=False, na=False)) , 'sort_category'] = '1_LoRA-MLE_tta'
        results_df.loc[(results_df['sort_category'] == '2_LoRA-MoE') &
        (results_df['config_type'].str.contains('bs', case=False, na=False)) & 
                    (~results_df['config_type'].str.contains('dbs', case=False, na=False)) &
                    (~results_df['config_type'].str.contains('stochastic', case=False, na=False)) &
                    (~results_df['config_type'].str.contains('expert', case=False, na=False)), 'sort_category'] = '2_LoRA-MoE_bs'
        results_df.loc[(results_df['sort_category'] == '2_LoRA-MoE') &
        (results_df['config_type'].str.contains('dbs', case=False, na=False)) & 
                    (~results_df['config_type'].str.contains('stochastic', case=False, na=False)) &
                    (~results_df['config_type'].str.contains('expert', case=False, na=False)), 'sort_category'] = '2_LoRA-MoE_dbs'
        results_df.loc[(results_df['sort_category'] == '2_LoRA-MoE') &
        (results_df['config_type'].str.contains('stochastic', case=False, na=False)) , 'sort_category'] = '2_LoRA-MoE_stochastic'
        results_df.loc[(results_df['sort_category'] == '2_LoRA-MoE') &
        (results_df['config_type'].str.contains('expert', case=False, na=False)) &
                    (~results_df['config_type'].str.contains('stochastic', case=False, na=False)), 'sort_category'] = '2_LoRA-MoE_expert'
        results_df.loc[(results_df['sort_category'] == '3_LoRA-Random') &
        (results_df['config_type'].str.contains('bs', case=False, na=False)) & 
                    (~results_df['config_type'].str.contains('dbs', case=False, na=False)), 'sort_category'] = '3_LoRA-Random_bs'
        results_df.loc[(results_df['sort_category'] == '4_LoRA-MCL') & 
                    (results_df['config_type'].str.contains('sampling', case=False, na=False)), 'sort_category'] = '4_LoRA-MCL_sampling'
        results_df.loc[(results_df['sort_category'] == '4_LoRA-MCL') & 
                    (results_df['config_type'].str.contains('dbs', case=False, na=False)), 'sort_category'] = '4_LoRA-MCL_dbs'
        results_df.loc[(results_df['sort_category'] == '4_LoRA-MCL') & 
                    (results_df['config_type'].str.contains('bs', case=False, na=False)) & 
                    (~results_df['config_type'].str.contains('dbs', case=False, na=False)), 'sort_category'] = '4_LoRA-MCL_bs'

        results_df['lambda'] = results_df['config_type'].apply(extract_lambda)
        results_df['epsilon'] = results_df['config_type'].apply(extract_epsilon)
        results_df['rank'] = results_df['config_type'].apply(extract_rank)
        # Sort the dataframe by the sort category and beam size
        results_df = results_df.sort_values(by=['sort_category', 'beam_size', 'lambda', 'epsilon', 'rank'])

        # Drop the lambda column
        results_df = results_df.drop('lambda', axis=1)
        results_df = results_df.drop('epsilon', axis=1)
        results_df = results_df.drop('rank', axis=1)

        # Filter out sampling results if show_sampling is False
        if not show_sampling:
            results_df = results_df[~results_df['config_type'].str.contains('sampling', case=False, na=False)]

        # Render the beam size column as an integer
        results_df['beam_size'] = results_df['beam_size'].astype(int)

        # Rename the beam_size column to B
        results_df = results_df.rename(columns={'beam_size': 'Beam Size'})

        results_df['Training'] = results_df['config_type'].apply(extract_training_method)
        results_df['Decoding'] = results_df['config_type'].apply(extract_decoding_method)

        # Drop the config_type column
        results_df_copy = results_df.drop('config_type', axis=1)

        # Put the training and decoding columns at the beginning of the dataframe
        results_df_copy = results_df_copy[['Training', 'Decoding'] + [col for col in results_df_copy.columns if col not in ['Training', 'Decoding']]]

        # # Apply bold formatting
        results_df_bold = bold_extreme_values(results_df_copy)

        # Drop the sort column
        results_df_bold = results_df_bold.drop('sort_category', axis=1)

        # Rename config_type to Config
        results_df_bold = results_df_bold.rename(columns={'config_type': 'Config'})

        # Generate basic LaTeX table
        latex_table = results_df_bold.to_latex(index=False)

        # Split the table into lines
        lines = latex_table.split('\n')

        # Find the header line (after the \toprule)
        header_index = None
        for i, line in enumerate(lines):
            if '\\toprule' in line:
                header_index = i
                break

        if header_index is not None:
            # Initialize variables to track categories
            current_category = None
            midrule_positions = []
            
            # Go through each data row and identify category transitions
            for i in range(header_index + 2, len(lines)):
                if '\\bottomrule' in lines[i]:
                    break
                    
                line = lines[i]
                if '\\\\' not in line:
                    continue  # Skip non-data rows
                    
                # Determine the category of the current row
                category = None
                if 'LoRA-MLE' in line:
                    if 'dbs' in line:
                        category = 'LoRA-MLE_dbs'
                    elif 'bs' in line:
                        category = 'LoRA-MLE_bs'
                    else:
                        category = 'LoRA-MLE'
                elif 'LoRA-MCL' in line:
                    if 'dbs' in line:
                        category = 'LoRA-MCL_dbs'
                    elif 'bs' in line:
                        category = 'LoRA-MCL_bs'
                    else:
                        category = 'LoRA-MCL'
                        
                # If we transition to a new category, add a midrule
                if category is not None and current_category is not None and category != current_category:
                    midrule_positions.append(i)
                    
                current_category = category
            
            # Insert midrules at the identified positions (in reverse order to maintain indices)
            for pos in sorted(midrule_positions, reverse=True):
                # Add double midrule between major categories
                lines.insert(pos, '\\midrule\\midrule')
                    
            # Rejoin the lines
            latex_table = '\n'.join(lines)

        caption = f"Results for {dataset_name} with {N} hypotheses"

        if dataset_name == 'clotho':
            caption = f"Results for Clotho with {N} hypotheses"

        a = display_latex_table(latex_table, caption)
        print(a)

        ### Table the table in a txt file
        with open(os.path.join(output_dir, f"table_{dataset_name}.txt"), "w") as f:
            f.write(latex_table)

        return results_df

def plot_results_for_dataset(num_hyps_to_plot, dataset_name, ax, metric_name, quality_metric_name, plot=True,list_epsilon = [0.0005, 0.02, 0.05], legend_is_plotted=False, label_r=False, a=0.1, b=0.1, c=2.5, d=2.5, e=0.05, f=0.05, g=0.06, h=0.004, fontsize_labels=20, recompute_nll=False, show_wta_random=False, results_df=None):

    for N in [num_hyps_to_plot]:

        columns_to_render_dict = {
            'diversity_1': 'Div1',
            'diversity_2': 'Div2',
            'mbleu_4': 'mBLEU-4',
            'vocabulary_size': 'Vocab Size',
            'aac_oracle_bert_score.precision': 'BERTScore-P',
            'aac_oracle_bert_score.recall': 'BERTScore-R',
            'aac_oracle_bert_score.f1': 'BERTScore-F1',
            'aac_oracle_bleu_4': 'BLEU-4',
            'aac_oracle_bleu_1': 'BLEU-1',
            'aac_oracle_bleu_2': 'BLEU-2',
            'aac_oracle_bleu_3': 'BLEU-3',
            'aac_oracle_cider_d': 'CIDEr-D',
            'aac_oracle_sbert_sim': 'SBERT-Sim',
            'aac_oracle_fense': 'FENSE',
            'aac_oracle_meteor': 'METEOR',
            'aac_oracle_rouge_l': 'ROUGE-L',
            'aac_oracle_spice': 'SPICE',
            'aac_oracle_spider': 'SPIDEr',
            'acc_oracle_fer': 'FER',
            'oracle_mean_nll': 'Mean NLL'
        }

        new_columns = []

        for col in results_df.columns:
            if 'clotho' in col and col.split('clotho_eval_')[-1] in columns_to_render_dict.keys():
                new_columns.append(columns_to_render_dict[col.split('clotho_eval_')[-1]])
            elif 'audiocaps' in col and col.split('audiocaps_test_')[-1] in columns_to_render_dict.keys():
                new_columns.append(columns_to_render_dict[col.split('audiocaps_test_')[-1]])
            else:
                if col in columns_to_render_dict.keys():
                    new_columns.append(columns_to_render_dict[col])
                else:
                    new_columns.append(col)

        results_df.columns = new_columns

        if plot is True:

            ############################# PLOT HERE
            results_to_plot = results_df[['config_type', metric_name, quality_metric_name]]

            ####
            B = 1
            # Filter results_to_plot based on beam size
            filtered_results = results_to_plot.copy()

            # Function to extract beam size from config_type
            def extract_beam_size(config_type):
                beam_match = re.search(r'beam=(\d+)', config_type)
                if beam_match:
                    return int(beam_match.group(1))
                return None

            # Filter based on beam size relationship
            is_one_hyp = filtered_results['config_type'].str.contains('LoRA-MLE', case=False, na=False)
            is_n_hyp = ~is_one_hyp

            sampling_mode = False
            search_mode = True
            tta_mode = True
            if sampling_mode:
                filtered_results = results_to_plot[~results_to_plot['config_type'].str.contains('bs')]
            elif search_mode:
                if tta_mode:
                    filtered_results= results_to_plot[results_to_plot['config_type'].str.contains('bs')]

            # Increase font size and use LaTeX
            marker_1_hyp_sampling = 'o'
            marker_5_hyp_sampling_eps005 = 'X'  # X for epsilon = 0.05
            marker_5_hyp_sampling_eps01 = 'P'   # P for epsilon = 0.1
            marker_5_hyp_sampling_eps001 = 'D'  # D for epsilon = 0.01
            marker_5_hyp_sampling_eps002 = 's'  # s for epsilon = 0.02
            marker_5_hyp_sampling_eps02 = 'o'  # o for epsilon = 0.2
            marker_5_hyp_sampling_eps0001 = 'o'  # o for epsilon = 0.001
            marker_5_hyp_sampling_eps0002 = 'o'  # o for epsilon = 0.002
            marker_5_hyp_sampling_eps0005 = 'o'  # o for epsilon = 0.005
            marker_5_hyp_sampling_eps00001 = 'o'  # o for epsilon = 0.0001
            marker_5_hyp_sampling_eps00002 = 'o'  # o for epsilon = 0.0002
            marker_5_hyp_sampling_eps00005 = 'o'  # o for epsilon = 0.0005
            marker_5_hyp_sampling_eps = 'o'  # o for epsilon = 0.05
            marker_5_hyp_sampling_eps0 = 'o'  # o for epsilon = 0.0
            marker_1_hyp_search_rank8 = 'D'     # circle for rank 8
            marker_1_hyp_search_rank40 = 's'    # square for rank 40
            marker_5_hyp_search_eps = 'o'  # D for epsilon = 0.01
            marker_tta = 'D'
            marker_moe = '^'
            marker_random = 'H'

            color_temperature = 'red'
            color_top_k = 'orange'
            color_nucleus = 'goldenrod'
            color_beam_search = 'blue'
            color_diverse_beam = 'green'
            color_moe = 'darkorange'
            color_annealed = 'blue'
            color_stochastic_router = 'cyan'
            color_expert_specific = 'magenta'

            markers = {
                '1_hyp_sampling': marker_1_hyp_sampling,     # circle for 1-hyp sampling
                '5_hyp_sampling_eps': marker_5_hyp_sampling_eps,  # X for 5-hyp with epsilon
                '1_hyp_search_rank8': marker_1_hyp_search_rank8,       # circle for rank 8
                '1_hyp_search_rank40': marker_1_hyp_search_rank40,     # square for rank 40
                '5_hyp_search_eps': marker_5_hyp_search_eps,  # X for 5-hyp search with epsilon
                '1_hyp_search_tta': marker_tta,  # tta for 1-hyp search
                'MoE-stochastic-router': marker_moe,  # o for MoE with stochastic router and beam size 1
                'MoE-expert-specific': marker_moe,  # o for MoE with expert specific and beam size 1
                'MoE-diverse-beam-search': marker_moe,  # o for MoE with diverse beam search and beam size 1 and lambda 0.5
                'MoE-beam-search': marker_moe,  # o for MoE with beam search and beam size 1
                'random': marker_random,  # H for random
            }

            # Define colors for specific methods
            method_colors = {
                'temperature': color_temperature,
                'top_k': color_top_k,
                'nucleus': color_nucleus,          
                'beam_search': color_beam_search,
                'diverse_beam': color_diverse_beam,
                'moe': color_moe,
                'annealed': color_annealed,
                'stochastic_router': color_stochastic_router,
                'expert_specific': color_expert_specific,
            }

            # Create the plot
            # Do not create a new figure here; use the provided ax

            points_by_category = {
                '1_hyp_sampling': [],
                '5_hyp_sampling_eps005': [],
                '5_hyp_sampling_eps01': [],
                '5_hyp_sampling_eps001': [],
                '5_hyp_sampling_eps002': [],
                '5_hyp_sampling_eps02': [],
                '5_hyp_sampling_eps0001': [],
                '5_hyp_sampling_eps0002': [],
                '5_hyp_sampling_eps0005': [],
                '5_hyp_sampling_eps00001': [],
                '5_hyp_sampling_eps00002': [],
                '5_hyp_sampling_eps00005': [],
                '5_hyp_sampling_eps0': [],
                '5_hyp_sampling_eps-1': [],
                '1_hyp_search': [],
                '5_hyp_search_eps005': [],
                '5_hyp_search_eps01': [],
                '5_hyp_search_eps001': [],
                '5_hyp_search_eps002': [],
                '5_hyp_search_eps02': [],
                '5_hyp_search_eps0001': [],
                '5_hyp_search_eps0002': [],
                '5_hyp_search_eps0005': [],
                '5_hyp_search_eps00001': [],
                '5_hyp_search_eps00002': [],
                '5_hyp_search_eps00005': [],
                '5_hyp_search_eps-1': [],
                '5_hyp_search_eps0': [],
                'MoE-stochastic-router': [],
                'MoE-expert-specific': [],
                'MoE-diverse-beam-search': [],
                'MoE-beam-search': []
            }

            for config_type in filtered_results['config_type'].unique():
                mask = filtered_results['config_type'] == config_type
                marker, color, alpha, size = get_marker_and_color(N=N, config_type=config_type, markers=markers, method_colors=method_colors, num_hyp=num_hyps_to_plot)      
                if '\varepsilon=-1.0' in config_type.lower():
                    edgecolors = 'darkred'
                elif 'varepsilon' in config_type.lower():
                    edgecolors = 'black'
                else:
                    edgecolors = None
                scatter = ax.scatter(
                    filtered_results[mask][metric_name],
                    filtered_results[mask][quality_metric_name],
                    label=config_type,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    s=size,
                    edgecolors=edgecolors,
                    linewidths=5 if edgecolors is not None else 1,
                )

                is_moe = 'moe' in config_type.lower()
                is_one_hyp = 'lora-mle' in config_type.lower()
                is_search = 'bs' in config_type.lower()
                
                if is_moe:
                    if 'stochastic router bs' in config_type.lower():
                        category = 'MoE-stochastic-router'
                    elif 'expert specific bs' in config_type.lower():
                        category = 'MoE-expert-specific'
                    elif 'dbs' in config_type.lower():
                        category = 'MoE-diverse-beam-search'
                    elif 'bs' in config_type.lower():
                        category = 'MoE-beam-search'
                elif is_one_hyp:
                    category = '1_hyp_search' if is_search else '1_hyp_sampling'
                else:
                    if '\\varepsilon=0.05' in config_type or 'eps0.05' in config_type:
                        category = '5_hyp_search_eps005' if is_search else '5_hyp_sampling_eps005'
                    elif '\\varepsilon=-1' in config_type or 'eps-1' in config_type:
                        category = '5_hyp_search_eps-1' if is_search else '5_hyp_sampling_eps-1'
                    elif '\\varepsilon=0' in config_type or 'eps0' in config_type:
                        category = '5_hyp_search_eps0' if is_search else '5_hyp_sampling_eps0'
                points_by_category[category].append((scatter, config_type))

            # First collect all text positions and labels
            text_positions = []
            for config_type in filtered_results['config_type'].unique():
                mask = filtered_results['config_type'] == config_type
                x = filtered_results[mask][metric_name].values[0]
                y = filtered_results[mask][quality_metric_name].values[0]
                
                if 'dbs' in config_type.lower():
                    lambda_match = re.search(r'lambda=(\d+\.?\d*)', config_type)
                    if lambda_match:
                        lambda_val = lambda_match.group(1)
                        text_positions.append((x, y, f'$\\lambda={lambda_val}$'))
                elif 'epsilon' in config_type:
                    eps_val = extract_epsilon(config_type)
                    if eps_val is not None:
                        text_positions.append((x, y, f'$\\varepsilon={eps_val}$'))

            # Create custom legend
            legend_elements = []

            legend_elements.append(Line2D([0], [0], marker=marker_5_hyp_search_eps,
                               color='w', markerfacecolor='gray',
                               markersize=20, markeredgecolor='darkred',
                               markeredgewidth=5, label='LoRA-MCL (Annealed)'))

            legend_elements.append(Line2D([0], [0], marker=marker_5_hyp_search_eps,
                                        color='w', markerfacecolor='gray',
                                        markersize=20, markeredgecolor='black',
                                        markeredgewidth=5, label='LoRA-MCL (Relaxed)'))
                                        
            legend_elements.append(Line2D([0], [0], marker=marker_moe, color='w', markerfacecolor='gray', 
                                            markersize=20, label='LoRA-MoE'))
            if show_wta_random is True:
                legend_elements.append(Line2D([0], [0], marker=marker_random, color='w', markerfacecolor='gray', 
                                            markersize=20, label='LoRA-Random'))
            legend_elements.append(Line2D([0], [0], marker=marker_1_hyp_search_rank8, color='w', markerfacecolor='gray', 
                                            markersize=20, label='LoRA-MLE ($r=8$)'))
            if label_r is True:
                legend_elements.append(Line2D([0], [0], marker=marker_1_hyp_search_rank40, color='w', markerfacecolor='gray', 
                                                markersize=20, label=f'LoRA-MLE ($r={N*8}$)'))
                
            ### Draw lines between points associated with 1 and 5 hyp search methods, when the beam_size(1 hyp) = beam_size(5 hyp)*5
            # After the scatter plot but before creating the legend
            # Create a dictionary to store points by configuration type
            points_by_config = {}

            # First pass: collect all points
            for config_type in filtered_results['config_type'].unique():
                mask = filtered_results['config_type'] == config_type
                x = filtered_results[mask][metric_name].values[0]
                y = filtered_results[mask][quality_metric_name].values[0]
                
                # Extract beam size if present
                beam_match = re.search(r'beam=(\d+)', config_type)
                if beam_match:
                    beam_size = int(beam_match.group(1))
                    
                    # Create a key that identifies the configuration type without beam size
                    # For 1-hyp: keep rank and method
                    # For N-hyp: keep epsilon and method
                    if 'LoRA-MLE' in config_type:
                        # Extract rank
                        rank_match = re.search(r'r=(\d+)', config_type)
                        if rank_match:
                            rank = rank_match.group(1)
                            # For diverse beam search, include lambda in the key
                            if 'dbs' in config_type.lower():
                                lambda_match = re.search(r'lambda=(\d+\.?\d*)', config_type)
                                if lambda_match:
                                    lambda_val = lambda_match.group(1)
                                    config_key = f"1_hyp_rank{rank}_diverse_beam_lambda{lambda_val}"
                            elif 'tta' in config_type.lower():
                                freq = extract_params(config_type, 'freq')
                                time = extract_params(config_type, 'time')
                                config_key = f"1_hyp_rank{rank}_specaugment_freq{freq}_time{time}"
                            else:
                                config_key = f"1_hyp_rank{rank}_beam_search"
                    
                    else:
                        # Extract epsilon
                        eps_match = re.search(r'\\varepsilon=([\d.]+)', config_type)
                        if eps_match:
                            eps = eps_match.group(1)
                            # For diverse beam search, include lambda in the key
                            if 'dbs' in config_type.lower():
                                lambda_match = re.search(r'lambda=(\d+\.?\d*)', config_type)
                                if lambda_match:
                                    lambda_val = lambda_match.group(1)
                                    config_key = f"N_hyp_eps{eps}_diverse_beam_lambda{lambda_val}"
                            else:
                                config_key = f"N_hyp_eps{eps}_beam_search"
                    
                    # Store the point
                    if config_key not in points_by_config:
                        points_by_config[config_key] = []
                    points_by_config[config_key].append((x, y, beam_size, config_type))

            # Dictionary to store midpoints for each beam size transition
            beam_transition_midpoints = {
                'beam_search_rank8': {},
                'beam_search_rank40': {},
                'diverse_beam_rank8': {},
                'diverse_beam_rank40': {},
                'N_hyp': {}
            }

            # Second pass: draw lines between points of same configuration
            for config_key, points in points_by_config.items():
                # Sort points by beam size
                # Determine line color based on method
                line_color = 'k'  # default black
                if 'diverse_beam' in config_key:
                    line_color = method_colors['diverse_beam']
                elif 'beam_search' in config_key:
                    line_color = method_colors['beam_search']
                elif 'N_hyp' in config_key:
                    line_color = method_colors['beam_search']  # or any other color you prefer for N-hyp
                
                points.sort(key=lambda x: x[2])  # x[2] is beam_size
                
                # Draw lines between consecutive points
                for i in range(len(points)-1):
                    x1, y1, beam1, config1 = points[i]
                    x2, y2, beam2, config2 = points[i+1]
                    
                    # Only draw line if beam sizes are different
                    if beam1 != beam2:
                        # Draw a dashed line between the points
                        ax.plot([x1, x2], [y1, y2], line_color, alpha=0.3, linewidth=1)
                        
                        # Calculate midpoint
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        
                        # Store midpoint in appropriate baseline group
                        if '1_hyp_rank8_beam_search' in config_key:
                            if f"{beam1}→{beam2}" not in beam_transition_midpoints['beam_search_rank8']:
                                beam_transition_midpoints['beam_search_rank8'][f"{beam1}→{beam2}"] = []
                            beam_transition_midpoints['beam_search_rank8'][f"{beam1}→{beam2}"].append((mid_x, mid_y))
                        elif '1_hyp_rank40_beam_search' in config_key:
                            if f"{beam1}→{beam2}" not in beam_transition_midpoints['beam_search_rank40']:
                                beam_transition_midpoints['beam_search_rank40'][f"{beam1}→{beam2}"] = []
                            beam_transition_midpoints['beam_search_rank40'][f"{beam1}→{beam2}"].append((mid_x, mid_y))
                        elif '1_hyp_rank8_diverse_beam' in config_key:
                            if f"{beam1}→{beam2}" not in beam_transition_midpoints['diverse_beam_rank8']:
                                beam_transition_midpoints['diverse_beam_rank8'][f"{beam1}→{beam2}"] = []
                            beam_transition_midpoints['diverse_beam_rank8'][f"{beam1}→{beam2}"].append((mid_x, mid_y))
                        elif '1_hyp_rank40_diverse_beam' in config_key:
                            if f"{beam1}→{beam2}" not in beam_transition_midpoints['diverse_beam_rank40']:
                                beam_transition_midpoints['diverse_beam_rank40'][f"{beam1}→{beam2}"] = []
                            beam_transition_midpoints['diverse_beam_rank40'][f"{beam1}→{beam2}"].append((mid_x, mid_y))
                        elif 'N_hyp' in config_key:
                            if f"{beam1}→{beam2}" not in beam_transition_midpoints['N_hyp']:
                                beam_transition_midpoints['N_hyp'][f"{beam1}→{beam2}"] = []
                            beam_transition_midpoints['N_hyp'][f"{beam1}→{beam2}"].append((mid_x, mid_y))
                 
            for epsilon in list_epsilon:
                if sampling_mode:
                    legend_elements.append(Line2D([0], [0], marker=marker_5_hyp_sampling_eps, color='w', markerfacecolor='gray', 
                                            markersize=20, label=f'$\\varepsilon={epsilon}$ (sampling)'))
                elif search_mode:
                    legend_elements.append(Line2D([0], [0], marker=marker_5_hyp_search_eps, color='w', markerfacecolor='gray', 
                                            markersize=20, label=f'LoRA-MCL'))
                    break
                else:
                    legend_elements.append(Line2D([0], [0], marker=marker_5_hyp_sampling_eps, color='w', markerfacecolor='gray', 
                                            markersize=20, label=f'$\\varepsilon={epsilon}$ (sampling/search)'))

            # Add alpha explanation
            lw = 10

            # Add color explanations
            if search_mode:
                legend_elements.append(
                    Line2D([0], [0], color='none', marker='', label='Beam Search'))
                legend_elements.append(
                    Line2D([0], [0], color='none', marker='', label='Diverse Beam Search'))
                legend_elements.append(
                    Line2D([0], [0], color='none', marker='', label='TTA'))


            if sampling_mode:
                legend_elements.append(Line2D([0], [0], color=color_temperature, lw=lw, label='Top $k$'))
                legend_elements.append(Line2D([0], [0], color=color_nucleus, lw=lw, label='Nucleus'))

            dataset_to_render = {
                'audiocaps': 'AudioCaps',
                'clotho': 'Clotho'
            }

            # plt.xlabel('Diversity-1', fontsize=25)
            ax.set_xlabel(metric_name + ' $(\\downarrow)$', fontsize=40, labelpad=10)
            ax.set_ylabel(quality_metric_name + ' $(\\uparrow)$', fontsize=40, labelpad=10)
            
            # Increase the fontsize of the tick labels
            ax.tick_params(axis='both', which='major', labelsize=30)

            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            # Add the custom legend
            # plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
            ax.set_title(f'{dataset_to_render[dataset_name]}', fontsize=50, pad=15)
            # ax.set_title(f'{metric_name} vs. SPIDEr on {dataset_to_render[dataset_name]} with {N}-hyps', fontsize=40, pad=20, x=0.55)
            ax.grid(True, linestyle='--', alpha=0.7)
            # Do not add the legend here; return the legend_elements for use in the main block
            return legend_elements, color_beam_search, color_diverse_beam

#%%

def main(display_table, plot_figure):

    use_long_clotho = False
    num_hyps_to_plot_list = [5]
    use_long_clotho = False
    list_epsilon = [0.0005, 0.05, -1.0]

    if display_table:
        for dataset_name in ['audiocaps', 'clotho']:
            results_df = display_latex_table_for_dataset(dataset_name, show_sampling=False, num_hyps=num_hyps_to_plot_list[0], list_epsilon=list_epsilon, use_long_clotho=use_long_clotho)

    if plot_figure:
        plt.rcParams.update({
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    "font.size": 18
                })

        params_dict = {
            '2': {
                'a': 0.5,
                'b': 1.0,
                'c': 2.5,
                'd': 2.5,
                'e': 0.05,
                'f': 0.05,
                'g': 0.03,
                'h': 0.0021,
                'fontsize_labels': 20
            },
            '3': {
                'a': 0.5,
                'b': 1.0,
                'c': 2.5,
                'd': 2.5,
                'e': 0.05,
                'f': 0.05,
                'g': 0.03,
                'h': 0.0024,
                'fontsize_labels': 20
            },
            '4': {
                'a': 0.0,
                'b': 1.0,
                'c': 2.5,
                'd': 2.5,
                'e': 0.05,
                'f': 0.05,
                'g': 0.03,
                'h': 0.007,
                'fontsize_labels': 20
            },
            '5': {
                'clotho': {
                    'a': 0.2, # Initial radius
                    'b': 0.2,
                    'c': 2.0, # Maximum radius
                    'd': 2.0,
                    'e': 0.02, # Radius step
                    'f': 0.02,
                    'g': 0.06, # Minimum distance_y
                    'h': 0.0055, # Minimum distance_x
                    'fontsize_labels': 25
                },
                'audiocaps': {
                    'a': 0.05, # Initial radius
                    'b': 0.05,
                    'c': 2.0, # Maximum radius
                    'd': 2.0,
                    'e': 0.05, # Radius step
                    'f': 0.05,
                    'g': 0.051, # Minimum distance_x
                    'h': 0.009, # Minimum distance_y
                    'fontsize_labels': 25
                }
            }
        }

        color_tta = 'darkorange'
        color_stochastic_router = 'cyan'
        color_expert_specific = 'magenta'
        metric_name='mBLEU-4'
        # quality_metric_name='FENSE'
        quality_metric_name='SPIDEr'
        show_other_moe = False
        show_tta = True
        show_wta_random = False

        for num_hyps_to_plot in num_hyps_to_plot_list:

            if num_hyps_to_plot == 5:
                a = params_dict[str(num_hyps_to_plot)]['audiocaps']['a']
                b = params_dict[str(num_hyps_to_plot)]['audiocaps']['b']
                c = params_dict[str(num_hyps_to_plot)]['audiocaps']['c']
                d = params_dict[str(num_hyps_to_plot)]['audiocaps']['d']
                e = params_dict[str(num_hyps_to_plot)]['audiocaps']['e']
                f = params_dict[str(num_hyps_to_plot)]['audiocaps']['f']
                g = params_dict[str(num_hyps_to_plot)]['audiocaps']['g']
                h = params_dict[str(num_hyps_to_plot)]['audiocaps']['h']
                fontsize_labels = params_dict[str(num_hyps_to_plot)]['audiocaps']['fontsize_labels']
            else:
                a = params_dict[str(num_hyps_to_plot)]['a']
                b = params_dict[str(num_hyps_to_plot)]['b']
                c = params_dict[str(num_hyps_to_plot)]['c']
                d = params_dict[str(num_hyps_to_plot)]['d']
                e = params_dict[str(num_hyps_to_plot)]['e']
                f = params_dict[str(num_hyps_to_plot)]['f']
                g = params_dict[str(num_hyps_to_plot)]['g']
                h = params_dict[str(num_hyps_to_plot)]['h']
                fontsize_labels = params_dict[str(num_hyps_to_plot)]['fontsize_labels']

            fig, axes = plt.subplots(1, 2, figsize=(26, 12))  # 1 row, 2 columns

            legend_elements, color_beam_search, color_diverse_beam = plot_results_for_dataset(results_df=results_df, num_hyps_to_plot=num_hyps_to_plot, dataset_name='audiocaps', ax=axes[0], metric_name=metric_name, quality_metric_name=quality_metric_name, plot=plot_figure, list_epsilon=list_epsilon, legend_is_plotted=False, label_r=True if num_hyps_to_plot != 3 else False, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, fontsize_labels=fontsize_labels)
            
            if num_hyps_to_plot == 5:
                a = params_dict[str(num_hyps_to_plot)]['clotho']['a']
                b = params_dict[str(num_hyps_to_plot)]['clotho']['b']
                c = params_dict[str(num_hyps_to_plot)]['clotho']['c']
                d = params_dict[str(num_hyps_to_plot)]['clotho']['d']
                e = params_dict[str(num_hyps_to_plot)]['clotho']['e']
                f = params_dict[str(num_hyps_to_plot)]['clotho']['f']
                g = params_dict[str(num_hyps_to_plot)]['clotho']['g']
                h = params_dict[str(num_hyps_to_plot)]['clotho']['h']
                fontsize_labels = params_dict[str(num_hyps_to_plot)]['clotho']['fontsize_labels']

            legend_elements, color_beam_search, color_diverse_beam = plot_results_for_dataset(results_df=results_df, num_hyps_to_plot=num_hyps_to_plot, dataset_name='clotho', ax=axes[1], metric_name=metric_name, quality_metric_name=quality_metric_name, plot=plot_figure, list_epsilon=list_epsilon, legend_is_plotted=True, label_r=True if num_hyps_to_plot != 3 else False, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, fontsize_labels=fontsize_labels)

            count_training_methods = 5

            training_methods = legend_elements[:count_training_methods]  # First 5 are training methods with markers
            decoding_methods = legend_elements[count_training_methods+1:]  # Rest are decoding methods with colors
            
            frameon = True
            single_row_mode = False

            if single_row_mode is True:

                # Combine both training and decoding methods into a single list
                all_legend_elements = training_methods + decoding_methods

                # Create a single legend with all entries in one row
                leg = fig.legend(handles=all_legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                                ncol=len(all_legend_elements)+1, fontsize=30, frameon=True)

                # Set the color of the legend text only for decoding methods
                legend_text_colors = [color_beam_search, color_diverse_beam, color_tta]  # Add more if needed
                decoding_start_index = len(training_methods)  # Start coloring from the first decoding method

                legend_text_colors = [color_beam_search, color_diverse_beam]  # Add more if needed
                for text, color in zip(leg.get_texts()[-2:], legend_text_colors):
                    text.set_color(color)

                for text in leg.get_texts():
                    text.set_horizontalalignment('center')

            if single_row_mode is False:

                # Create two separate legends
                leg2 = fig.legend(handles=decoding_methods, loc='upper center', bbox_to_anchor=(0.5, 0.95), 
                                ncol=len(decoding_methods), fontsize=30, frameon=frameon, handletextpad=-1.5) 
                leg1 = fig.legend(handles=training_methods, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
                                ncol=len(training_methods)+2, fontsize=30, frameon=frameon)
                # Set the color of the decoding methods legend text
                legend_text_colors = [color_beam_search, color_diverse_beam, color_tta, color_stochastic_router, color_expert_specific]  # Add more if needed
                for text, color in zip(leg2.get_texts(), legend_text_colors[:len(leg2.get_texts())]):
                    print(text, color)
                    text.set_color(color)

                for text in leg2.get_texts():
                    text.set_horizontalalignment('center')
            
                # Add the first legend back to the figure (matplotlib removes it when creating the second one)
                fig.add_artist(leg1)

            # Add a single, shared legend at the top
            # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=len(legend_elements), fontsize=30)
            plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave more space for the two-row legend
            
            if 0. in list_epsilon :
                plt.savefig(f'{os.path.join(results_dir, "results", "images", f"{metric_name}_vs_{quality_metric_name}_ac_clotho_with_{num_hyps_to_plot}-hyps_wtaincluded.png")}', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'{os.path.join(results_dir, "results", "images", f"{metric_name}_vs_{quality_metric_name}_ac_clotho_with_{num_hyps_to_plot}-hyps.png")}', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # %%

display_table = True
plot_figure = True
main(display_table, plot_figure)
# %%
