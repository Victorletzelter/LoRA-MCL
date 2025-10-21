import os
import pickle
import random
import time
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import logging

from analysis import (
    train_and_save, 
    MarkovChainDataset, 
    custom_collate_fn, 
    build_gpt_model, 
    calculate_stationary_distribution, 
    compute_entropy_monte_carlo,
    plot_transition_matrices_comparison,
    create_results_folder,
    build_hypotheses_list
)
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = logging.getLogger(__name__)

# Constants
DEFAULT_EVAL_SEQUENCES = 1000
DEFAULT_EVAL_SEED = 99
PLOT_STEPS_LIMIT = 45

# Runtime flags
LOAD_PICKLE = True
DO_PLOT = True
DO_SAVE_PLOT = True

def train(my_args, base_path, model_weights=None):
    # Datasets
    train_dataset = MarkovChainDataset(
        n_sequences=my_args['batch_size']*my_args['N_it'],
        seq_len=my_args['seq_len'],
        vocab_size=my_args['vocab_size'],
        transition_matrices=my_args['transition_matrices'],
        generate_sequences_parallel=True,
        generate_on_fly=True,
        n_latent_variables=my_args['n_latent_variables'],
        seed=my_args['seed']
    )
    eval_dataset = MarkovChainDataset(
        n_sequences=DEFAULT_EVAL_SEQUENCES,
        seq_len=my_args['seq_len'],
        vocab_size=my_args['vocab_size'],
        seed=DEFAULT_EVAL_SEED,
        transition_matrices=my_args['transition_matrices'],
        n_latent_variables=my_args['n_latent_variables']
    )

    my_args['vocab_size'] = train_dataset.vocab_size

    # Collator
    data_collator = custom_collate_fn

    steps = {}
    train_losses = {}
    eval_steps = {}
    eval_losses = {}
    avg_model_losses = {}
    model_eval_steps = {}
    markov_model_loss = {}
    unigram_entropy = {}
    optimal_loss = {}
    model_classes = {}

    N_hyps_max = my_args['N_hyps_max']
    list_of_hyps = build_hypotheses_list(N_hyps_max)

    start_time = time.time()

    entropy_rates = {}
    stationary_dict = {}
    losses = {}
    losses_mh = {}
    losses_wta = {}

    # Warm-start model weights if provided via path_pickle
    if model_weights is None and my_args['path_pickle'] is not None:
        with open(os.path.join(my_args['path_pickle'], "results.pkl"), "rb") as f:
            results_dict = pickle.load(f)
        model_weights = results_dict['model_weights']

    # Create results folder
    results_folder = create_results_folder(base_path, my_args)

    for n_hyps in list_of_hyps:
        if my_args['use_latent_variables'] is True:
            steps[n_hyps], train_losses[n_hyps], eval_steps[n_hyps], eval_losses[n_hyps], avg_model_losses[n_hyps], model_eval_steps[n_hyps], model_classes[n_hyps], entropy_rates[n_hyps], stationary_list, losses[n_hyps], losses_mh[n_hyps], losses_wta[n_hyps] = train_and_save(n_hyps=n_hyps, my_args=my_args, plot=False, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, model_weights=model_weights, log_transition_matrices_during_training=my_args['log_transition_matrices_during_training'], results_folder=results_folder)
        else:
            steps[n_hyps], train_losses[n_hyps], eval_steps[n_hyps], eval_losses[n_hyps], avg_model_losses[n_hyps], model_eval_steps[n_hyps], markov_model_loss[n_hyps], stationnary_list, unigram_entropy[n_hyps], optimal_loss[n_hyps], model_classes[n_hyps] = train_and_save(n_hyps=n_hyps, my_args=my_args, plot=False, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, model_weights=model_weights, log_transition_matrices_during_training=my_args['log_transition_matrices_during_training'], results_folder=results_folder)

        log.info(f"Time taken for {n_hyps} hypotheses: {time.time() - start_time:.2f} seconds")

        stationary_dict[n_hyps] = stationary_list

    # Create a dictionary to store all the results
    results_dict = {
        'steps': steps,
        'train_losses': train_losses,
        'eval_steps': eval_steps,
        'eval_losses': eval_losses,
        'model_eval_steps': model_eval_steps,
        'model_weights': {},
        'entropy_rates': entropy_rates[list(entropy_rates.keys())[-1]],
        'stationary_distributions': stationary_dict[list(stationary_dict.keys())[-1]],
    }

    # Save model weights for each number of hypotheses
    for n_hyps in model_classes:
        results_dict['model_weights'][n_hyps] = model_classes[n_hyps].state_dict()

    # Save the results
    with open(os.path.join(results_folder, "results.pkl"), "wb") as f:
        pickle.dump(results_dict, f)

    # Create a yaml file with the config used to generate the results
    # Convert numpy arrays to lists for YAML serialization
    yaml_safe_args = {}
    for key, value in my_args.items():
        if key == 'transition_matrices':
            yaml_safe_args[key] = [matrix.tolist() for matrix in value]
        elif isinstance(value, np.ndarray):
            yaml_safe_args[key] = value.tolist()
        else:
            yaml_safe_args[key] = value

    with open(os.path.join(results_folder, "config.yaml"), "w") as f:
        yaml.dump(yaml_safe_args, f)

    return results_folder

def plot(results_folder, transition_matrices):
    # Load results and config
    if LOAD_PICKLE is True:
        with open(os.path.join(results_folder, "results.pkl"), "rb") as f:
            results_dict = pickle.load(f)

        with open(os.path.join(results_folder, "config.yaml"), "r") as f:
            loaded_args = yaml.safe_load(f)
            # Convert lists back to numpy arrays
            my_args = {}
            for key, value in loaded_args.items():
                if key == 'transition_matrices':
                    my_args[key] = [np.array(matrix) for matrix in value]
                elif isinstance(value, list):
                    my_args[key] = np.array(value)
                else:
                    my_args[key] = value

        eval_losses = results_dict['eval_losses']
        model_eval_steps = results_dict['model_eval_steps']
        model_weights = results_dict['model_weights']
        entropy_rates = results_dict['entropy_rates']

        # Build models
        model_classes = {}
        for key in model_weights:
            model = build_gpt_model(my_args['vocab_size'], my_args['hidden_size'], my_args['n_layer'], my_args['n_head'], use_lora=True, lora_r=my_args['lora_r'], lora_alpha=my_args['lora_alpha'], lora_dropout=my_args['lora_dropout'], num_hyps=key)
            model.load_state_dict(model_weights[key])
            model_classes[key] = model

        # Determine hypotheses for plotting
        N_hyps_max = max(list(model_weights.keys()))
        list_of_hyps = build_hypotheses_list(N_hyps_max, always_include_one=True)

        if DO_PLOT is True:
            # Enable latex
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
            
            # Plot the results
            plt.figure(figsize=(10, 6))

            baseline_color = '#1f77b4'  # Blue
            k2_color = '#d62728'       # Red
            all_results = [{'model_eval_steps': model_eval_steps, 'eval_losses': eval_losses}]

            # Plot training loss with mean and std
            for n_hyps in list_of_hyps:

                # Collect data from all runs
                all_steps = []
                all_losses = []
                for results in all_results:
                    steps = results['model_eval_steps'][n_hyps][:min(PLOT_STEPS_LIMIT, len(results['model_eval_steps'][n_hyps]))]
                    losses = results['eval_losses'][n_hyps][1:][:min(PLOT_STEPS_LIMIT, len(results['eval_losses'][n_hyps][1:]))]
                    all_steps.append(steps)
                    all_losses.append(losses)
                
                # Convert to numpy arrays for easier computation
                all_steps = np.array(all_steps)
                all_losses = np.array(all_losses)
                
                # Calculate mean and std
                mean_steps = np.mean(all_steps, axis=0)
                mean_losses = np.mean(all_losses, axis=0)
                if len(all_losses) > 1:
                    std_losses = np.std(all_losses, axis=0)
                else:
                    std_losses = np.zeros_like(mean_losses)
                
                if n_hyps == 1:
                    label = f'$\\mathcal{{L}}(\\theta)$' + ' (MLE)'
                    color = baseline_color
                    marker = 'o'
                elif n_hyps == 2:
                    label = r'$\mathcal{L}^{\mathrm{WTA}}(\theta)$' + ' (MCL)'
                    color = k2_color
                    marker = 'D'
                
                # Plot mean with error band
                plt.plot(mean_steps, mean_losses, label=label, marker=marker, color=color)
                plt.fill_between(mean_steps, 
                                mean_losses - std_losses, 
                                mean_losses + std_losses, 
                                color=color, alpha=0.2)

            if type(entropy_rates) == dict:
                entropy_rates_considered = entropy_rates[list(entropy_rates.keys())[-1]]
            else:
                entropy_rates_considered = entropy_rates

            # Theoretical Optimal Loss (MLE)
            if 'plot_data.pkl' in os.listdir(results_folder):
                with open(os.path.join(results_folder, "plot_data.pkl"), "rb") as f:
                    label = "Th. Opt. Loss (MLE)"
                    plot_data = pickle.load(f)
                    th_loss_mle = plot_data['th_loss_mle']
                    plt.axhline(y=th_loss_mle/(my_args['seq_len']-1), linestyle='--', label=label)
            else:
                label = "Th. Opt. Loss (MLE)"
                th_loss_mle, std_th_loss_mle = compute_entropy_monte_carlo(transition_matrices, [calculate_stationary_distribution(transition_matrices[0]), calculate_stationary_distribution(transition_matrices[1])], T=my_args['seq_len'], log_base=np.e)
                # Save the results in a pickle file named 'plot_data.pkl'
                with open(os.path.join(results_folder, "plot_data.pkl"), "wb") as f:
                    pickle.dump({'th_loss_mle': th_loss_mle, 'std_th_loss_mle': std_th_loss_mle}, f)
                plt.axhline(y=th_loss_mle/(my_args['seq_len']-1), linestyle='--', label=label)

            # Oracle Entropy rate 
            upper_bound = sum([e for e in entropy_rates_considered])/len(entropy_rates_considered)

            plot_true_wta_loss = True
            if plot_true_wta_loss is True:
                if my_args['normalize_loss_by_T'] is True:
                    lower_bound=(th_loss_mle - np.log(my_args['N_hyps_max']))/(my_args['seq_len']-1)

            plt.axhspan(lower_bound, upper_bound, color='grey', alpha=0.4, label='Th. Opt. Loss Range (MCL)')

            plt.xlabel('Training Steps', fontsize=28)
            plt.ylabel('Loss', fontsize=28)
            plt.title('Validation Loss vs. Training Steps', fontsize=40, pad=20)
            plt.legend(fontsize=22)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            # Save the plot
            if DO_SAVE_PLOT is True:
                plt.tight_layout()
                plt.savefig(os.path.join(results_folder, "training_loss_vs_training_steps_with_std.png"), dpi=300, bbox_inches='tight')

        plt.show()

    for n_hyps_to_visualise in [1,2]:
        plot_transition_matrices_comparison(transition_matrices=my_args['transition_matrices'], model_classes=model_classes, vocab_size=my_args['vocab_size'], do_save_plot=True, results_folder=results_folder)

    return None

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed if provided, otherwise use environment variable
    seed = cfg.seed if not('SEED_EXP' in os.environ) else int(os.environ['SEED_EXP'])
    log.info(f'Seed used: {seed}')
    base_path = os.environ["PROJECT_ROOT"]
    
    # Set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Convert DictConfig to regular dict for compatibility
    my_args = OmegaConf.to_container(cfg, resolve=True)
    
    # Create transition matrices
    transition_matrices = []
    p1, q1 = cfg.p1, cfg.q1
    transition_matrices.append(np.array([[1-p1, p1], [q1, 1-q1]]))
    
    p2, q2 = cfg.p2, cfg.q2
    transition_matrices.append(np.array([[1-p2, p2], [q2, 1-q2]]))
    
    my_args['transition_matrices'] = transition_matrices
    my_args['n_latent_variables'] = len(transition_matrices)

    results_folder = None
    if my_args.get('train', False) is True:
        results_folder = train(my_args, base_path)
    
    if my_args.get('path_pickle') is not None:
        results_folder = my_args['path_pickle']

    if results_folder is None:
        return None

    plot(results_folder, transition_matrices)

    return None

if __name__ == "__main__":
    main() 