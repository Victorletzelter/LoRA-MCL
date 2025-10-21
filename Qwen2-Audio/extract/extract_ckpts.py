"""
Checkpoint Extraction Script

This script extracts and organizes model checkpoints from a log directory. It processes folders
with names in the format: 'YYYY-MM-DD_HH-MM-SS_${training_wta_mode}_epsilon_${epsilon}-${n_hyps}-hyp_rank-${rank}' (placed in a folder with the dataset name)
and creates a structured JSON file organizing checkpoints by dataset, model, and number of hypotheses.

The script keeps only the most recent run for each unique combination of
(dataset_name, model_name, model_specificities, num_hypotheses, rank).

Example folder structure:
logs/clotho/2025-07-10_17-37-12_wta_epsilon-0.0-1-hyp_rank-8/adapter_model/
logs/clotho/2025-05-06_00-05-37_relaxed-wta-epsilon-0.05-5-hyp_rank-16/adapter_model
logs/audiocaps/2025-05-08_10-10-46_wta_epsilon-0.1-1-hyp_rank-24/adapter_model

Example output JSON structure: (key: dataset_name, value: {key: model_name_model_specificities, value: {key: num_hypotheses, value: [checkpoint_paths]}})
{
    "audiocaps": {
        "LoRA-MCL_wta": {
            "1": ["/path/to/checkpoint1.ckpt", "/path/to/checkpoint2.ckpt"]
        },
        "LoRA-MCL_relaxed_wta_0.05": {
            ...
            }
        "LoRA-MCL_relaxed_wta_0.01": {
            ...
            }
    },
}
"""

import sys, os, rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.append(os.path.dirname(os.environ["PROJECT_ROOT"]))

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse


def parse_folder_name(
    folder_name: str,
) -> Optional[Tuple[str, str, str, str, Optional[str]]]:
    """
    Parse a folder name to extract its components.

    Args:
        folder_name: String in format:
            'YYYY-MM-DD_HH-MM-SS_${training_wta_mode}_epsilon_${epsilon}-${n_hyps}-hyp'
    Returns:
        Tuple containing (datetime_str, training_wta_mode, epsilon, num_hypotheses, model_specificities)
        where model_specificities contains additional training details if present
        Returns None if the folder name doesn't match expected format
    """
    try:
        parts = folder_name.split("_")

        # Extract date and time parts
        date_str = parts[0]
        time_str = parts[1]
        # Combine date and time
        datetime_str = f"{date_str} {time_str.replace('-', ':')}"

        # Extract training mode (could be "wta" or "relaxed-wta" etc.)
        training_wta_mode = parts[2]
        
        # Find "epsilon", "hyp", and "rank" keywords
    
        for i, part in enumerate(parts):
            if "epsilon" in part:
                epsilon_idx = i
            if "hyp" in part:
                num_hypotheses_idx = i
            if "rank" in part:
                rank_idx = i
            
        # Extract epsilon value and number of hypotheses from the part after "epsilon"
        epsilon_value = parts[epsilon_idx].split("-")[-1]
        num_hypotheses = parts[num_hypotheses_idx].split("-")[0]
        rank = parts[rank_idx].split("-")[-1]
        
        # Build model specificities from training mode and epsilon
        if training_wta_mode == "wta":
            model_specificities = f"wta"
        elif training_wta_mode == "moe":
            model_specificities = f"moe"
        else:
            # For cases like "relaxed-wta-epsilon", extract the full mode
            model_specificities = f"{training_wta_mode}_epsilon-{epsilon_value}"

        return (
            datetime_str,
            training_wta_mode,
            epsilon_value,
            num_hypotheses,
            rank,
            model_specificities)
    except (ValueError, IndexError):
        return None

def build_checkpoint_structure(
    base_log_dir: Path, start_date: datetime, datasets: List[str]
) -> Dict:
    """
    Build a dictionary structure of checkpoints from specified dataset directories.

    Args:
        base_log_dir: Base path containing dataset-specific log directories
        start_date: Datetime object representing the earliest date to consider
        datasets: List of dataset names to process

    Returns:
        Dictionary with structure:
        {
            "dataset_model_num_hypotheses[_modelspecificities]": [list of checkpoint paths]
        }
    """
    # First, collect all folders with their dates
    model_folders = {}

    # Iterate through specified datasets
    for dataset in datasets:
        dataset_log_dir = base_log_dir / dataset
        if not dataset_log_dir.is_dir():
            print(
                f"Warning: Directory for dataset '{dataset}' not found at {dataset_log_dir}"
            )
            continue

        if "runs" in os.listdir(dataset_log_dir):
            dataset_log_dir = dataset_log_dir / "runs"
        else:
            dataset_log_dir = dataset_log_dir

        for folder in os.listdir(dataset_log_dir):
            folder_path = dataset_log_dir / folder
            if not folder_path.is_dir():
                continue

            parsed = parse_folder_name(folder)
            if parsed is None:
                continue

            (
                datetime_str,
                training_wta_mode,
                epsilon,
                num_hypotheses,
                rank,
                model_specificities,
            ) = parsed

            try:
                folder_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                if folder_datetime < start_date:
                    continue
            except ValueError:
                continue

            # Use the dataset directory name as dataset_name
            dataset_name = dataset
            # Construct model name from training mode
            if training_wta_mode == "moe" or "moe_True" in model_specificities:
                model_name = f"LoRA-MoE_{model_specificities}"
            elif training_wta_mode == "wta" and num_hypotheses == "1":
                model_name = f"LoRA-MLE_{model_specificities}"
            else:
                model_name = f"LoRA-MCL_{model_specificities}"

            # Create combined key with model specificities
            key = f"{dataset_name}_{model_name}_{num_hypotheses}-hyp_rank-{rank}"

            if key not in model_folders:
                model_folders[key] = []
            model_folders[key].append((folder_datetime, folder_path, model_name, rank))

    # Build structure keeping only most recent folders
    structure = {}
    for key, folders in model_folders.items():
        # Sort folders by date (most recent first)
        model_name = folders[0][-1]
        folders.sort(key=lambda x: x[0], reverse=True)

        # Take only the most recent folder
        checkpoint_paths = None
        most_recent_folder = folders[0][1]

        checkpoint_paths = most_recent_folder / "adapter_model"

        # Skip if no checkpoints found in any folder
        if not checkpoint_paths:
            continue

        # Convert PosixPath to str
        structure[key] = str(checkpoint_paths)

    return structure


def main():
    """
    Main function to run the checkpoint extraction script.

    Command line arguments:
        --log_dir: Base path containing dataset-specific log directories
                  (default: './Qwen2-Audio/logs/')
        --datasets: Comma-separated list of dataset names (default: 'audiocaps,clotho')
        --start_date: Start date in YYYYMMDD format (default: '20250115')
        --output_file: Output JSON file path (default: 'ckpts.json')

    Example usage:
        python extract_ckpts.py --log_dir ./logs --datasets audiocaps,clotho --start_date 20240101 --output_file checkpoints.json
    """
    parser = argparse.ArgumentParser(
        description="Extract checkpoints from log directory"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=f'{os.environ["PROJECT_ROOT"]}/logs',
        help="Base path containing dataset-specific log directories",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="clotho,audiocaps",
        help="Comma-separated list of dataset names",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="20250115",
        help="Start date in YYYYMMDD format",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=f'{os.environ["PROJECT_ROOT"]}/ckpts.json',
        help="Output JSON file path",
    )

    args = parser.parse_args()

    # Convert start date string to datetime
    start_date = datetime.strptime(args.start_date, "%Y%m%d")

    # Parse datasets list
    datasets = [d.strip() for d in args.datasets.split(",")]

    # Build checkpoint structure
    log_dir = Path(args.log_dir).resolve()  # Convert to absolute path
    structure = build_checkpoint_structure(log_dir, start_date, datasets)

    # Write to JSON file
    output_path = Path(args.output_file).resolve()  # Convert to absolute path
    with open(output_path, "w") as f:
        json.dump(structure, f, indent=4)
    print(f"Checkpoints paths extracted and saved to {output_path}")

if __name__ == "__main__":
    main()
