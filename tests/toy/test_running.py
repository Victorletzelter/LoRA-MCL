import os
import sys
import subprocess
import time
from pathlib import Path
import pytest
import rootutils
os.environ["LORA_MCL_ROOT"]=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.environ["LORA_MCL_ROOT"])

def test_train_setup_runs():
    cmd = [sys.executable, "train.py", "experiment=setup"]
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cwd = os.path.join(os.environ["LORA_MCL_ROOT"],"toy")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Training script failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )



@pytest.fixture(scope="module")
def run_training():
    cmd = [sys.executable, "train.py", "experiment=setup"]
    env = os.environ.copy()
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    project_root = os.environ["LORA_MCL_ROOT"]
    cwd = os.path.join(project_root, "toy")

    # Track existing result directories before run
    results_base = os.path.join(project_root, "toy", "results", "pkl_files")
    before = set()
    if os.path.isdir(results_base):
        before = {name for name in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, name))}

    start_time = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )

    # Detect newly created results folder
    results_folder = None
    if os.path.isdir(results_base):
        after = {name for name in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, name))}
        new_dirs = list(after - before)
        if new_dirs:
            # Pick the newest by mtime among new_dirs
            new_paths = [os.path.join(results_base, d) for d in new_dirs]
            results_folder = max(new_paths, key=lambda p: os.path.getmtime(p))
        else:
            # Fallback: pick the newest folder overall
            all_dirs = [os.path.join(results_base, d) for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d))]
            if all_dirs:
                results_folder = max(all_dirs, key=lambda p: os.path.getmtime(p))

    logs_dir = os.path.join(project_root, "toy", "logs")

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "results_folder": results_folder,
        "logs_dir": logs_dir,
        "start_time": start_time,
    }


def test_results_folder_and_pngs_created(run_training):
    assert run_training["returncode"] == 0, (
        f"Training failed. STDOUT:\n{run_training['stdout']}\nSTDERR:\n{run_training['stderr']}"
    )
    results_folder = run_training["results_folder"]
    assert results_folder is not None and os.path.isdir(results_folder), "No results folder detected."

    # Required outputs
    expected_files = [
        "results.pkl",
        "config.yaml",
        "training_loss_vs_training_steps_with_std.png",
        "transition_matrices_comparison_grid_weighted.png",
    ]
    missing = [f for f in expected_files if not os.path.isfile(os.path.join(results_folder, f))]
    assert not missing, f"Missing files in results folder {results_folder}: {missing}"


def test_tensorboard_events_created(run_training):
    logs_dir = run_training["logs_dir"]
    start_time = run_training["start_time"]
    assert os.path.isdir(logs_dir), f"Logs directory not found: {logs_dir}"

    # Find TensorBoard event files newer than the training start
    events = []
    for p in Path(logs_dir).glob("events.out.tfevents.*"):
        try:
            if p.stat().st_mtime >= start_time - 1:  # allow slight clock skew
                events.append(str(p))
        except FileNotFoundError:
            continue
    assert events, f"No recent TensorBoard event files found in {logs_dir} after training."

if __name__ == "__main__":
    test_train_setup_runs()
    test_results_folder_and_pngs_created(run_training)
    test_tensorboard_events_created(run_training)