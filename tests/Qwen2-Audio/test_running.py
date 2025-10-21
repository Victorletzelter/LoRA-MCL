import os
import shutil
import subprocess
import sys
import time
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def _has_gpu():
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _project_root():
    # rootutils sets PROJECT_ROOT env in Qwen2-Audio scripts; best effort fallback
    return os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _qwen2_audio_dir():
    return os.path.join(_project_root(), "Qwen2-Audio")


def _run(cmd, cwd=None, timeout=3600):
    env = os.environ.copy()
    env.setdefault("NGPUS", "1")
    start = time.time()
    # Inherit parent's stdio -> live output
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError(f"Command timed out after {timeout}s: {' '.join(cmd)}")
    duration = time.time() - start
    # No buffered output to return; keep empty string
    return rc, "", duration


def test_qwen2_audio_smoke_train_and_eval():
    if not _has_gpu():
        import pytest
        pytest.skip("GPU not available; skipping Qwen2-Audio smoke test")

    qwen2_audio_path = _qwen2_audio_dir()
    assert os.path.isdir(qwen2_audio_path), f"Qwen2-Audio directory not found at {qwen2_audio_path}"

    # Locate the setup scripts in tests/Qwen2-Audio/setup_scripts/
    setup_dir = os.path.join(_project_root(), "tests", "Qwen2-Audio", "setup_scripts")
    train_loop_script = os.path.join(setup_dir, "setup_train_loop.sh")
    eval_scripts_script = os.path.join(setup_dir, "setup_eval_scripts.sh")
    
    assert os.path.isfile(train_loop_script), "Missing tests/Qwen2-Audio/setup_scripts/setup_train_loop.sh"
    assert os.path.isfile(eval_scripts_script), "Missing tests/Qwen2-Audio/setup_scripts/setup_eval_scripts.sh"

    # Use a tiny number of samples and low LoRA rank for speed
    output_dir = os.path.join(qwen2_audio_path, "logs", "ci_smoke")
    os.makedirs(output_dir, exist_ok=True)

    # # Training loop: run setup_train_loop.sh
    rc, out, dur = _run(["bash", "-lc", f"bash {train_loop_script}"], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Training loop failed (rc={rc}). Output:\n{out}"

    # Quick sanity: ensure some predictions or outputs are produced in the logs directory
    produced_any = False
    logs_dir = os.path.join(qwen2_audio_path, "logs")
    if os.path.isdir(logs_dir):
        # Check for artifacts in the logs directory
        # Look for common output patterns: adapter_model, predictions, checkpoints, etc.
        for root, dirs, files in os.walk(logs_dir):
            # Look for model artifacts or prediction outputs
            if any(f.endswith(".pkl") or f.endswith(".txt") or f.endswith(".json") or 
                   f == "adapter_model" or "checkpoint" in f.lower() for f in files):
                produced_any = True
                break
            # Check if adapter_model directory exists (common output structure)
            if "adapter_model" in dirs:
                produced_any = True
                break
    
    assert produced_any, "No artifacts (.pkl/.txt/.json/adapter_model) found in Qwen2-Audio/logs/ directories"

    ## Download the provided checkpoints
    rc, out, dur = _run(["python", os.path.join(qwen2_audio_path,"/download/download_ckpts.py")], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Download checkpoints failed (rc={rc}). Output:\n{out}"

    ## Extract the checkpoints
    rc, out, dur = _run(["python", os.path.join(qwen2_audio_path,"/extract/extract_ckpts.py"), "--log_dir", "ckpts"], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Extract checkpoints failed (rc={rc}). Output:\n{out}"

    ## Set the environment variables
    rc, out, dur = _run(["python", os.path.join(qwen2_audio_path,"/extract/generate_checkpoint_variables.py"), "--input", "ckpts.json", "--output", "checkpoint_vars.sh", "--format", "export"], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Generate checkpoint variables failed (rc={rc}). Output:\n{out}"

    ## Source the environment variables
    rc, out, dur = _run(["source", "checkpoint_vars.sh"], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Source checkpoint variables failed (rc={rc}). Output:\n{out}"

    # Evaluation: run setup_eval_scripts.sh with 5 hypotheses and clotho dataset
    # The eval script expects arguments: <number_of_hypotheses> <dataset_name>
    rc, out, dur = _run(["bash", "-lc", f"bash {eval_scripts_script} 5 clotho"], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Eval scripts failed (rc={rc}). Output:\n{out}"

    # Evaluation: run setup_eval_scripts.sh with 5 hypotheses and audiocaps dataset
    # The eval script expects arguments: <number_of_hypotheses> <dataset_name>
    rc, out, dur = _run(["bash", "-lc", f"bash {eval_scripts_script} 5 clotho"], cwd=_project_root(), timeout=7200)
    assert rc == 0, f"Eval scripts failed (rc={rc}). Output:\n{out}"