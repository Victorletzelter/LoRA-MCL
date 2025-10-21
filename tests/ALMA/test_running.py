import os
import shutil
import subprocess
import sys
import time


def _has_gpu():
    try:
        import torch  # noqa: F401
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def _project_root():
    # rootutils sets PROJECT_ROOT env in ALMA scripts; best effort fallback
    return os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _alma_dir():
    return os.path.join(_project_root(), "ALMA")


def _run(cmd, cwd=None, timeout=900):
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


def test_alma_mt_smoke_train_and_eval():
    if not _has_gpu():
        import pytest
        pytest.skip("GPU not available; skipping ALMA MT smoke test")

    alma_path = _alma_dir()
    assert os.path.isdir(alma_path), f"ALMA directory not found at {alma_path}"

    # Prefer user-provided setup scripts that already constrain steps/samples
    setup_dir = os.path.join(_project_root(), "tests", "ALMA", "setup_scripts")
    train_1h_script = os.path.join(setup_dir, "setup_train_1h.sh")
    eval_1h_script = os.path.join(setup_dir, "setup_eval_1h.sh")
    train_3h_script = os.path.join(setup_dir, "setup_train_3h.sh")
    eval_3h_script = os.path.join(setup_dir, "setup_eval_3h.sh")
    train_2h_script = os.path.join(setup_dir, "setup_train_2h.sh")
    eval_2h_script = os.path.join(setup_dir, "setup_eval_2h.sh")
    
    assert os.path.isfile(train_1h_script), "Missing tests/ALMA/setup_scripts/setup_train_1h.sh"
    assert os.path.isfile(eval_1h_script), "Missing tests/ALMA/setup_scripts/setup_eval_1h.sh"
    assert os.path.isfile(train_3h_script), "Missing tests/ALMA/setup_scripts/setup_train_3h.sh"
    assert os.path.isfile(eval_3h_script), "Missing tests/ALMA/setup_scripts/setup_eval_3h.sh"
    assert os.path.isfile(train_2h_script), "Missing tests/ALMA/setup_scripts/setup_train_2h.sh"
    assert os.path.isfile(eval_2h_script), "Missing tests/ALMA/setup_scripts/setup_eval_2h.sh"

    # Use a tiny number of samples and low LoRA rank for speed
    output_dir = os.path.join(alma_path, "logs", "ci_smoke")
    os.makedirs(output_dir, exist_ok=True)

    # Training 1h: run 1-hypothesis training
    rc, out, dur = _run(["bash", "-lc", f"bash {train_1h_script}"], cwd=_project_root(), timeout=3600)
    assert rc == 0, f"1h training failed (rc={rc}). Output:\n{out}"
    
    # Evaluation 1h: run 1-hypothesis evaluation
    rc, out, dur = _run(["bash", "-lc", f"bash {eval_1h_script}"], cwd=_project_root(), timeout=3600)
    assert rc == 0, f"1h eval failed (rc={rc}). Output:\n{out}"

    # # Training 2h: run 2-hypothesis training
    # rc, out, dur = _run(["bash", "-lc", f"bash {train_2h_script}"], cwd=_project_root(), timeout=3600)
    # assert rc == 0, f"2h training failed (rc={rc}). Output:\n{out}"
    
    # # Evaluation 2h: run 2-hypothesis evaluation
    # rc, out, dur = _run(["bash", "-lc", f"bash {eval_2h_script}"], cwd=_project_root(), timeout=3600)
    # assert rc == 0, f"2h eval failed (rc={rc}). Output:\n{out}"

    # Quick sanity: ensure some predictions or pickle are produced in the expected setup script output dirs
    produced_any = False
    logs_dir = os.path.join(alma_path, "logs")
    if os.path.isdir(logs_dir):
        # Check for artifacts in setup script output directories
        expected_patterns = ["setup_1h_rank_"]
        for root, dirs, files in os.walk(logs_dir):
            # Look for directories matching our setup script patterns
            if any(pattern in root for pattern in expected_patterns):
                if any(f.endswith(".pkl") or f.endswith(".txt") for f in files):
                    produced_any = True
                    break
    assert produced_any, "No eval artifacts (.pkl/.txt) found in ALMA/logs/setup_* directories"


