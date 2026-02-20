# This file is part of LLaVA (https://github.com/haotian-liu/LLaVA/blob/main/llava/train/train_mem.py)

import sys, os, importlib.util
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

llava2_path = os.path.abspath(f"{os.environ['PROJECT_ROOT']}/LLaVA/llava/__init__.py")
spec = importlib.util.spec_from_file_location("llava", llava2_path)
llava = importlib.util.module_from_spec(spec)
sys.modules["llava"] = llava
spec.loader.exec_module(llava)
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
