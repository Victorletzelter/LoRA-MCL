# This file is part of LLaVA

from llava.train.train import val

if __name__ == "__main__":
    val(attn_implementation="flash_attention_2")
