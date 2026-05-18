#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and push to HF Hub.

Usage:
    python src/merge_and_export.py \
        --adapter checkpoints/mtup_2tasks/MTUP2/s42/final \
        --hf_repo YOUR_USERNAME/vn-amr-mtup2-merged \
        --hf_token YOUR_HF_TOKEN
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",   required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--hf_repo",   required=True, help="HF Hub repo to push to, e.g. username/vn-amr-mtup2-merged")
    parser.add_argument("--hf_token",  required=True, help="HF token with write access")
    args = parser.parse_args()

    print(f"[1/4] Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",        # merge on CPU to avoid VRAM limit
        trust_remote_code=True,
    )

    print(f"[2/4] Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("[3/4] Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    model.eval()

    print(f"[4/4] Pushing merged model to HF Hub: {args.hf_repo}")
    model.push_to_hub(args.hf_repo, token=args.hf_token, private=True)
    tokenizer.push_to_hub(args.hf_repo, token=args.hf_token, private=True)

    print(f"\n✓ Done! Model available at: https://huggingface.co/{args.hf_repo}")
    print("Next step: download GGUF from Colab using llama.cpp")


if __name__ == "__main__":
    main()
