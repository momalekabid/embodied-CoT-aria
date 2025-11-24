"""
merge_lora_checkpoint.py

Merge lora adapter weights into base model after training.
run this once after training finishes to create a deployable checkpoint.

usage:
    python vla-scripts/merge_lora_checkpoint.py \
        --base_model_path openvla/openvla-7b \
        --adapter_path adapter-tmp/openvla-7b+dataset+b16+lr-2e-05+lora-r32+dropout-0.0 \
        --output_path runs/openvla-7b+dataset+b16+lr-2e-05+lora-r32+dropout-0.0
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


def merge_lora_adapter(base_model_path: str, adapter_path: str, output_path: str, use_cpu: bool = False):
    """
    merge lora adapter into base model and save.

    args:
        base_model_path: path to base vla model (huggingface hub or local)
        adapter_path: path to saved lora adapter weights
        output_path: where to save merged model
        use_cpu: if true, load on cpu to avoid gpu oom (slower but safer)
    """
    print(f"loading base model from {base_model_path}...")

    # load base model
    device = "cpu" if use_cpu else "cuda"
    base_vla = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if not use_cpu:
        base_vla = base_vla.to(device)

    print(f"Loading adapter from {adapter_path}...")

    # load and merge adapter
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_path)

    print("merging adapter into base model...")
    merged_vla = merged_vla.merge_and_unload()

    # save merged model
    print(f"saving merged model to {output_path}...")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    merged_vla.save_pretrained(output_path)

    # also copy processor if it exists in adapter path
    adapter_path = Path(adapter_path)
    if (adapter_path.parent.parent / "runs" / adapter_path.name).exists():
        processor_path = adapter_path.parent.parent / "runs" / adapter_path.name
        if (processor_path / "preprocessor_config.json").exists():
            print("copying processor config...")
            processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
            processor.save_pretrained(output_path)

    print(f"done! merged model saved to {output_path}")
    print(f"\nto use this model:")
    print(f"  from transformers import AutoModelForVision2Seq")
    print(f"  vla = AutoModelForVision2Seq.from_pretrained('{output_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge lora adapter into base model")
    parser.add_argument("--base_model_path", type=str, default="openvla/openvla-7b", help="path to base vla model")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="path to lora adapter checkpoint (e.g., adapter-tmp/experiment-id)",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="where to save merged model (e.g., runs/experiment-id)"
    )
    parser.add_argument("--use_cpu", action="store_true", help="load on cpu instead of gpu (slower but avoids oom)")

    args = parser.parse_args()

    merge_lora_adapter(args.base_model_path, args.adapter_path, args.output_path, args.use_cpu)
