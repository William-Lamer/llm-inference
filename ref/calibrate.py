# Runs HF in bf16 on the CPU and writes its activations in the engine's dump
# format. The CPU rounds to bf16 at the same points as the GPU reference but
# sums in a different order, like the engine will, so
#   uv run ref/compare.py -v dumps/<model> build/calib/<model>
# shows how much error a correct implementation has. That sets the tolerances
# in compare.py, and it also checks compare.py itself: this run must pass.
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from dump import install_hooks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--ref", default="dumps")
    parser.add_argument("--out", default="build/calib")
    args = parser.parse_args()

    model_dir = Path(args.model)
    ref_root = Path(args.ref) / model_dir.name
    out_root = Path(args.out) / model_dir.name
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="eager")
    model = model.eval()

    acts = {}
    install_hooks(model, acts)
    for ref_dir in sorted(ref_root.glob("p*")):
        manifest = json.loads((ref_dir / "manifest.json").read_text())
        acts.clear()
        with torch.no_grad():
            model(torch.tensor([manifest["tokens"]]))
        out_dir = out_root / ref_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, t in acts.items():
            t.numpy().tofile(out_dir / f"{name}.bin")
        print(f"{ref_dir.name}: {len(acts)} tensors -> {out_dir}")


if __name__ == "__main__":
    main()
