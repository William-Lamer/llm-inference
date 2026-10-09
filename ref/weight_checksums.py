import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

CHUNK = 1 << 24


# Same checksums as print_weight_checksums in src/loader.cu, computed from safetensors.
def checksums(bits):
    total = np.uint64(0)
    weighted = np.uint64(0)
    # Wrapping mod 2^64 is intended, as with uint64_t in C.
    with np.errstate(over="ignore"):
        for start in range(0, bits.size, CHUNK):
            chunk = bits[start:start + CHUNK].astype(np.uint64)
            index = np.arange(start + 1, start + 1 + chunk.size, dtype=np.uint64)
            total += chunk.sum(dtype=np.uint64)
            weighted += (index * chunk).sum(dtype=np.uint64)
    return total, weighted


def main():
    model_dir = Path(sys.argv[1])
    config = json.loads((model_dir / "config.json").read_text())
    for path in sorted(model_dir.glob("*.safetensors")):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                if name == "lm_head.weight" and config["tie_word_embeddings"]:
                    continue
                t = f.get_tensor(name)
                bits = t.view(torch.int16).flatten().numpy().view(np.uint16)
                total, weighted = checksums(bits)
                first = float(t.flatten()[0])
                print(f"{name} {bits.size} {total} {weighted} {first:.6g}")


if __name__ == "__main__":
    main()
