import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Worst-row relative L2 error allowed. Calibrated with ref/calibrate.py (HF bf16 on
# the CPU, same rounding points as the dumps, different summation order): its worst
# row is 0.006 in layer 0 and 0.10 over all layers. A RoPE layout bug gives 0.26 at
# position 1 and up to 0.93. Layer 0 is held tighter so subtle bugs show up before
# depth hides them.
TOL_FIRST_LAYER = 0.02
TOL = 0.15

# A greedy flip is only accepted where the reference top-2 logits are this close.
# HF logits are bf16, so 0.125 is one step at their magnitude.
TIE_GAP = 0.25


def tolerance(name):
    if name == "embed" or name.startswith("L00."):
        return TOL_FIRST_LAYER
    return TOL


def row_errors(mine, ref):
    diff = np.linalg.norm(mine - ref, axis=1)
    norm = np.linalg.norm(ref, axis=1)
    return diff / norm


def compare_prompt(ref_dir, out_dir, verbose):
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    tensors = manifest["tensors"]
    n_rows = None
    worst_name, worst_err = None, 0.0
    layer_worst = {}
    missing = None

    for name, shape in tensors.items():
        path = out_dir / f"{name}.bin"
        if not path.exists():
            missing = name
            break
        ref = np.fromfile(ref_dir / f"{name}.bin", dtype=np.float32).reshape(shape).astype(np.float64)
        mine = np.fromfile(path, dtype=np.float32).astype(np.float64)
        if mine.size % shape[1] != 0 or mine.size // shape[1] > shape[0]:
            print(f"  {name}: engine wrote {mine.size} values, reference is {shape}")
            return False
        mine = mine.reshape(-1, shape[1])
        rows = mine.shape[0]
        if n_rows is None:
            n_rows = rows
        if rows != n_rows:
            print(f"  {name}: {rows} rows, earlier tensors have {n_rows}")
            return False

        errs = row_errors(mine, ref[:rows])
        err = errs.max()
        if verbose >= 2:
            print(f"  {name:<16} {err:.4f}")
        group = name.split(".")[0]
        if err >= layer_worst.get(group, (0.0, ""))[0]:
            layer_worst[group] = (err, name)
        if err > worst_err:
            worst_name, worst_err = name, err

        if not err <= tolerance(name):
            print(f"  FAIL {name}: worst row error {err:.4f} > {tolerance(name)}")
            print("  per row: " + " ".join(f"{e:.3f}" for e in errs))
            return False

    if verbose == 1:
        for group, (err, name) in layer_worst.items():
            print(f"  {group:<10} {err:.4f}  {name}")

    checked = len(tensors) if missing is None else list(tensors).index(missing)
    summary = f"{checked}/{len(tensors)} tensors, {n_rows or 0}/{len(manifest['tokens'])} rows"
    if worst_name is not None:
        summary += f", worst {worst_name} {worst_err:.4f}"
    if missing is not None:
        print(f"  ok so far ({summary}); next missing tensor: {missing}")
        return False

    assert n_rows is not None
    ref = np.fromfile(ref_dir / "logits.bin", dtype=np.float32).reshape(tensors["logits"])[:n_rows]
    mine = np.fromfile(out_dir / "logits.bin", dtype=np.float32).reshape(-1, ref.shape[1])
    ok = True
    for row in range(n_rows):
        ref_best = ref[row].argmax()
        mine_best = mine[row].argmax()
        if ref_best == mine_best:
            continue
        gap = ref[row, ref_best] - ref[row, mine_best]
        tied = gap <= TIE_GAP
        verdict = "accepted, near tie" if tied else "FAIL"
        print(f"  position {row}: argmax {mine_best} vs reference {ref_best}, gap {gap:.3f} ({verdict})")
        ok = ok and tied
    print(f"  {'ok' if ok else 'FAIL'} ({summary})")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Compare engine activation dumps against the HF dumps.")
    parser.add_argument("ref_root", help="e.g. dumps/Qwen3-0.6B")
    parser.add_argument("out_root", help="engine dumps, one pNN directory per prompt")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v: worst error per layer, -vv: every tensor")
    args = parser.parse_args()

    out_dirs = sorted(p for p in Path(args.out_root).iterdir() if p.is_dir())
    if not out_dirs:
        print(f"no prompt directories in {args.out_root}")
        sys.exit(1)
    all_ok = True
    for out_dir in out_dirs:
        print(f"{out_dir.name}:")
        ok = compare_prompt(Path(args.ref_root) / out_dir.name, out_dir, args.verbose)
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
