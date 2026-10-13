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

# Single rows over the limit that were investigated and accepted. Confirmed to be
# rouding drift. On different hardware or versions they might differ and throw an error.
KNOWN_OUTLIERS = {
    ("Qwen3-0.6B", "p05"): {
        "L08.v_proj": (27, 0.1588),
        "L12.k_proj": (27, 0.2118),
        "L12.v_proj": (27, 0.2335),
    },
    ("Qwen3-1.7B", "p01"): {
        "L10.mlp_act": (13, 0.2197),
        "L10.down": (13, 0.1982),
        "L12.mlp_act": (13, 0.2654),
        "L12.down": (13, 0.2188),
        "L14.mlp_act": (13, 0.2003),
        "L14.down": (13, 0.1884),
    },
    ("Qwen3-1.7B", "p05"): {
        "L12.k_proj": (23, 0.2544),
        "L12.v_proj": (23, 0.2466),
        "L13.k_proj": (23, 0.2372),
        "L13.v_proj": (23, 0.2667),
        "L14.k_proj": (23, 0.2145),
        "L14.v_proj": (23, 0.2020),
    },
}
# The pinned errors are rounded to 4 decimals
OUTLIER_MATCH = 0.0001


def tolerance(name):
    if name == "embed" or name.startswith("L00."):
        return TOL_FIRST_LAYER
    return TOL


def row_errors(mine, ref):
    diff = np.linalg.norm(mine - ref, axis=1)
    norm = np.linalg.norm(ref, axis=1)
    return diff / norm


def compare_prompt(ref_dir, out_dir, verbose, known):
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

        limit_errs = errs
        if name in known:
            row, value = known[name]
            worst_row = errs.argmax()
            if worst_row != row or abs(err - value) > OUTLIER_MATCH:
                print(f"  FAIL {name}: worst row {worst_row} at {err:.4f}, known outlier is row {row} at {value:.4f}")
                return False
            print(f"  known outlier {name}: row {row} at {err:.4f} (accepted)")
            limit_errs = np.delete(errs, row)

        if not limit_errs.max() <= tolerance(name):
            print(f"  FAIL {name}: worst row error {limit_errs.max():.4f} > {tolerance(name)}")
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
    parser.add_argument("--known-outliers", action="store_true",
                        help="accept the engine's investigated outliers in KNOWN_OUTLIERS")
    args = parser.parse_args()

    out_dirs = sorted(p for p in Path(args.out_root).iterdir() if p.is_dir())
    if not out_dirs:
        print(f"no prompt directories in {args.out_root}")
        sys.exit(1)
    model = Path(args.ref_root).name
    all_ok = True
    for out_dir in out_dirs:
        print(f"{out_dir.name}:")
        known = {}
        if args.known_outliers:
            known = KNOWN_OUTLIERS.get((model, out_dir.name), {})
        ok = compare_prompt(Path(args.ref_root) / out_dir.name, out_dir, args.verbose, known)
        all_ok = all_ok and ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
