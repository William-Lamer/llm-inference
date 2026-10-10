import argparse
import json
import sys
from pathlib import Path

import numpy as np

from compare import TIE_GAP


# the engine generates from its own tokens, there can be some differences so
# after the first different the context will be different and we can't compare
# anything anymore
def check_free(manifest, generated):
    ref = manifest["greedy"]
    gaps = manifest["greedy_top2_gap"]
    n = min(len(ref), len(generated))
    first_diff = next((i for i in range(n) if ref[i] != generated[i]), n)
    if generated == ref:
        print(f"  free-running: all {len(ref)} tokens match")
        return True
    if first_diff == len(ref):
        print(f"  FAIL free-running: engine generated {len(generated)} tokens, reference {len(ref)}")
        return False
    got = generated[first_diff] if first_diff < len(generated) else "nothing (stopped)"
    gap = gaps[first_diff]
    tied = gap <= TIE_GAP
    verdict = "accepted, near tie" if tied else "FAIL"
    print(f"  free-running: {first_diff}/{len(ref)} tokens match, then {got} vs reference "
          f"{ref[first_diff]}, gap {gap:.3f} ({verdict})")
    return tied


# the engine is given the prompt and the reference tokens, so its
# context will match at every step, and then we check each step on its own.
def check_forced(manifest, logits_path):
    ref = manifest["greedy"]
    gaps = manifest["greedy_top2_gap"]
    n_prompt = len(manifest["tokens"])
    logits = np.fromfile(logits_path, dtype=np.float32)
    logits = logits.reshape(-1, logits.size // (n_prompt + len(ref)))
    ok = True
    n_flips = 0
    for step in range(len(ref)):
        mine = int(logits[n_prompt - 1 + step].argmax())
        if mine == ref[step]:
            continue
        n_flips += 1
        tied = gaps[step] <= TIE_GAP
        verdict = "accepted, near tie" if tied else "FAIL"
        print(f"  forced step {step}: argmax {mine} vs reference {ref[step]}, gap {gaps[step]:.3f} ({verdict})")
        ok = ok and tied
    print(f"  {'ok' if ok else 'FAIL'} teacher-forced: {len(ref) - n_flips}/{len(ref)} steps match")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Check engine greedy output against HF generate.")
    parser.add_argument("ref_root", help="e.g. dumps/Qwen3-0.6B")
    parser.add_argument("out_root", help="one pNN directory per prompt with generated.txt and logits.bin")
    args = parser.parse_args()

    out_dirs = sorted(p for p in Path(args.out_root).iterdir() if p.is_dir())
    if not out_dirs:
        print(f"no prompt directories in {args.out_root}")
        sys.exit(1)
    all_ok = True
    for out_dir in out_dirs:
        print(f"{out_dir.name}:")
        manifest = json.loads((Path(args.ref_root) / out_dir.name / "manifest.json").read_text())
        generated = [int(t) for t in (out_dir / "generated.txt").read_text().split()]
        free_ok = check_free(manifest, generated)
        forced_ok = check_forced(manifest, out_dir / "logits.bin")
        all_ok = all_ok and free_ok and forced_ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
