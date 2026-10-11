import argparse
import csv
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

from bench import PROMPT

OUT = Path("build/profile")


def run(cmd):
    print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}")
    return result.stdout


def read_report(path, name_column):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            # "gemv_kernel(__nv_bfloat16 *, ...)" -> "gemv_kernel"
            name = row[name_column].split("(")[0].strip("[]")
            rows.append((name, int(row["Total Time (ns)"]), int(row[list(row)[2]])))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Per-kernel GPU time of decode steps, with nsys.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer(args.prompt).input_ids
    tokens_path = OUT / "prompt_tokens.txt"
    tokens_path.write_text(" ".join(str(t) for t in prompt) + "\n")

    # The engine wraps its timed decode steps in cudaProfilerStart/Stop, so the capture
    # skips loading, the prompt and the warmup run.
    report = OUT / "engine"
    engine_out = run(["nsys", "profile", "--trace=cuda", "--capture-range=cudaProfilerApi",
                      "--capture-range-end=stop", "--force-overwrite=true", "-o", str(report),
                      "build/engine", args.model, "--tokens", str(tokens_path),
                      "--bench", str(args.steps), "--runs", "1"])
    run_line = next(line for line in engine_out.splitlines() if line.startswith("run "))
    wall_ms = float(run_line.split()[3]) * 1e3 / args.steps

    stats = OUT / "stats"
    run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum",
         "--report", "cuda_gpu_mem_time_sum", "--format", "csv", "--output", str(stats),
         str(report) + ".nsys-rep"])
    kernels = read_report(OUT / "stats_cuda_gpu_kern_sum.csv", "Name")
    copies = read_report(OUT / "stats_cuda_gpu_mem_time_sum.csv", "Operation")
    entries = sorted(kernels + copies, key=lambda e: e[1], reverse=True)

    gpu_ns = sum(total for _, total, _ in entries)
    gpu_ms = gpu_ns / 1e6 / args.steps
    print(f"\n{Path(args.model).name}, prompt {len(prompt)} tokens, {args.steps} decode steps, "
          f"batch 1, bf16 (nsys)")
    print(f"  {'kernel':<34} {'per step':>8} {'us/step':>10} {'us each':>9} {'share':>7}")
    for name, total, count in entries:
        us_per_step = total / 1e3 / args.steps
        print(f"  {name:<34} {count / args.steps:8.0f} {us_per_step:10.1f} "
              f"{total / 1e3 / count:9.2f} {total / gpu_ns:7.1%}")
    print(f"  GPU busy {gpu_ms:.3f} ms/step, wall {wall_ms:.3f} ms/step under nsys, "
          f"idle {wall_ms - gpu_ms:.3f} ms/step ({1 - gpu_ms / wall_ms:.1%})")


if __name__ == "__main__":
    main()
