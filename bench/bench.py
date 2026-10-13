import argparse
import csv
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer

SPEC_GBS = 504.0
PROMPT = "The capital of France is"
RESULTS = Path("bench/results.csv")
FIELDS = ["date", "commit", "model", "metric", "value", "unit", "prompt_tokens", "gen_tokens",
          "gpu", "driver", "cuda", "os", "notes"]


def run(cmd):
    print("$ " + " ".join(cmd), file=sys.stderr, flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}")
    return result.stdout


# Lines are "key value...". Run lines are "run <i> <steps> <seconds>".
def parse(stdout):
    values = {}
    runs = []
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "run":
            runs.append(float(parts[3]))
        elif len(parts) == 2:
            values[parts[0]] = parts[1]
    return values, runs


def environment():
    query = "--query-gpu=name,driver_version"
    smi = run(["nvidia-smi", query, "--format=csv,noheader"]).strip().split(", ")
    nvcc = run(["nvcc", "--version"])
    cuda = nvcc.split("release ")[1].split(",")[0]
    if "microsoft" in Path("/proc/version").read_text().lower():
        os_name = "WSL2"
    else:
        os_release = Path("/etc/os-release").read_text()
        os_name = os_release.split('PRETTY_NAME="')[1].split('"')[0]
    return {"gpu": smi[0], "driver": smi[1], "cuda": cuda, "os": os_name}


def git_commit():
    dirty = run(["git", "status", "--porcelain", "--", "src", "bench", "Makefile",
                 f":!{RESULTS}"])
    if dirty.strip():
        sys.exit("src/, bench/ or Makefile has uncommitted changes: commit before recording\n" + dirty)
    return run(["git", "rev-parse", "--short", "HEAD"]).strip()


def record(rows, model, n_prompt, n_steps):
    commit = git_commit()
    env = environment()
    date = datetime.date.today().isoformat()
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for metric, value, unit, notes in rows:
            writer.writerow({"date": date, "commit": commit, "model": model, "metric": metric,
                             "value": value, "unit": unit, "prompt_tokens": n_prompt,
                             "gen_tokens": n_steps, "notes": notes, **env})
    print(f"appended {len(rows)} rows to {RESULTS} (commit {commit})")


def main():
    parser = argparse.ArgumentParser(description="Decode tok/s, roofline and HF baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--llamacpp", help="llama.cpp checkout with build/bin/llama-bench and "
                        "models/<model>-BF16.gguf")
    parser.add_argument("--record", action="store_true", help=f"append the numbers to {RESULTS}")
    args = parser.parse_args()

    model = Path(args.model).name
    out_dir = Path("build/bench")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = tokenizer(args.prompt).input_ids
    tokens_path = out_dir / "prompt_tokens.txt"
    tokens_path.write_text(" ".join(str(t) for t in prompt) + "\n")
    n_prompt = len(prompt)

    copy_out = run(["build/bandwidth"])
    copy_gbs = float(parse(copy_out)[0]["copy_bandwidth_gbs"])

    engine_cmd = ["build/engine", args.model, "--tokens", str(tokens_path),
                  "--bench", str(args.steps), "--runs", str(args.runs)]
    engine_out = run(engine_cmd)
    engine_values, engine_runs = parse(engine_out)
    engine_s = statistics.median(engine_runs)
    ms_per_step = engine_s * 1e3 / args.steps
    tok_s = args.steps / engine_s

    # The same steps launched one kernel at a time, to measure what the CUDA graph saves.
    eager_runs = parse(run(engine_cmd + ["--no-graph"]))[1]
    eager_s = statistics.median(eager_runs)
    eager_ms_per_step = eager_s * 1e3 / args.steps
    eager_tok_s = args.steps / eager_s

    rows = [
        ("copy_bandwidth", f"{copy_gbs:.1f}", "GB/s", "device-to-device cudaMemcpy, 512 MiB"),
        ("decode_ms_per_step", f"{ms_per_step:.3f}", "ms", f"median of {args.runs} runs"),
        ("decode_tok_s", f"{tok_s:.1f}", "tok/s", f"median of {args.runs} runs, greedy, batch 1"),
        ("decode_ms_per_step_no_graph", f"{eager_ms_per_step:.3f}", "ms",
         f"median of {args.runs} runs, --no-graph (one launch per kernel)"),
        ("decode_tok_s_no_graph", f"{eager_tok_s:.1f}", "tok/s",
         f"median of {args.runs} runs, --no-graph (one launch per kernel)"),
    ]
    print(f"\n{model}, prompt {n_prompt} tokens, {args.steps} decode steps, batch 1, bf16")
    print(f"  copy bandwidth      {copy_gbs:8.1f} GB/s ({copy_gbs / SPEC_GBS:.1%} of {SPEC_GBS:.0f})")
    print(f"  engine              {ms_per_step:8.3f} ms/step  {tok_s:8.1f} tok/s")
    print(f"  engine, no graph    {eager_ms_per_step:8.3f} ms/step  {eager_tok_s:8.1f} tok/s "
          f"(graph saves {eager_ms_per_step - ms_per_step:.3f} ms/step)")

    if "weight_bytes_per_step" in engine_values:
        weight_bytes = int(engine_values["weight_bytes_per_step"])
        kv_bytes_per_position = int(engine_values["kv_bytes_per_position"])
        # Step i attends over n_prompt + i + 1 positions
        mean_positions = n_prompt + (args.steps + 1) / 2
        step_bytes = weight_bytes + kv_bytes_per_position * mean_positions
        effective_gbs = step_bytes / (ms_per_step * 1e-3) / 1e9
        pct_spec = effective_gbs / SPEC_GBS
        pct_measured = effective_gbs / copy_gbs
        print(f"  bytes per step      {step_bytes / 1e9:8.4f} GB "
              f"(weights {weight_bytes / 1e9:.4f}, kv mean {(step_bytes - weight_bytes) / 1e6:.2f} MB)")
        print(f"  effective bandwidth {effective_gbs:8.1f} GB/s")
        print(f"  roofline            {pct_spec:8.1%} of {SPEC_GBS:.0f} GB/s spec, "
              f"{pct_measured:.1%} of measured copy bandwidth")
        rows += [
            ("bytes_per_step", f"{step_bytes:.0f}", "B", "weights + mean kv cache read"),
            ("effective_bandwidth", f"{effective_gbs:.1f}", "GB/s", "bytes_per_step / ms_per_step"),
            ("roofline_pct_spec", f"{pct_spec * 100:.1f}", "%", f"of {SPEC_GBS:.0f} GB/s"),
            ("roofline_pct_measured", f"{pct_measured * 100:.1f}", "%", "of copy_bandwidth"),
        ]
    else:
        print("  roofline            skipped: the engine does not print weight_bytes_per_step and "
              "kv_bytes_per_position yet")

    if not args.skip_hf:
        hf_out = run([sys.executable, "bench/hf_baseline.py", "--model", args.model,
                      "--tokens", str(tokens_path), "--steps", str(args.steps),
                      "--runs", str(args.runs)])
        hf_values, hf_runs = parse(hf_out)
        hf_s = statistics.median(hf_runs)
        hf_tok_s = args.steps / hf_s
        speedup = hf_s / engine_s
        attn = hf_values["attn_implementation"]
        print(f"  HF transformers     {hf_s * 1e3 / args.steps:8.3f} ms/step  {hf_tok_s:8.1f} tok/s "
              f"({attn} attention)")
        print(f"  speedup over HF     {speedup:8.2f}x")
        rows += [
            ("hf_decode_tok_s", f"{hf_tok_s:.1f}", "tok/s",
             f"transformers eager bf16, {attn} attention, KV cache, median of {args.runs} runs"),
            ("speedup_vs_hf", f"{speedup:.2f}", "x", "decode_tok_s / hf_decode_tok_s"),
        ]

    if args.llamacpp:
        llama_dir = Path(args.llamacpp)
        gguf = llama_dir / "models" / f"{model}-BF16.gguf"
        if not gguf.exists():
            sys.exit(f"{gguf} not found: convert with llama.cpp's convert_hf_to_gguf.py --outtype bf16")
        llama_cmd = [str(llama_dir / "build/bin/llama-bench"), "-m", str(gguf), "-ngl", "99",
                     "-p", "0", "-n", str(args.steps), "-d", str(n_prompt),
                     "-r", str(args.runs), "-o", "json"]
        llama = json.loads(run(llama_cmd))[0]
        llama_tok_s = statistics.median(llama["samples_ts"])
        llama_speedup = tok_s / llama_tok_s
        llama_build = llama["build_commit"]
        print(f"  llama.cpp           {1e3 / llama_tok_s:8.3f} ms/step  {llama_tok_s:8.1f} tok/s "
              f"(build {llama_build}, BF16 GGUF)")
        print(f"  speedup over llama.cpp {llama_speedup:5.2f}x")
        rows += [
            ("llamacpp_decode_tok_s", f"{llama_tok_s:.1f}", "tok/s",
             f"llama.cpp {llama_build} llama-bench, CUDA, BF16 GGUF, default flags, "
             f"tg{args.steps} at depth {n_prompt}, median of {args.runs} runs"),
            ("speedup_vs_llamacpp", f"{llama_speedup:.2f}", "x",
             "decode_tok_s / llamacpp_decode_tok_s"),
        ]

    if args.record:
        record(rows, model, n_prompt, args.steps)


if __name__ == "__main__":
    main()
