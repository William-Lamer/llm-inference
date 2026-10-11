import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


# Same measurement as the engine's --bench.
def decode(model, prompt_ids, n_steps):
    out = model(input_ids=prompt_ids, use_cache=True)
    cache = out.past_key_values
    token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    token.item()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        token.item()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description="Time HF transformers greedy decode at batch 1.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True, help="prompt token IDs, like tokens.txt")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    prompt = [int(t) for t in Path(args.tokens).read_text().split()]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.cuda().eval()  # pyright: ignore[reportCallIssue]
    prompt_ids = torch.tensor([prompt], device="cuda")
    print(f"attn_implementation {model.config._attn_implementation}", flush=True)

    with torch.inference_mode():
        # run 0 is warmup
        for run in range(args.runs + 1):
            seconds = decode(model, prompt_ids, args.steps)
            if run > 0:
                print(f"run {run} {args.steps} {seconds:.6f}", flush=True)


if __name__ == "__main__":
    main()
