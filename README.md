# llm-inference

LLM inference engine in C/CUDA, written from scratch. Runs Qwen3 (0.6B and 1.7B) in bf16 at batch 1 on hand-written kernels, checked layer by layer against Hugging Face transformers. (at least thats the goal for now lol).

Work in progress.

## Requirements

- Linux, NVIDIA GPU and CUDA toolkit 13 (developed on an RTX 4070 Super, `sm_89`, on CachyOS)
- [uv](https://docs.astral.sh/uv/) for the Python reference scripts

## Setup

```
uv sync
uv run hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
make
```
