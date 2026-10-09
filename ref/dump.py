import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3


# Dumped tensors are 2-D: one row per prompt position, so the engine's
# position t compares against row t. Every tensor HF hands us has batch first.
def to_rows(t):
    seq_len = t.shape[1]
    return t[0].detach().float().reshape(seq_len, -1).cpu()


def save_output(acts, name):
    def hook(module, args, output):
        acts[name] = to_rows(output)
    return hook


def save_input(acts, name):
    def hook(module, args):
        acts[name] = to_rows(args[0])
    return hook


def install_hooks(model, acts):
    handles = []
    inner = model.model

    def add_output(module, name):
        handles.append(module.register_forward_hook(save_output(acts, name)))

    def add_input(module, name):
        handles.append(module.register_forward_pre_hook(save_input(acts, name)))

    add_output(inner.embed_tokens, "embed")
    for i, layer in enumerate(inner.layers):
        p = f"L{i:02d}"
        attn = layer.self_attn
        mlp = layer.mlp
        add_output(layer.input_layernorm, f"{p}.attn_norm")
        add_output(attn.q_proj, f"{p}.q_proj")
        add_output(attn.k_proj, f"{p}.k_proj")
        add_output(attn.v_proj, f"{p}.v_proj")
        add_output(attn.q_norm, f"{p}.q_norm")
        add_output(attn.k_norm, f"{p}.k_norm")
        # The input of o_proj is the attention output, heads concatenated.
        add_input(attn.o_proj, f"{p}.attn_out")
        add_output(attn.o_proj, f"{p}.o_proj")
        # The input of post_attention_layernorm is x after the attention residual add.
        add_input(layer.post_attention_layernorm, f"{p}.attn_resid")
        add_output(layer.post_attention_layernorm, f"{p}.mlp_norm")
        add_output(mlp.gate_proj, f"{p}.gate")
        add_output(mlp.up_proj, f"{p}.up")
        # The input of down_proj is silu(gate) * up.
        add_input(mlp.down_proj, f"{p}.mlp_act")
        add_output(mlp.down_proj, f"{p}.down")
        add_output(layer, f"{p}.out")
    add_output(inner.norm, "final_norm")
    add_output(model.lm_head, "logits")

    # RoPE is a plain function so it can't have hooks. Wrap it.
    current = {"layer": None}

    def set_layer(i):
        def hook(module, args, kwargs):
            current["layer"] = i
        return hook

    for i, layer in enumerate(inner.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(set_layer(i), with_kwargs=True))

    original_rope = modeling_qwen3.apply_rotary_pos_emb

    def rope_and_save(q, k, cos, sin, *args, **kwargs):
        q_embed, k_embed = original_rope(q, k, cos, sin, *args, **kwargs)
        p = f"L{current['layer']:02d}"
        # q and k are [batch, heads, seq, head_dim] here; make them [batch, seq, heads, head_dim].
        acts[f"{p}.q_rope"] = to_rows(q_embed.transpose(1, 2))
        acts[f"{p}.k_rope"] = to_rows(k_embed.transpose(1, 2))
        return q_embed, k_embed

    modeling_qwen3.apply_rotary_pos_emb = rope_and_save

    def remove():
        for h in handles:
            h.remove()
        modeling_qwen3.apply_rotary_pos_emb = original_rope

    return remove


def bf16_add(a, b):
    return (a.bfloat16() + b.bfloat16()).float()


# Exact checks that each hook captured what its name says.
def check_wiring(acts, n_layers, embed_weight, ids):
    assert torch.equal(acts["embed"], embed_weight[ids].float().cpu()), "embed rows"
    x = acts["embed"]
    for i in range(n_layers):
        p = f"L{i:02d}"
        assert torch.equal(acts[f"{p}.attn_resid"], bf16_add(x, acts[f"{p}.o_proj"])), f"{p}.attn_resid"
        x = acts[f"{p}.out"]
        assert torch.equal(x, bf16_add(acts[f"{p}.attn_resid"], acts[f"{p}.down"])), f"{p}.out"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-0.6B")
    parser.add_argument("--prompts", default="ref/prompts.txt")
    parser.add_argument("--out", default="dumps")
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()

    model_dir = Path(args.model)
    out_root = Path(args.out) / model_dir.name
    prompts = [line for line in Path(args.prompts).read_text().splitlines() if line]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="eager")
    model = model.cuda().eval()
    n_layers = model.config.num_hidden_layers
    tensors_per_layer = 17
    embed_weight = model.model.embed_tokens.weight

    acts = {}
    remove_hooks = install_hooks(model, acts)
    manifests = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        acts.clear()
        with torch.no_grad():
            model(ids)
        assert len(acts) == 1 + n_layers * tensors_per_layer + 2, f"captured {len(acts)} tensors"
        check_wiring(acts, n_layers, embed_weight, ids[0])

        out_dir = out_root / f"p{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, t in acts.items():
            t.numpy().tofile(out_dir / f"{name}.bin")
        manifests.append({
            "prompt": prompt,
            "tokens": ids[0].tolist(),
            "dtype": "float32",
            "tensors": {name: list(t.shape) for name, t in acts.items()},
            "last_argmax": int(acts["logits"][-1].argmax()),
        })
        print(f"p{i:02d}: {ids.shape[1]} tokens, {len(acts)} tensors -> {out_dir}")
    remove_hooks()

    for i, m in enumerate(manifests):
        ids = torch.tensor([m["tokens"]]).cuda()
        with torch.no_grad():
            gen = model.generate(
                ids,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                return_dict_in_generate=True,
                output_logits=True,
            )
        greedy = gen.sequences[0, ids.shape[1]:].tolist()
        # A small top-2 gap means bf16 rounding can flip the argmax, which
        # explains a greedy mismatch at that step without any bug.
        gaps = []
        for step_logits in gen.logits:
            top2 = step_logits[0].float().topk(2).values
            gaps.append(round(float(top2[0] - top2[1]), 4))
        assert greedy[0] == m["last_argmax"], "first greedy token differs from the dumped logits argmax"
        m["greedy"] = greedy
        m["greedy_top2_gap"] = gaps

        out_dir = out_root / f"p{i:02d}"
        (out_dir / "manifest.json").write_text(json.dumps(m, indent=1))
        (out_dir / "tokens.txt").write_text(" ".join(map(str, m["tokens"])) + "\n")
        (out_dir / "greedy.txt").write_text(" ".join(map(str, greedy)) + "\n")
        text = tokenizer.decode(greedy)
        print(f"p{i:02d}: {m['prompt']!r} -> {text!r} (min top-2 gap {min(gaps)})")


if __name__ == "__main__":
    main()
