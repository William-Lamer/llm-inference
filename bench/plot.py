# Draws the README charts from bench/results.csv.
# usage: uv run --with matplotlib python bench/plot.py
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("bench/results.csv")
OUT = Path("docs")
FINAL = "f0420e6"
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
SPEC_GBS = 504.0

# (commit, label) for the 0.6B optimization steps, all at 256 steps after a 5-token prompt
HISTORY = [
    ("e1ea710", "original GEMV\n(thread per row)"),
    ("23a495f", "warp per row"),
    ("94f1811", "16-byte loads"),
    ("9bee197", "CUDA graph"),
    ("a5ef78e", "kernel fusion"),
]

THEMES = {
    "light": {"surface": "#ffffff", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e0",
              "engine": "#2a78d6", "llama": "#eb6834", "hf": "#1baf7a"},
    "dark": {"surface": "#0d1117", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#30363d",
             "engine": "#3987e5", "llama": "#d95926", "hf": "#199e70"},
}


def load():
    rows = {}
    with RESULTS.open() as f:
        for r in csv.DictReader(f):
            key = (r["commit"], r["model"], r["metric"], r["gen_tokens"])
            rows[key] = float(r["value"])
    return rows


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["muted"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(t, width, height, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), facecolor=t["surface"])
    return fig, axes


def save(fig, name, theme):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def label_bars(ax, bars, values, fmt, t):
    for bar, v in zip(bars, values):
        ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=t["text"])


def legend(ax, t, **kwargs):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=t["text"], **kwargs)
    return leg


def throughput(rows, t, theme):
    fig, axes = new_figure(t, 9, 3.2, ncols=3)
    names = ["this engine", "llama.cpp", "HF transformers"]
    colors = [t["engine"], t["llama"], t["hf"]]
    for ax, model in zip(axes, MODELS):
        values = [rows[(FINAL, model, "decode_tok_s", "256")],
                  rows[(FINAL, model, "llamacpp_decode_tok_s", "256")],
                  rows[(FINAL, model, "hf_decode_tok_s", "256")]]
        style(ax, t)
        bars = ax.bar(range(3), values, width=0.62, color=colors, label=names)
        label_bars(ax, bars, values, "{:.0f}", t)
        ax.set_xticks([])
        ax.set_ylim(0, max(values) * 1.15)
        ax.set_title(model.replace("Qwen3-", "Qwen3 "), color=t["text"], fontsize=11)
    axes[0].set_ylabel("decode tokens/s", color=t["muted"])
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
                     bbox_to_anchor=(0.5, -0.06))
    for text in leg.get_texts():
        text.set_color(t["text"])
    fig.text(0.5, -0.13, "RTX 4070 Super, bf16, batch 1, 5-token prompt, 256 generated tokens, "
             "median of 5 runs", ha="center", fontsize=8, color=t["muted"])
    save(fig, "throughput", theme)


def bandwidth(rows, t, theme):
    fig, ax = new_figure(t, 8, 3.8)
    style(ax, t)
    read_peak = rows[(FINAL, MODELS[0], "read_bandwidth", "256")]
    width = 0.26
    series = [("this engine", "decode_tok_s", t["engine"]),
              ("llama.cpp", "llamacpp_decode_tok_s", t["llama"]),
              ("HF transformers", "hf_decode_tok_s", t["hf"])]
    for i, (name, metric, color) in enumerate(series):
        # Same weights and cache for every engine, so bytes per step is the engine's count
        values = [rows[(FINAL, m, "bytes_per_step", "256")] * rows[(FINAL, m, metric, "256")] / 1e9
                  for m in MODELS]
        x = [j + (i - 1) * (width + 0.02) for j in range(len(MODELS))]
        bars = ax.bar(x, values, width=width, color=color, label=name)
        label_bars(ax, bars, values, "{:.0f}", t)
    # Labels at the left end, where the 0.6B bars leave room under the read-peak line
    ax.axhline(SPEC_GBS, color=t["muted"], linewidth=1)
    ax.annotate(f"spec {SPEC_GBS:.0f} GB/s", (-0.45, SPEC_GBS), xytext=(0, 3),
                textcoords="offset points", va="bottom", fontsize=8, color=t["muted"])
    ax.axhline(read_peak, color=t["muted"], linewidth=1)
    ax.annotate(f"measured read peak {read_peak:.0f} GB/s", (-0.45, read_peak), xytext=(0, -3),
                textcoords="offset points", va="top", fontsize=8, color=t["muted"])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m.replace("Qwen3-", "Qwen3 ") for m in MODELS], color=t["text"])
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_ylim(0, 560)
    ax.set_ylabel("effective bandwidth (GB/s)", color=t["muted"])
    legend(ax, t, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13))
    save(fig, "bandwidth", theme)


def history(rows, t, theme):
    fig, ax = new_figure(t, 8, 3.6)
    style(ax, t)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    model = MODELS[0]
    values = [rows[(c, model, "decode_ms_per_step", "256")] for c, _ in HISTORY]
    y = list(range(len(HISTORY)))[::-1]
    bars = ax.barh(y, values, height=0.6, color=t["engine"])
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.2f} ms", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=t["text"])
    step_bytes = rows[(FINAL, model, "bytes_per_step", "256")]
    read_peak = rows[(FINAL, model, "read_bandwidth", "256")]
    floor_ms = step_bytes / (read_peak * 1e9) * 1e3
    ax.axvline(floor_ms, color=t["muted"], linewidth=1)
    ax.annotate(f"{floor_ms:.2f} ms: every byte read at the measured peak", (floor_ms, 1.5),
                xytext=(9, 1.5), textcoords="data", fontsize=8, color=t["muted"], va="center",
                arrowprops={"arrowstyle": "-", "color": t["muted"], "linewidth": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in HISTORY], color=t["text"], fontsize=9)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(t["muted"])
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    ax.set_xlabel("ms per decode step, Qwen3 0.6B (lower is better)", color=t["muted"])
    save(fig, "history", theme)


def vs_llamacpp(rows, t, theme):
    fig, axes = new_figure(t, 8, 3.2, ncols=2)
    for ax, gen in zip(axes, ["64", "256"]):
        style(ax, t)
        values = [rows[(FINAL, m, "speedup_vs_llamacpp", gen)] for m in MODELS]
        ax.axhline(1.0, color=t["llama"], linewidth=2, label="llama.cpp = 1.00")
        ax.plot(range(len(MODELS)), values, linestyle="none", marker="o", markersize=9,
                color=t["engine"], markeredgecolor=t["surface"], markeredgewidth=2,
                label="this engine")
        for x, v in enumerate(values):
            ax.annotate(f"{v:.2f}x", (x, v), xytext=(0, 9), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, color=t["text"])
        ax.set_ylim(0.9, 1.1)
        ax.set_xlim(-0.5, len(MODELS) - 0.3)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([m.replace("Qwen3-", "") for m in MODELS], color=t["text"])
        ax.set_title(f"{gen} generated tokens", color=t["text"], fontsize=11)
    axes[0].set_ylabel("decode speed relative to llama.cpp", color=t["muted"])
    legend(axes[0], t, loc="lower left")
    save(fig, "vs-llamacpp", theme)


def main():
    rows = load()
    for theme, t in THEMES.items():
        throughput(rows, t, theme)
        bandwidth(rows, t, theme)
        history(rows, t, theme)
        vs_llamacpp(rows, t, theme)
    print(f"wrote {len(list(OUT.glob('*.png')))} charts to {OUT}/")


if __name__ == "__main__":
    main()
