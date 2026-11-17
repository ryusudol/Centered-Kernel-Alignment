"""Render the README speed chart (docs/figures/bench_{black,white}.png).

Reads the summary table printed by speed_comparison_libraries.ipynb and plots each
method's speed relative to the vanilla CPU baseline.

    python docs/benchmarks/plot_speed_chart.py [--font-dir DIR]

--font-dir should contain Inter-Regular.ttf, Inter-SemiBold.ttf and Inter-Bold.ttf
(https://github.com/rsms/inter); without it matplotlib's default font is used.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "speed_comparison_libraries.ipynb"
FIGURES = HERE.parent / "figures"

# Label shown for each benchmarked method, bottom bar last.
DISPLAY_NAMES = {
    "vanilla (CPU)": "Vanilla CKA",
    "torch-cka": "Library A",
    "pytorch-cka": "pytorch-cka",
}
BASELINE = "vanilla (CPU)"
HIGHLIGHT = "pytorch-cka"

# Canvas geometry in pixels (1950 x 356 at 100 dpi).
WIDTH, HEIGHT, DPI = 1950, 356, 100
AXES_LEFT, AXES_RIGHT = 227, 1847
GRID_TOP, GRID_BOTTOM = 40, 261
BAR_CENTERS = (90, 150, 210)
BAR_HEIGHT = 42
SPEED_LABEL_X, SPEED_LABEL_Y = 1016, 292
LONGEST_BAR_FRACTION = 0.978  # longest bar ends just short of the last gridline
N_GRID_INTERVALS = 5

GRADIENT = ("#6B10C8", "#EA405B")
THEMES = {
    # README picks bench_black.png for light mode and bench_white.png for dark mode.
    "black": {"text": "#161616", "bar": "#C8C8C8", "grid": "#EAEAEA", "axis_label": "#CECECE"},
    "white": {"text": "#FFFFFF", "bar": "#808080", "grid": "#2B2D31", "axis_label": "#595959"},
}


def read_speedups(notebook: Path) -> dict[str, float]:
    """Each method's speed relative to the baseline, from the notebook's summary table.

    Uses the "pytorch-cka speedup" column (mean / pytorch-cka mean), which the notebook
    computes from unrounded times; the printed means are rounded to 0.01 s.
    """
    cells = json.loads(notebook.read_text())["cells"]
    for cell in cells:
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", []))
            if "pytorch-cka speedup" not in text:
                continue
            slowdown = {}  # mean time relative to pytorch-cka
            for line in text.splitlines():
                for method in DISPLAY_NAMES:
                    if line.startswith(method + " "):
                        slowdown[method] = float(line.split()[-1])
            if set(slowdown) == set(DISPLAY_NAMES):
                return {m: slowdown[BASELINE] / s for m, s in slowdown.items()}
    raise ValueError(f"No complete summary table found in {notebook}")


def load_fonts(font_dir: Path | None) -> dict[str, font_manager.FontProperties]:
    files = {"regular": "Inter-Regular.ttf", "semibold": "Inter-SemiBold.ttf", "bold": "Inter-Bold.ttf"}
    if font_dir is None:
        weights = {"regular": "normal", "semibold": "semibold", "bold": "bold"}
        return {k: font_manager.FontProperties(weight=w) for k, w in weights.items()}
    return {k: font_manager.FontProperties(fname=str(font_dir / f)) for k, f in files.items()}


def sized(font: font_manager.FontProperties, size: float) -> font_manager.FontProperties:
    font = font.copy()
    font.set_size(size)
    return font


def px_x(px: float) -> float:
    return px / WIDTH


def px_y(px_from_top: float) -> float:
    return 1 - px_from_top / HEIGHT


def render(speedups: dict[str, float], theme: dict[str, str], fonts, out: Path) -> None:
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_alpha(0)

    ax = fig.add_axes(
        [px_x(AXES_LEFT), px_y(GRID_BOTTOM), px_x(AXES_RIGHT - AXES_LEFT), (GRID_BOTTOM - GRID_TOP) / HEIGHT]
    )
    ax.patch.set_alpha(0)
    x_max = max(speedups.values()) / LONGEST_BAR_FRACTION
    ax.set_xlim(0, x_max)
    ax.set_ylim(GRID_BOTTOM, GRID_TOP)  # data y == pixel y from the top
    ax.axis("off")

    for k in range(N_GRID_INTERVALS + 1):
        ax.axvline(x_max * k / N_GRID_INTERVALS, color=theme["grid"], linewidth=1.5, zorder=0)

    gradient = LinearSegmentedColormap.from_list("bench", GRADIENT)
    for center, (method, label) in zip(BAR_CENTERS, DISPLAY_NAMES.items()):
        value = speedups[method]
        top, bottom = center - BAR_HEIGHT / 2, center + BAR_HEIGHT / 2
        highlight = method == HIGHLIGHT
        if highlight:
            ax.imshow(
                np.linspace(0, 1, 512)[None, :],
                cmap=gradient,
                extent=(0, value, bottom, top),
                aspect="auto",
                interpolation="bicubic",
                zorder=2,
            )
        else:
            ax.fill_between([0, value], top, bottom, color=theme["bar"], linewidth=0, zorder=2)

        weight = "bold" if highlight else "regular"
        fig.text(
            px_x(AXES_LEFT - 22), px_y(center), label,
            ha="right", va="center_baseline", color=theme["text"], fontproperties=sized(fonts[weight], 20),
        )
        ax.text(
            value + x_max * 0.012, center, f"{value:.1f}x",
            ha="left", va="center_baseline", color=theme["text"], fontproperties=sized(fonts[weight], 20),
        )

    fig.text(
        px_x(SPEED_LABEL_X), px_y(SPEED_LABEL_Y), "Speed",
        ha="center", va="center_baseline", color=theme["axis_label"], fontproperties=sized(fonts["semibold"], 26),
    )

    fig.savefig(out, dpi=DPI, transparent=True)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    parser.add_argument("--font-dir", type=Path, default=None)
    args = parser.parse_args()

    speedups = read_speedups(args.notebook)
    print("Speed vs. vanilla CPU:", {m: round(s, 2) for m, s in speedups.items()})

    fonts = load_fonts(args.font_dir)
    for name, theme in THEMES.items():
        out = args.out_dir / f"bench_{name}.png"
        render(speedups, theme, fonts, out)
        print("Wrote", out)


if __name__ == "__main__":
    main()
