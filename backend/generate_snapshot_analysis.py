from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, median, pstdev
from xml.sax.saxutils import escape


ITER_RE = re.compile(r"iter_(\d+)_(.+)")


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(results_dir.glob("*_summary.csv")):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            try:
                row = next(reader)
            except StopIteration:
                continue

        snapshot = csv_path.name.removesuffix("_summary.csv")
        match = ITER_RE.fullmatch(snapshot)
        if not match:
            continue

        iteration = int(match.group(1))
        tag = match.group(2)
        wins = float(row["score_a"])
        losses = float(row["score_b"])
        games = int(float(row["games"]))
        win_rate = 100.0 * wins / games if games else 0.0

        rows.append(
            {
                "snapshot": snapshot,
                "iteration": iteration,
                "tag": tag,
                "group": "Candidate" if tag == "candidate" else "Champion",
                "agent": row["agent_a"],
                "opponent": row["agent_b"],
                "wins": wins,
                "losses": losses,
                "games": games,
                "win_rate": win_rate,
            }
        )

    if not rows:
        raise SystemExit(f"No snapshot summaries found in {results_dir}")

    rows.sort(key=lambda row: (row["iteration"], row["tag"]))
    return rows


def save_ranked_csv(rows: list[dict], out_path: Path) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (-row["win_rate"], -row["wins"], row["losses"], row["snapshot"]),
    )
    fieldnames = [
        "snapshot",
        "iteration",
        "tag",
        "group",
        "agent",
        "opponent",
        "wins",
        "losses",
        "games",
        "win_rate",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)


def make_svg_plot(rows: list[dict], out_path: Path, title: str) -> None:
    group_styles = {
        "Candidate": {"color": "#2b6cb0", "label": "Candidate"},
        "Champion": {"color": "#2f855a", "label": "Champion"},
    }
    width, height = 1100, 620
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 70, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    iterations = [row["iteration"] for row in rows]
    win_rates = [row["win_rate"] for row in rows]
    min_iter, max_iter = min(iterations), max(iterations)
    min_wr = min(win_rates) - 2
    max_wr = max(win_rates) + 4
    avg_wr = mean(win_rates)

    def x_map(value: float) -> float:
        if max_iter == min_iter:
            return margin_left + plot_w / 2
        return margin_left + (value - min_iter) / (max_iter - min_iter) * plot_w

    def y_map(value: float) -> float:
        if max_wr == min_wr:
            return margin_top + plot_h / 2
        return margin_top + plot_h - (value - min_wr) / (max_wr - min_wr) * plot_h

    grid_values = list(range(int(min_wr // 5 * 5), int(max_wr) + 1, 5))
    polyline_points = " ".join(f"{x_map(r['iteration']):.1f},{y_map(r['win_rate']):.1f}" for r in rows)
    best = max(rows, key=lambda row: row["win_rate"])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="32" text-anchor="middle" font-size="24" font-family="Arial" fill="#1a202c">{escape(title)}</text>',
    ]

    for gv in grid_values:
        y = y_map(gv)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left-12}" y="{y+5:.1f}" text-anchor="end" font-size="14" font-family="Arial" fill="#4a5568">{gv}</text>')

    parts.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#1a202c" stroke-width="2"/>',
            f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#1a202c" stroke-width="2"/>',
        ]
    )

    tick_step = max(1, (max_iter - min_iter) // 8)
    for tick in range(min_iter, max_iter + 1, tick_step):
        x = x_map(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{height-margin_bottom}" x2="{x:.1f}" y2="{height-margin_bottom+6}" stroke="#1a202c" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-margin_bottom+24}" text-anchor="middle" font-size="14" font-family="Arial" fill="#4a5568">{tick}</text>')

    parts.append(f'<polyline points="{polyline_points}" fill="none" stroke="#a0aec0" stroke-width="2" opacity="0.8"/>')
    parts.append(
        f'<line x1="{margin_left}" y1="{y_map(avg_wr):.1f}" x2="{width-margin_right}" y2="{y_map(avg_wr):.1f}" '
        f'stroke="#718096" stroke-width="2" stroke-dasharray="7 6"/>'
    )

    for row in rows:
        style = group_styles.get(row["group"], {"color": "#4a5568", "label": row["group"]})
        parts.append(
            f'<circle cx="{x_map(row["iteration"]):.1f}" cy="{y_map(row["win_rate"]):.1f}" r="5.5" '
            f'fill="{style["color"]}" fill-opacity="0.9"/>'
        )

    bx, by = x_map(best["iteration"]), y_map(best["win_rate"])
    parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="8" fill="none" stroke="#1a202c" stroke-width="2"/>')
    label = f"Best: {best['snapshot']} ({best['win_rate']:.0f}%)"
    parts.append(
        f'<rect x="{bx+12:.1f}" y="{by-28:.1f}" width="275" height="28" fill="white" stroke="#cbd5e0" rx="4"/>'
    )
    parts.append(
        f'<text x="{bx+20:.1f}" y="{by-9:.1f}" font-size="13" font-family="Arial" fill="#1a202c">{escape(label)}</text>'
    )

    legend_x = margin_left + 10
    legend_y = 48
    legend_items = [("Candidate", "#2b6cb0"), ("Champion", "#2f855a"), (f"Mean ({avg_wr:.1f}%)", "#718096")]
    for idx, (label_text, color) in enumerate(legend_items):
        x = legend_x + idx * 260
        if "Mean" in label_text:
            parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+24}" y2="{legend_y}" stroke="{color}" stroke-width="2" stroke-dasharray="7 6"/>')
        else:
            parts.append(f'<circle cx="{x+10}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{x+32}" y="{legend_y+5}" font-size="14" font-family="Arial" fill="#2d3748">{escape(label_text)}</text>')

    parts.append(
        f'<text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="16" font-family="Arial" fill="#1a202c">Training iteration</text>'
    )
    parts.append(
        f'<text x="24" y="{height/2:.1f}" text-anchor="middle" font-size="16" font-family="Arial" fill="#1a202c" transform="rotate(-90 24 {height/2:.1f})">Win rate vs Greedy (%)</text>'
    )
    parts.append("</svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")


def print_summary(rows: list[dict], top_n: int) -> None:
    win_rates = [row["win_rate"] for row in rows]
    ranked = sorted(
        rows,
        key=lambda row: (-row["win_rate"], -row["wins"], row["losses"], row["snapshot"]),
    )
    print(f"Loaded {len(rows)} snapshot evaluations")
    print(
        "Summary: "
        f"min={min(win_rates):.1f}% "
        f"max={max(win_rates):.1f}% "
        f"mean={mean(win_rates):.2f}% "
        f"median={median(win_rates):.1f}% "
        f"std={pstdev(win_rates):.2f}"
    )
    print("\nTop snapshots:")
    for idx, row in enumerate(ranked[:top_n], start=1):
        print(
            f"{idx:>2}. {row['snapshot']:<28} "
            f"{row['win_rate']:>5.1f}% "
            f"({row['wins']:.0f}-{row['losses']:.0f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ranked tables and plots for snapshot evaluations.")
    parser.add_argument("results_dir", help="Directory containing *_summary.csv snapshot results.")
    parser.add_argument("--csv-out", default="results/snapshot_ranked.csv", help="Output CSV for ranked snapshots.")
    parser.add_argument("--plot-out", default="charts/snapshot_winrate_progress.svg", help="Output plot path.")
    parser.add_argument("--title", default="Checkpoint win rate vs Greedy over training", help="Plot title.")
    parser.add_argument("--top", type=int, default=10, help="Number of top snapshots to print.")
    args = parser.parse_args()

    rows = load_rows(Path(args.results_dir))
    csv_out = Path(args.csv_out)
    plot_out = Path(args.plot_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    plot_out.parent.mkdir(parents=True, exist_ok=True)

    save_ranked_csv(rows, csv_out)
    make_svg_plot(rows, plot_out, args.title)
    print_summary(rows, args.top)
    print(f"\nSaved ranked CSV to {csv_out}")
    print(f"Saved plot to {plot_out}")


if __name__ == "__main__":
    main()
