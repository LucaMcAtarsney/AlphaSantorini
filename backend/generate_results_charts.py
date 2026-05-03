from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = ROOT / "charts"
PALETTE = ["#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#99B898", "#547980"]
AGENT_DISPLAY = {
    "Random": "Random",
    "Greedy": "Greedy",
    "Heuristic": "Heuristic",
    "Minimax-d2": "Minimax",
    "Vanilla-MCTS": "MCTS",
    "MCTS-Heuristic": "Heuristic MCTS",
    "MCTS-NN": "NN MCTS",
}
CANONICAL_AGENT_ORDER = [
    "Random",
    "Greedy",
    "Heuristic",
    "Minimax-d2",
    "Vanilla-MCTS",
    "MCTS-Heuristic",
    "MCTS-NN",
]


def load_summary(filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / filename)


def load_games(filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / filename)


def savefig(filename: str) -> None:
    plt.savefig(CHARTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def setup_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Serif",
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#333333",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.8,
        },
    )


def style_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.8)
    ax.grid(axis="both" if grid_axis == "both" else ("y" if grid_axis == "x" else "x"), visible=False)


def palette_for(n: int) -> List[str]:
    if n <= len(PALETTE):
        return PALETTE[:n]
    extra = sns.color_palette("crest", n - len(PALETTE)).as_hex()
    return PALETTE + list(extra)


def ordered_agents(agents: Iterable[str]) -> List[str]:
    agent_set = set(agents)
    ordered = [agent for agent in CANONICAL_AGENT_ORDER if agent in agent_set]
    extras = sorted(agent_set - set(ordered))
    return ordered + extras


def display_agent(name: str) -> str:
    return AGENT_DISPLAY.get(name, name)


def compute_agent_stats_from_games(df: pd.DataFrame) -> pd.DataFrame:
    wins: Dict[str, float] = defaultdict(float)
    games: Dict[str, int] = defaultdict(int)
    think: Dict[str, float] = defaultdict(float)
    decisions: Dict[str, int] = defaultdict(int)

    agents = sorted(set(df["agent_a"]).union(df["agent_b"]))

    for _, row in df.iterrows():
        agent_a = row["agent_a"]
        agent_b = row["agent_b"]
        winner = row["winner"]

        games[agent_a] += 1
        games[agent_b] += 1

        think[agent_a] += float(row["think_time_a"])
        think[agent_b] += float(row["think_time_b"])

        decisions[agent_a] += int(row["decisions_a"])
        decisions[agent_b] += int(row["decisions_b"])

        if winner == "draw":
            wins[agent_a] += 0.5
            wins[agent_b] += 0.5
        else:
            wins[winner] += 1.0

    stats = pd.DataFrame(
        {
            "agent": agents,
            "wins": [wins[a] for a in agents],
            "games": [games[a] for a in agents],
            "win_rate": [100.0 * wins[a] / games[a] if games[a] else 0.0 for a in agents],
            "avg_move_time_ms": [
                1000.0 * think[a] / decisions[a] if decisions[a] else 0.0
                for a in agents
            ],
        }
    )
    stats["agent_display"] = stats["agent"].map(display_agent)
    return stats.sort_values("win_rate", ascending=False)


def build_heatmap(summary_df: pd.DataFrame) -> pd.DataFrame:
    agents = ordered_agents(set(summary_df["agent_a"]).union(summary_df["agent_b"]))
    heatmap = pd.DataFrame(np.nan, index=agents, columns=agents)

    for _, row in summary_df.iterrows():
        agent_a = row["agent_a"]
        agent_b = row["agent_b"]
        heatmap.loc[agent_a, agent_b] = float(row["score_a"])
        heatmap.loc[agent_b, agent_a] = float(row["score_b"])

    return heatmap


def create_heatmap(summary_df: pd.DataFrame, title: str, filename: str) -> None:
    heatmap = build_heatmap(summary_df)
    heatmap.index = [display_agent(agent) for agent in heatmap.index]
    heatmap.columns = [display_agent(agent) for agent in heatmap.columns]
    fig, ax = plt.subplots(figsize=(8, 6.8), constrained_layout=True)
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Win Rate (%)"},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    savefig(filename)


def create_sim_sweep_charts() -> None:
    sim_budgets = [50, 100, 200, 400, 800]
    stats_frames = []
    durations = []

    for sims in sim_budgets:
        games_df = load_games(f"sim_sweep_{sims}_games.csv")
        stats_df = compute_agent_stats_from_games(games_df)
        stats_df["sims"] = sims
        stats_frames.append(stats_df)
        durations.append(
            {
                "sims": sims,
                "avg_game_duration": games_df["game_duration"].mean(),
            }
        )

    sim_df = pd.concat(stats_frames, ignore_index=True)
    duration_df = pd.DataFrame(durations)
    hue_order = ordered_agents(sim_df["agent"].unique())
    sim_df["agent_display"] = sim_df["agent"].map(display_agent)
    display_hue_order = [display_agent(agent) for agent in hue_order]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    sns.lineplot(data=sim_df, x="sims", y="win_rate", hue="agent_display", hue_order=display_hue_order, marker="o", linewidth=2.2, palette=palette_for(sim_df["agent"].nunique()), ax=ax)
    ax.set_title("T4 Simulation Budget Sweep: Win Rate vs Simulation Count", pad=12)
    ax.set_xlabel("MCTS Simulations")
    ax.set_ylabel("Win Rate (%)")
    ax.legend(frameon=False, title=None, loc="best")
    style_axes(ax, grid_axis="y")
    savefig("t4_sim_sweep_winrate.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    sns.lineplot(data=sim_df, x="sims", y="avg_move_time_ms", hue="agent_display", hue_order=display_hue_order, marker="o", linewidth=2.2, palette=palette_for(sim_df["agent"].nunique()), ax=ax)
    ax.set_title("T4 Simulation Budget Sweep: Average Move Time vs Simulation Count", pad=12)
    ax.set_xlabel("MCTS Simulations")
    ax.set_ylabel("Average Move Time (ms)")
    ax.legend(frameon=False, title=None, loc="best")
    style_axes(ax, grid_axis="y")
    savefig("t4_sim_sweep_move_time.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    sns.lineplot(data=duration_df, x="sims", y="avg_game_duration", marker="o", color="#355C7D", linewidth=2.2, ax=ax)
    ax.set_title("T4 Simulation Budget Sweep: Average Game Duration", pad=12)
    ax.set_xlabel("MCTS Simulations")
    ax.set_ylabel("Average Game Duration (s)")
    style_axes(ax, grid_axis="y")
    savefig("t4_sim_sweep_game_duration.png")


def create_nn_vs_baselines_chart() -> None:
    opponents = ["Random", "Greedy", "Heuristic", "Minimax-d2"]
    rows = []

    for opponent in opponents:
        summary_df = load_summary(f"nn_vs_{opponent}_summary.csv")
        score = float(summary_df.iloc[0]["score_a"])
        rows.append({"opponent": opponent, "opponent_display": display_agent(opponent), "nn_win_rate": score})

    nn_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    sns.barplot(data=nn_df, x="opponent_display", y="nn_win_rate", hue="opponent_display", order=[display_agent(o) for o in opponents], palette=PALETTE[:4], legend=False, ax=ax)
    ax.set_title("T5 MCTS-NN: Win Rate Against Baselines", pad=12)
    ax.set_xlabel("Opponent")
    ax.set_ylabel("MCTS-NN Win Rate (%)")
    ax.set_ylim(0, 100)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
    style_axes(ax, grid_axis="y")
    savefig("t5_nnmcts_vs_baselines_winrate.png")


def main() -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    setup_theme()

    t1_summary = load_summary("full_tournament_summary.csv")
    create_heatmap(t1_summary, "T1 Full Tournament: Head-to-Head Win Rates (%)", "t1_full_heatmap.png")

    t2_summary = load_summary("baseline_summary.csv")
    create_heatmap(t2_summary, "T2 Baseline Agents: Head-to-Head Win Rates (%)", "t2_baseline_heatmap.png")

    t3_summary = load_summary("search_agents_summary.csv")
    create_heatmap(t3_summary, "T3 Search Agents: Head-to-Head Win Rates (%)", "t3_search_heatmap.png")

    create_sim_sweep_charts()
    create_nn_vs_baselines_chart()

    print(f"Charts generated in: {CHARTS_DIR}")


if __name__ == "__main__":
    main()