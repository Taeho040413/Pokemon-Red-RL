#!/usr/bin/env python3
"""Render docs/assets/pokemon_red_rl_structure_overview.png (three-column overview)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Prefer CJK-capable font on typical Linux images
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def _font_prop() -> fm.FontProperties:
    for p in _FONT_CANDIDATES:
        if Path(p).is_file():
            return fm.FontProperties(fname=p)
    return fm.FontProperties(family="sans-serif")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "pokemon_red_rl_structure_overview.png"
    fp = _font_prop()

    fig_w, fig_h = 18, 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")

    title = "Pokemon Red RL (PufferLib) — 구조 개요"
    ax.text(
        fig_w / 2,
        fig_h - 0.55,
        title,
        fontsize=22,
        ha="center",
        va="top",
        fontproperties=fp,
        color="#1a1a2e",
        weight="bold",
    )

    col_w = 5.15
    gap = 0.45
    x0 = 0.65
    y_panel = 1.0
    h_panel = fig_h - 2.15
    radius = 0.25

    columns = [
        (
            "환경 — PyBoy / RedGymEnv",
            (
                "관측 (Dict)\n"
                "─────────────────\n"
                "• 시각·공간: screen, visited_mask\n"
                "• (선택) global_map\n"
                "• 상태: direction, map_id,\n"
                "  blackout_map_id, battle_type\n"
                "• 가방: bag_items, bag_quantity\n"
                "• 파티(6): species, hp, stats,\n"
                "  types, level, moves …\n"
                "• 진행: events (비트열)\n"
                "\n"
                "info / 로그 (관측 아님)\n"
                "─────────────────\n"
                "• 예: game_corner_rocket\n"
                "• safari_zone 등\n"
                "\n"
                "출력 → 액션 Discrete(7)\n"
                "(↓ ← → ↑  A  B  START)"
            ),
            "#e8f4fd",
            "#1565c0",
        ),
        (
            "정책 — MultiConvolutionalPolicy + RNN",
            (
                "인코딩\n"
                "─────────────────\n"
                "• CNN: screen ∥ visited_mask\n"
                "• 임베딩: map, blackout, 가방,\n"
                "  파티(종족·타입·기술)\n"
                "• one-hot: direction, battle_type\n"
                "• events: 비트 → EVENTS_IDXS\n"
                "• (선택) global_map CNN\n"
                "\n"
                "concat → encode_linear → z\n"
                "\n"
                "헤드\n"
                "─────────────────\n"
                "• Value: V(s)\n"
                "• Actor: 7 logits\n"
                "\n"
                "※ HM 전용 서브헤드 없음\n"
                "(auto_teach / auto_use는 환경)"
            ),
            "#f3e5f5",
            "#6a1b9a",
        ),
        (
            "보상 — ExplorationInteractionRewardEnv",
            (
                "계층\n"
                "─────────────────\n"
                "BaselineRewardEnv → RedGymEnv\n"
                "\n"
                "누적 dict\n"
                "─────────────────\n"
                "get_game_state_reward()\n"
                "→ event, new_tile, new_building,\n"
                "  wild_encounter_penalty, …\n"
                "\n"
                "스텝 스칼라\n"
                "─────────────────\n"
                "update_reward:\n"
                "  step = sum(새) − sum(이전)\n"
                "\n"
                "가중치: config.yaml\n"
                "reward.baseline.\n"
                "ExplorationInteractionRewardEnv\n"
                "\n"
                "※ Reward Machine(rm_*) 없음"
            ),
            "#e8f5e9",
            "#2e7d32",
        ),
    ]

    for i, (col_title, body, bg, edge) in enumerate(columns):
        x = x0 + i * (col_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y_panel),
            col_w,
            h_panel,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=2.2,
            edgecolor=edge,
            facecolor=bg,
            alpha=0.95,
        )
        ax.add_patch(rect)
        ax.text(
            x + col_w / 2,
            y_panel + h_panel - 0.38,
            col_title,
            fontsize=13.5,
            ha="center",
            va="top",
            fontproperties=fp,
            color=edge,
            weight="bold",
        )
        ax.text(
            x + 0.28,
            y_panel + h_panel - 0.95,
            body,
            fontsize=10.2,
            ha="left",
            va="top",
            fontproperties=fp,
            color="#222",
            linespacing=1.35,
        )

    # Flow arrows between columns
    y_arrow = y_panel + 0.35
    for i in range(2):
        xa = x0 + (i + 1) * col_w + i * gap
        ax.annotate(
            "",
            xy=(xa + gap + 0.05, y_arrow),
            xytext=(xa - 0.02, y_arrow),
            arrowprops=dict(arrowstyle="->", color="#546e7a", lw=2.5),
        )

    ax.text(
        fig_w / 2,
        0.35,
        "obs → 정책 → action → 환경(PyBoy) → 보상(Δ)",
        fontsize=11,
        ha="center",
        fontproperties=fp,
        color="#455a64",
        style="italic",
    )

    plt.tight_layout(pad=0.2)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.patch.get_facecolor())
    plt.close()
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
