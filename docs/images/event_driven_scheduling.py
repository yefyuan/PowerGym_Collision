"""
Publication-quality figure: heterogeneous event-driven execution challenges.

Illustrates the THREE concrete failure modes from the paper:
  1. Observation staleness / physics drift — env evolves between obs and effect
  2. Action delay — visible compute + effect delay pipeline
  3. Late coordination — directive arrives after recipient already acted

Layout:
  - Top row:    Environment / Physics timeline (continuous state evolution)
  - Middle row: Agent 1 (full obs → compute → effect pipeline with delays)
  - Bottom row: Agent 2 (late-arriving message + stale action)

NeurIPS 2024 style: 5.5 in width, serif font, Okabe-Ito palette.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── NeurIPS dimensions ──────────────────────────────────────────────────────
TEXT_WIDTH = 5.5
FIG_H = TEXT_WIDTH * 0.75  # enough vertical room for 3 rows + annotations

# ── Okabe-Ito palette ──────────────────────────────────────────────────────
OI_BLUE      = "#0072B2"   # Agent 1 / obs
OI_VERMILION = "#D55E00"   # Agent 2
OI_ORANGE    = "#E69F00"   # warnings
OI_SKY       = "#56B4E9"   # environment
OI_GREEN     = "#009E73"   # action effects
OI_PURPLE    = "#CC79A7"   # messages

# ── Style ───────────────────────────────────────────────────────────────────
BG            = "#EEF2F7"
TEXT_DARK     = "#2D3748"
TIMELINE_GREY = "#C3CBDA"
DORMANT_GREY  = "#8B95A5"
BANNER_BG     = "#FDDEDE"
BANNER_EDGE   = "#F5B3B3"
BANNER_TEXT   = "#8B2020"
WARN_BG       = "#FFF3CD"
WARN_EDGE     = "#F0D78C"
WARN_TEXT     = "#856404"

# ── Layout ──────────────────────────────────────────────────────────────────
TL_LEFT, TL_RIGHT = 0.15, 0.95
LABEL_X = 0.06

ROW_ENV = 0.855
ROW_A1  = 0.555
ROW_A2  = 0.215

NODE_S = 15 ** 2   # main node area
SM_S   = 11 ** 2   # smaller nodes


def tx(frac):
    return TL_LEFT + frac * (TL_RIGHT - TL_LEFT)


# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(TEXT_WIDTH, FIG_H))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1.0)
ax.set_ylim(0.06, 1.0)
ax.axis("off")

# ── Title banner ────────────────────────────────────────────────────────────
banner = FancyBboxPatch(
    (0.06, 0.935), 0.88, 0.048,
    boxstyle="round,pad=0.010",
    facecolor=BANNER_BG, edgecolor=BANNER_EDGE, linewidth=0.7,
)
ax.add_patch(banner)
ax.text(
    0.50, 0.959,
    "Heterogeneous event-driven execution at deployment",
    ha="center", va="center",
    fontsize=8.5, fontweight="bold", color=BANNER_TEXT, fontfamily="serif",
)

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT / PHYSICS ROW
# ═══════════════════════════════════════════════════════════════════════════
ax.text(LABEL_X, ROW_ENV, "Env", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=OI_SKY, fontfamily="serif")

ax.annotate("", xy=(TL_RIGHT + 0.015, ROW_ENV), xytext=(TL_LEFT - 0.008, ROW_ENV),
            arrowprops=dict(arrowstyle="->, head_width=0.13, head_length=0.08",
                            color=OI_SKY, lw=1.1))
ax.text(TL_RIGHT + 0.03, ROW_ENV, "$t$", fontsize=9, color=TEXT_DARK,
        va="center", fontfamily="serif")

# Physics state markers
phys_fracs = [0.0, 0.10, 0.22, 0.34, 0.44, 0.55, 0.66, 0.78, 0.90]
for i, f in enumerate(phys_fracs):
    x = tx(f)
    ax.plot(x, ROW_ENV, "|", color=OI_SKY, markersize=4.5, markeredgewidth=0.7)
    ax.text(x, ROW_ENV + 0.020, f"$s_{{{i}}}$", ha="center", va="bottom",
            fontsize=4.5, color=OI_SKY, fontfamily="serif")

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1 ROW
# ═══════════════════════════════════════════════════════════════════════════
ax.text(LABEL_X, ROW_A1, "Agent 1", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=TEXT_DARK, fontfamily="serif")
ax.plot([TL_LEFT, TL_RIGHT], [ROW_A1, ROW_A1], color=TIMELINE_GREY, lw=0.4)

# --- Full pipeline: obs(s₁) → compute → effect(applied at s₄) ---
obs1_x   = tx(0.10)
comp1_x  = tx(0.24)
effect_x = tx(0.44)

# (a) Obs node
ax.scatter(obs1_x, ROW_A1, s=NODE_S, c=OI_BLUE, edgecolors="white",
           linewidths=0.8, zorder=4, marker="o")
ax.text(obs1_x, ROW_A1, "obs", ha="center", va="center",
        fontsize=4.5, color="white", fontweight="bold", zorder=5, fontfamily="serif")

# Dotted line from env → agent (state captured)
ax.annotate("", xy=(obs1_x, ROW_A1 + 0.025),
            xytext=(obs1_x, ROW_ENV - 0.025),
            arrowprops=dict(arrowstyle="->, head_width=0.08, head_length=0.05",
                            color=OI_SKY, lw=0.5, linestyle=":"), zorder=2)
ax.text(obs1_x - 0.018, (ROW_A1 + ROW_ENV) / 2 + 0.02, "sees $s_1$",
        fontsize=4.5, color=OI_SKY, fontfamily="serif", fontstyle="italic",
        ha="right", va="center")

# (b) Compute bracket (above the row)
bracket_y = ROW_A1 + 0.032
ax.annotate("", xy=(comp1_x, bracket_y), xytext=(obs1_x, bracket_y),
            arrowprops=dict(arrowstyle="<->", color=DORMANT_GREY, lw=0.5))
ax.text((obs1_x + comp1_x) / 2, bracket_y + 0.012, "compute",
        ha="center", va="bottom", fontsize=4, color=DORMANT_GREY,
        fontstyle="italic", fontfamily="serif")

# (c) Effect delay bracket (above the row)
ax.annotate("", xy=(effect_x, bracket_y), xytext=(comp1_x, bracket_y),
            arrowprops=dict(arrowstyle="<->", color=DORMANT_GREY, lw=0.5))
ax.text((comp1_x + effect_x) / 2, bracket_y + 0.012, "effect delay",
        ha="center", va="bottom", fontsize=4, color=DORMANT_GREY,
        fontstyle="italic", fontfamily="serif")

# (d) Action-effect diamond
ax.scatter(effect_x, ROW_A1, s=NODE_S * 1.1, c=OI_GREEN, edgecolors="white",
           linewidths=0.8, zorder=4, marker="D")
ax.text(effect_x, ROW_A1, "eff", ha="center", va="center",
        fontsize=4, color="white", fontweight="bold", zorder=5, fontfamily="serif")

# Dotted line from agent → env (action applied)
ax.annotate("", xy=(effect_x, ROW_ENV - 0.025),
            xytext=(effect_x, ROW_A1 + 0.025),
            arrowprops=dict(arrowstyle="->, head_width=0.08, head_length=0.05",
                            color=OI_GREEN, lw=0.5, linestyle=":"), zorder=2)
ax.text(effect_x + 0.018, (ROW_A1 + ROW_ENV) / 2 + 0.02, "hits $s_4$",
        fontsize=4.5, color=OI_GREEN, fontfamily="serif", fontstyle="italic",
        ha="left", va="center")

# --- FAILURE MODE 1: Physics drift warning (below Agent 1 row) ---
warn1_y = ROW_A1 - 0.045
warn1_box = FancyBboxPatch(
    (obs1_x - 0.008, warn1_y - 0.016), effect_x - obs1_x + 0.016, 0.028,
    boxstyle="round,pad=0.004",
    facecolor=WARN_BG, edgecolor=WARN_EDGE, linewidth=0.5, zorder=3,
)
ax.add_patch(warn1_box)
ax.text((obs1_x + effect_x) / 2, warn1_y,
        "physics drifted $s_1 \\!\\to\\! s_4$ between obs and effect",
        ha="center", va="center", fontsize=4.2, color=WARN_TEXT,
        fontweight="semibold", fontfamily="serif", zorder=4)

# --- Agent 1 second event: alarm → act + sends message ---
alarm_x = tx(0.66)
ax.scatter(alarm_x, ROW_A1, s=NODE_S, c=OI_BLUE, edgecolors="white",
           linewidths=0.8, zorder=4)
ax.text(alarm_x, ROW_A1, "act", ha="center", va="center",
        fontsize=4.5, color="white", fontweight="bold", zorder=5, fontfamily="serif")
ax.text(alarm_x, ROW_A1 + 0.032, "alarm", ha="center", va="bottom",
        fontsize=5, color=OI_BLUE, fontweight="semibold", fontfamily="serif")
ax.annotate("", xy=(alarm_x, ROW_A1 + 0.023), xytext=(alarm_x, ROW_A1 + 0.030),
            arrowprops=dict(arrowstyle="->, head_width=0.06, head_length=0.03",
                            color=OI_BLUE, lw=0.5))

# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2 ROW
# ═══════════════════════════════════════════════════════════════════════════
ax.text(LABEL_X, ROW_A2, "Agent 2", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=TEXT_DARK, fontfamily="serif")
ax.plot([TL_LEFT, TL_RIGHT], [ROW_A2, ROW_A2], color=TIMELINE_GREY, lw=0.4)

# Dormant period
ax.plot([TL_LEFT + 0.005, tx(0.42)], [ROW_A2, ROW_A2],
        color=DORMANT_GREY, lw=0.7, linestyle=(0, (4, 3)), zorder=1)
ax.text((TL_LEFT + tx(0.42)) / 2, ROW_A2 - 0.020, "dormant",
        ha="center", va="top", fontsize=4.5, color=DORMANT_GREY,
        fontstyle="italic", fontfamily="serif")

# Agent 2 obs + act
a2_obs_x = tx(0.45)
a2_act_x = tx(0.56)

ax.scatter(a2_obs_x, ROW_A2, s=SM_S, c=OI_VERMILION, edgecolors="white",
           linewidths=0.8, zorder=4)
ax.text(a2_obs_x, ROW_A2, "obs", ha="center", va="center",
        fontsize=3.5, color="white", fontweight="bold", zorder=5, fontfamily="serif")

ax.scatter(a2_act_x, ROW_A2, s=NODE_S, c=OI_VERMILION, edgecolors="white",
           linewidths=0.8, zorder=4)
ax.text(a2_act_x, ROW_A2, "act", ha="center", va="center",
        fontsize=4.5, color="white", fontweight="bold", zorder=5, fontfamily="serif")

# Connect obs → act
ax.plot([a2_obs_x + 0.012, a2_act_x - 0.012], [ROW_A2, ROW_A2],
        color=OI_VERMILION, lw=0.7, zorder=2)

# Trigger label
ax.text(a2_obs_x, ROW_A2 + 0.032, "timer", ha="center", va="bottom",
        fontsize=5, color=OI_VERMILION, fontweight="semibold", fontfamily="serif")
ax.annotate("", xy=(a2_obs_x, ROW_A2 + 0.023), xytext=(a2_obs_x, ROW_A2 + 0.030),
            arrowprops=dict(arrowstyle="->, head_width=0.06, head_length=0.03",
                            color=OI_VERMILION, lw=0.5))

# --- Message from Agent 1 → Agent 2 (arrives LATE) ---
msg_recv_x = tx(0.88)
ax.annotate(
    "", xy=(msg_recv_x, ROW_A2 + 0.022),
    xytext=(alarm_x, ROW_A1 - 0.022),
    arrowprops=dict(
        arrowstyle="->, head_width=0.10, head_length=0.06",
        connectionstyle="arc3,rad=0.10",
        color=OI_PURPLE, lw=0.8, linestyle="--",
    ), zorder=3,
)

# Message label
msg_label_x = (alarm_x + msg_recv_x) / 2 + 0.025
msg_label_y = (ROW_A1 + ROW_A2) / 2
ax.text(msg_label_x, msg_label_y, "directive\n(network latency)",
        ha="center", va="center", fontsize=4, color=OI_PURPLE,
        fontstyle="italic", fontfamily="serif", linespacing=1.3)

# Message received square
ax.scatter(msg_recv_x, ROW_A2, s=SM_S, c=OI_PURPLE, edgecolors="white",
           linewidths=0.8, zorder=4, marker="s")
ax.text(msg_recv_x, ROW_A2, "rcv", ha="center", va="center",
        fontsize=3, color="white", fontweight="bold", zorder=5, fontfamily="serif")

# --- FAILURE MODE 2: Late coordination warning ---
warn2_y = ROW_A2 - 0.042
warn2_box = FancyBboxPatch(
    (a2_act_x - 0.008, warn2_y - 0.015), msg_recv_x - a2_act_x + 0.016, 0.026,
    boxstyle="round,pad=0.004",
    facecolor=WARN_BG, edgecolor=WARN_EDGE, linewidth=0.5, zorder=3,
)
ax.add_patch(warn2_box)
ax.text((a2_act_x + msg_recv_x) / 2, warn2_y,
        "directive arrived after Agent 2 already acted",
        ha="center", va="center", fontsize=4.2, color=WARN_TEXT,
        fontweight="semibold", fontfamily="serif", zorder=4)

# ═══════════════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════════════
legend_y = 0.090
items = [
    ("o",  OI_BLUE,      "observe / decide"),
    ("D",  OI_GREEN,     "action takes effect"),
    ("s",  OI_PURPLE,    "message received"),
    ("|",  OI_SKY,       "physics update"),
]
lx_start = 0.13
lx_step = 0.22

for i, (marker, color, label) in enumerate(items):
    lx = lx_start + i * lx_step
    if marker == "|":
        ax.plot(lx, legend_y, marker, color=color, markersize=5.5, markeredgewidth=0.8)
    else:
        ax.scatter(lx, legend_y, s=45, c=color, marker=marker,
                   edgecolors="white", linewidths=0.4, zorder=4)
    ax.text(lx + 0.014, legend_y, label, va="center", ha="left",
            fontsize=5, color=TEXT_DARK, fontfamily="serif")

# ── Export ──────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = f"docs/figures/event_driven_scheduling.{ext}"
    fig.savefig(path, bbox_inches="tight", facecolor=BG, dpi=300, pad_inches=0.08)
    print(f"Saved → {path}")
plt.close(fig)
