"""Generate event_driven_scheduling.png with two failure modes:
1. Physics drift (Agent observes s₁, action lands on s₄)
2. Disturbance collision (fault arrives mid-pipeline, invalidates pending action)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 9.0))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(-0.5, 9.0)
ax.axis('off')
fig.patch.set_facecolor('#EDF2F7')
ax.set_facecolor('#EDF2F7')

# Title
ax.text(7, 8.65, 'Heterogeneous event-driven execution at deployment',
        ha='center', va='center', fontsize=18, fontweight='bold',
        color='#C53030',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FED7D7', edgecolor='#FC8181', linewidth=1.5))

# ─── ENV TIMELINE ───
env_y = 7.3
ax.annotate('', xy=(14, env_y), xytext=(0.5, env_y),
            arrowprops=dict(arrowstyle='->', color='#63B3ED', lw=2.5))
ax.text(0.1, env_y, 'Env', fontsize=16, fontweight='bold', color='#E53E3E', va='center')
ax.text(14.3, env_y, 't', fontsize=16, fontstyle='italic', color='#63B3ED', va='center')

# State labels
states = ['s₀', 's₁', 's₂', 's₃', 's₄', 's₅', 's₆', 's₇', 's₈']
state_x = [1.0, 2.2, 3.8, 5.2, 6.5, 7.8, 9.2, 10.5, 12.0]
for i, (sx, label) in enumerate(zip(state_x, states)):
    ax.plot([sx, sx], [env_y - 0.15, env_y + 0.15], color='#63B3ED', lw=1.5)
    ax.text(sx, env_y + 0.35, label, ha='center', fontsize=10, color='#63B3ED')

# ─── FAILURE MODE 1: PHYSICS DRIFT ───
agent1_y = 4.8
ax.text(0.1, agent1_y, 'Agent 1', fontsize=15, fontweight='bold', color='#1A202C', va='center')

# Dormant line
ax.plot([1.5, 2.2], [agent1_y, agent1_y], '--', color='#A0AEC0', lw=1.5)
ax.text(1.7, agent1_y - 0.3, 'dormant', fontsize=8, fontstyle='italic', color='#A0AEC0')

# OBS event (circle)
obs1_x = 2.5
ax.add_patch(plt.Circle((obs1_x, agent1_y), 0.28, color='#3182CE', zorder=5))
ax.text(obs1_x, agent1_y, 'obs', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# "sees s₁" annotation
ax.annotate('', xy=(2.2, env_y - 0.3), xytext=(obs1_x, agent1_y + 0.35),
            arrowprops=dict(arrowstyle='->', color='#3182CE', lw=1, linestyle='dotted'))
ax.text(1.8, 5.8, 'sees s₁', fontsize=9, fontstyle='italic', color='#38A169')

# Compute delay arrow
ax.annotate('', xy=(4.5, agent1_y), xytext=(3.0, agent1_y),
            arrowprops=dict(arrowstyle='<->', color='#718096', lw=1.2))
ax.text(3.75, agent1_y + 0.25, 'compute', fontsize=8, fontstyle='italic', color='#718096', ha='center')

# Effect delay arrow
ax.annotate('', xy=(6.3, agent1_y), xytext=(4.7, agent1_y),
            arrowprops=dict(arrowstyle='<->', color='#718096', lw=1.2))
ax.text(5.5, agent1_y + 0.25, 'effect delay', fontsize=8, fontstyle='italic', color='#718096', ha='center')

# EFF event (diamond)
eff1_x = 6.5
diamond = mpatches.RegularPolygon((eff1_x, agent1_y), numVertices=4, radius=0.32,
                                   orientation=0, color='#319795', zorder=5)
ax.add_patch(diamond)
ax.text(eff1_x, agent1_y, 'eff', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# "hits s₄" annotation
ax.annotate('', xy=(6.5, env_y - 0.3), xytext=(eff1_x, agent1_y + 0.35),
            arrowprops=dict(arrowstyle='->', color='#319795', lw=1, linestyle='dotted'))
ax.text(6.9, 5.8, 'hits s₄', fontsize=9, fontstyle='italic', color='#38A169')

# Physics drift annotation box
ax.text(4.5, agent1_y - 0.6, 'physics drifted s₁→s₄ between obs and effect',
        ha='center', fontsize=9, fontstyle='italic', color='#744210',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEFCBF', edgecolor='#ECC94B', linewidth=1))

# Section label
ax.text(0.1, agent1_y + 0.9, '① Physics drift', fontsize=11, fontweight='bold', color='#2D3748')

# ─── FAILURE MODE 2: DISTURBANCE COLLISION ───
agent2_y = 2.0
ax.text(0.1, agent2_y, 'Agent 2', fontsize=15, fontweight='bold', color='#1A202C', va='center')

# Section label
ax.text(0.1, agent2_y + 0.9, '② Disturbance collision', fontsize=11, fontweight='bold', color='#2D3748')

# Dormant line
ax.plot([1.5, 7.5], [agent2_y, agent2_y], '--', color='#A0AEC0', lw=1.5)

# Timer label
ax.text(7.8, agent2_y + 0.55, 'timer', fontsize=9, fontweight='bold', color='#DD6B20')
ax.annotate('', xy=(8.0, agent2_y + 0.35), xytext=(8.0, agent2_y + 0.1),
            arrowprops=dict(arrowstyle='->', color='#DD6B20', lw=1.2))

# OBS event
obs2_x = 8.0
ax.add_patch(plt.Circle((obs2_x, agent2_y), 0.28, color='#DD6B20', zorder=5))
ax.text(obs2_x, agent2_y, 'obs', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# "sees s₅" annotation
ax.annotate('', xy=(7.8, env_y - 0.3), xytext=(obs2_x, agent2_y + 0.35),
            arrowprops=dict(arrowstyle='->', color='#DD6B20', lw=1, linestyle='dotted'))
ax.text(7.3, agent2_y + 1.4, 'sees s₅', fontsize=9, fontstyle='italic', color='#38A169')

# Compute delay arrow
ax.annotate('', xy=(9.5, agent2_y), xytext=(8.5, agent2_y),
            arrowprops=dict(arrowstyle='<->', color='#718096', lw=1.2))
ax.text(9.0, agent2_y + 0.25, 'compute', fontsize=8, fontstyle='italic', color='#718096', ha='center')

# ─── FAULT EVENT (lightning bolt on env timeline) ───
fault_x = 10.0
# Red zigzag / lightning bolt
bolt_x = [fault_x - 0.12, fault_x + 0.12, fault_x - 0.08, fault_x + 0.15, fault_x]
bolt_y = [env_y + 0.5, env_y + 0.25, env_y + 0.05, env_y - 0.2, env_y - 0.45]
ax.plot(bolt_x, bolt_y, color='#E53E3E', lw=2.5, zorder=6)
ax.text(fault_x + 0.4, env_y + 0.5, 'FAULT', fontsize=9, fontweight='bold',
        color='#E53E3E', ha='left',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#FED7D7', edgecolor='#FC8181', linewidth=1))

# Fault annotation: state changes discontinuously
ax.annotate('', xy=(10.5, env_y - 0.15), xytext=(9.5, env_y - 0.15),
            arrowprops=dict(arrowstyle='->', color='#E53E3E', lw=2))
ax.text(10.0, env_y - 0.55, 's₆ → s₆\'', fontsize=10, fontweight='bold',
        color='#E53E3E', ha='center')
ax.text(10.0, env_y - 0.85, '(topology changed)', fontsize=8, fontstyle='italic',
        color='#E53E3E', ha='center')

# Vertical dashed line from fault to agent timeline
ax.plot([fault_x, fault_x], [env_y - 1.0, agent2_y + 0.5], '--', color='#E53E3E', lw=1.5, alpha=0.6)
ax.text(10.35, 3.5, 'fault arrives\nmid-pipeline', fontsize=8, fontstyle='italic',
        color='#E53E3E', ha='left')

# Effect delay arrow (continues despite fault)
ax.annotate('', xy=(11.0, agent2_y), xytext=(9.7, agent2_y),
            arrowprops=dict(arrowstyle='<->', color='#718096', lw=1.2))
ax.text(10.35, agent2_y + 0.25, 'effect delay', fontsize=8, fontstyle='italic', color='#718096', ha='center')

# EFF event (diamond) — lands on post-fault state
eff2_x = 11.2
diamond2 = mpatches.RegularPolygon((eff2_x, agent2_y), numVertices=4, radius=0.32,
                                    orientation=0, color='#E53E3E', zorder=5)
ax.add_patch(diamond2)
ax.text(eff2_x, agent2_y, 'eff', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# "hits s₆' (post-fault)" annotation
ax.annotate('', xy=(11.2, env_y - 1.0), xytext=(eff2_x, agent2_y + 0.35),
            arrowprops=dict(arrowstyle='->', color='#E53E3E', lw=1, linestyle='dotted'))

# Disturbance collision annotation box
ax.text(10.0, agent2_y - 0.65, 'action computed for pre-fault s₅\nlanded on post-fault s₆\' — may be harmful',
        ha='center', fontsize=9, fontstyle='italic', color='#744210',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FED7D7', edgecolor='#FC8181', linewidth=1))

# ─── LEGEND ───
legend_y = 0.3

# Observe / decide
ax.add_patch(plt.Circle((1.7, legend_y), 0.15, color='#3182CE', zorder=5))
ax.text(2.05, legend_y, 'observe / decide', fontsize=9, va='center', color='#4A5568')

# Action takes effect
ax.add_patch(mpatches.RegularPolygon((5.0, legend_y), numVertices=4, radius=0.18,
                                      color='#319795', zorder=5))
ax.text(5.35, legend_y, 'action takes effect', fontsize=9, va='center', color='#4A5568')

# Fault / disturbance (zigzag)
zx, zy = 8.3, legend_y
ax.plot([zx-0.1, zx+0.05, zx-0.05, zx+0.1],
        [zy+0.15, zy+0.05, zy-0.05, zy-0.15],
        color='#E53E3E', lw=2.5, zorder=6)
ax.text(8.6, legend_y, 'fault / disturbance', fontsize=9, va='center', color='#4A5568')

# Physics update
ax.plot([11.3, 11.7], [legend_y, legend_y], color='#63B3ED', lw=2)
ax.plot([11.5, 11.5], [legend_y - 0.12, legend_y + 0.12], color='#63B3ED', lw=1.5)
ax.text(11.9, legend_y, 'physics update', fontsize=9, va='center', color='#4A5568')

plt.tight_layout()
plt.savefig('/Users/criss_w/Desktop/Research_and_ML/MARL/PowerGym/docs/images/event_driven_scheduling.png',
            dpi=150, bbox_inches='tight', facecolor='#EDF2F7')
plt.close()
print("Figure saved.")
