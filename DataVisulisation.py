import numpy as np
import matplotlib.pyplot as plt

# ==============================
# HARD-CODED PARAMETERS
# ==============================
WINDOW_SIZE = 168
T_START = 0
T_END = 350

# Sample 10
S10_START = 9
S10_END = S10_START + WINDOW_SIZE - 1  # 176

# Sample 110
S110_START = 109
S110_END = S110_START + WINDOW_SIZE - 1  # 276

# ==============================
# GENERATE STRICTLY NON-NEGATIVE TIME SERIES
# ==============================
np.random.seed(42)
t = np.arange(T_START, T_END + 1)

y = (
    0.5 * np.abs(np.sin(0.04 * t)) +
    0.3 * np.abs(np.sin(0.1 * t)) +
    0.02 * t / T_END +
    0.05 * np.abs(np.random.randn(len(t)))
)

# ==============================
# CREATE FIGURE
# ==============================
plt.figure(figsize=(12, 5))

# Plot time series
plt.plot(t, y, color="#1f77b4", linewidth=2.5)

# -----------------------------
# Draw vertical boundaries
# -----------------------------
def draw_boundary(x_position, color, label=None):
    plt.plot([x_position, x_position], [0, y[x_position]], 
             linestyle="--", linewidth=2, color=color, label=label)

# Sample 10 (RED)
draw_boundary(S10_START, "red", "Sample 10")
draw_boundary(S10_END, "red")

# Sample 110 (GREEN)
draw_boundary(S110_START, "green", "Sample 110")
draw_boundary(S110_END, "green")

# -----------------------------
# Fill the windows
# -----------------------------
# Fill Sample 10
plt.fill_between(
    t[S10_START:S10_END+1], 
    0, 
    y[S10_START:S10_END+1], 
    color="red", 
    alpha=0.2
)

# Fill Sample 110
plt.fill_between(
    t[S110_START:S110_END+1], 
    0, 
    y[S110_START:S110_END+1], 
    color="green", 
    alpha=0.2
)

# Overlap region
overlap_start = max(S10_START, S110_START)
overlap_end = min(S10_END, S110_END)

if overlap_start < overlap_end:
    plt.fill_between(
        t[overlap_start:overlap_end+1], 
        0, 
        y[overlap_start:overlap_end+1], 
        color="purple", 
        alpha=0.35
    )

# ==============================
# AXIS FORMATTING
# ==============================
plt.xlim(T_START, T_END)
plt.ylim(0, max(y) * 1.1)

plt.xticks([T_START, T_END], [r"$t_0$", r"$t_n$"])
plt.xlabel("Time")
plt.ylabel("Value")

plt.legend(loc="upper left", frameon=False)
plt.grid(alpha=0.25)

plt.tight_layout()

# ==============================
# SAVE FIGURE
# ==============================
plt.savefig("sliding_window_S10_S110_overlap.png", dpi=300)
plt.savefig("sliding_window_S10_S110_overlap.pdf")

plt.show()
