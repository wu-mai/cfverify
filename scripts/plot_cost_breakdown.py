"""
plot_cost_breakdown.py — Render a cost breakdown bar chart for CF-Verify.

Outputs:
    figures/cost_breakdown.png
    figures/cost_breakdown.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(exist_ok=True)

# Data from the cost analysis in §3.7 (HotpotQA baseline; 2Wiki is similar shape)
datasets = ["HotpotQA", "2WikiMultihopQA"]
single_answer_cost = [0.0037, 0.0021]   # one answer call (input tokens × $2.50/M)
k1_cost = [0.016, 0.010]                 # full CF-Verify at K=1
k5_cost = [0.030, 0.018]                 # full CF-Verify at K=5
merged_k1 = [0.012, 0.007]               # merged-prompt variant at K=1

x = np.arange(len(datasets))
width = 0.18

fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
b1 = ax.bar(x - 1.5*width, single_answer_cost, width, label="Single answer call (baseline)", color="#cccccc")
b2 = ax.bar(x - 0.5*width, k1_cost, width, label="CF-Verify, K=1 (separate calls)", color="#4a90d9")
b3 = ax.bar(x + 0.5*width, k5_cost, width, label="CF-Verify, K=5 (separate calls)", color="#1f4e79")
b4 = ax.bar(x + 1.5*width, merged_k1, width, label="CF-Verify, K=1 (merged prompt)", color="#a4c8e1")

ax.set_ylabel("USD per question", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 0.035)

# Label bars
for bars in (b1, b2, b3, b4):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0005,
                f"${h:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_title("Per-question USD cost of CF-Verify vs. a single answer call\n"
             "(GPT-5.4 list price: input \\$2.50/M, output \\$15.00/M)",
             fontsize=11)

plt.tight_layout()
out_png = FIG / "cost_breakdown.png"
out_pdf = FIG / "cost_breakdown.pdf"
plt.savefig(out_png, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")