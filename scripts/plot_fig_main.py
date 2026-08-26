"""
plot_fig_main.py — Re-render the CF-Verify pipeline figure as a clean
matplotlib figure. Avoid FancyArrowPatch (creates diamond artefacts).

Outputs:
    paper/figures/fig_main.png
    paper/figures/fig_main.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

FIG = Path("/root/gcy/cf/paper/figures")
FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4.5), dpi=160)
ax.set_xlim(0, 11.5)
ax.set_ylim(0, 5.0)
ax.axis("off")

BOX = dict(boxstyle="round,pad=0.45", linewidth=1.4, edgecolor="#1f4e79",
           facecolor="#deeaf6")
INPUT_BOX = dict(boxstyle="round,pad=0.4", linewidth=1.0, edgecolor="#888",
                 facecolor="#f5f5f5")
MERGED_BOX = dict(boxstyle="round,pad=0.4", linewidth=1.4, edgecolor="#c0504d",
                  facecolor="#fbe5e3")
DECISION_BOX = dict(boxstyle="round,pad=0.4", linewidth=1.4, edgecolor="#1f4e79",
                    facecolor="#fff2cc")


def box(x, y, w, h, text, style=BOX, fontsize=10, fontweight="normal"):
    p = FancyBboxPatch((x, y), w, h, **style)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


def arrow_simple(x1, y1, x2, y2, color="#1f4e79", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                shrinkA=0, shrinkB=0))


# Stage column headers
ax.text(0.9, 4.65, "Inputs", ha="center", fontsize=11, fontweight="bold", color="#666")
ax.text(3.5, 4.65, "(i) Self-rationale", ha="center", fontsize=11, fontweight="bold", color="#1f4e79")
ax.text(6.0, 4.65, "(ii) Targeted deletion", ha="center", fontsize=11, fontweight="bold", color="#1f4e79")
ax.text(8.5, 4.65, "(iii) Random deletion", ha="center", fontsize=11, fontweight="bold", color="#1f4e79")
ax.text(10.5, 4.65, "(iv) Routing", ha="center", fontsize=11, fontweight="bold", color="#c0504d")

# Title
ax.text(5.75, 4.95, "CF-Verify: Counterfactual Evidence Deletion Pipeline",
        ha="center", fontsize=13, fontweight="bold")

# Stage columns — three rows
# Row 1 (y=3.4): top-row boxes
# Row 2 (y=2.1): middle-row boxes
# Row 3 (y=0.8): bottom-row boxes

# Inputs column
box(0.1, 3.4, 1.6, 0.7, "Question Q", style=INPUT_BOX, fontsize=10)
box(0.1, 2.1, 1.6, 0.7, "Evidence E", style=INPUT_BOX, fontsize=10)

# (i) Self-rationale column
box(2.1, 3.4, 2.8, 0.7, "Answer call: a(0) ~ p(.|Q,E)", style=BOX, fontsize=9)
box(2.1, 2.1, 2.8, 0.7, "Self-rationale: predict Ĝ ⊆ E", style=BOX, fontsize=9)
box(2.1, 0.8, 2.8, 0.7, "Merged variant: a(0)+Ĝ in 1 call",
    style=MERGED_BOX, fontsize=9)

# (ii) Targeted column
box(4.7, 3.4, 2.6, 0.7, "E_ = E \\ Ĝ", style=BOX, fontsize=9)
box(4.7, 2.1, 2.6, 0.7, "a(1) ~ p(.|Q, E_)", style=BOX, fontsize=9)
box(4.7, 0.8, 2.6, 0.7, "F̂_T = 1[a(1) ≠ a(0)]", style=BOX, fontsize=9)

# (iii) Random column
box(7.4, 3.4, 2.2, 0.7, "R ~ Unif(E \\ Ĝ)", style=BOX, fontsize=9)
box(7.4, 2.1, 2.2, 0.7, "a(2) ~ p(.|Q, E \\ R)", style=BOX, fontsize=9)
box(7.4, 0.8, 2.2, 0.7, "F̂_R = 1[a(2) ≠ a(0)]", style=BOX, fontsize=9)

# (iv) Routing column
box(9.7, 3.4, 1.7, 0.7, "CFScore = F̂_T − F̂_R", style=BOX, fontsize=9)
box(9.7, 2.1, 1.7, 0.7, "Route on τ", style=DECISION_BOX, fontsize=9)
box(9.7, 0.8, 1.7, 0.7, "Accept / Revise / Re-Retrieve", style=DECISION_BOX, fontsize=8)

# Horizontal arrows between columns (right-pointing)
# Row 1
arrow_simple(1.7, 3.75, 2.1, 3.75)
arrow_simple(4.9, 3.75, 4.7, 3.75)  # forward
arrow_simple(7.3, 3.75, 7.4, 3.75)
arrow_simple(9.6, 3.75, 9.7, 3.75)
# Row 2
arrow_simple(1.7, 2.45, 2.1, 2.45)
arrow_simple(4.9, 2.45, 4.7, 2.45)
arrow_simple(7.3, 2.45, 7.4, 2.45)
arrow_simple(9.6, 2.45, 9.7, 2.45)

# Vertical arrows within each stage (down-pointing)
for x in [3.5, 6.0, 8.5]:
    arrow_simple(x, 3.4, x, 2.8)  # not used

# Cost annotation
ax.text(5.75, 0.20, r"Calls: 3+K (separate) or 2+K (merged).  "
        r"Cost: \$0.016/Q HotpotQA at K=1,  \$0.030/Q at K=5.",
        ha="center", fontsize=9, style="italic", color="#555")

plt.tight_layout()
out_png = FIG / "fig_main.png"
out_pdf = FIG / "fig_main.pdf"
plt.savefig(out_png, bbox_inches="tight", facecolor="white")
plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
print(f"Wrote {out_png}")
print(f"Wrote {out_pdf}")