"""2x2 concept figure: evidence compatibility vs behavioural evidence dependence.

Figure 1 of the ACL/ARR submission. Single-column width (~3.0in), matplotlib only.

Quadrants (rows = evidence compatible/incompatible; cols = dependent/independent):
    TL: Compatible + Dependent       -> TRUE grounded (ACCEPT)
    TR: Compatible + Independent     -> post-rationalization (the dangerous case
                                         CF-Verify catches; the case LLM-judges
                                         miss)
    BL: Incompatible + Dependent     -> answer follows evidence but evidence
                                         changed (REVISE / RE-RETRIEVE)
    BR: Incompatible + Independent   -> answer ignores evidence; collapse point:
                                         true->changed, false->unchanged
                                         (ABSTAIN)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Single-column ACL width on A4 with 2.5cm margins and 0.6cm columnsep is ~7.7cm
# (~3.03in). Use 3.0in and let bbox_inches='tight' crop.
FIG_W_IN, FIG_H_IN = 3.0, 3.2
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# ---- Layout grid (units of figure-percent) -------------------------------
# Top header band: y in [92, 100]
# Row labels band: x in [0, 14]
# Column headers band: y in [83, 92]
# Quadrant grid:    x in [14, 100], y in [10, 83]
# Bottom caption:   y in [0, 8]

# Quadrant boxes
GX0, GX1 = 15, 100     # left/right extents of grid
GY0, GY1 = 9, 81       # bottom/top extents of grid
MIDX = (GX0 + GX1) / 2
MIDY = (GY0 + GY1) / 2

boxes = [
    # (x, y, w, h, facecolor, edgecolor)
    (GX0, MIDY, MIDX - GX0, GY1 - MIDY, "#e8f2e8", "#3a7d3a"),   # TL
    (MIDX, MIDY, GX1 - MIDX, GY1 - MIDY, "#fdeaea", "#b03a3a"),   # TR
    (GX0, GY0,  MIDX - GX0, MIDY - GY0, "#fdf6e3", "#b8860b"),   # BL
    (MIDX, GY0, GX1 - MIDX, MIDY - GY0, "#eef0f4", "#555f6e"),   # BR
]
for x, y, w, h, fc, ec in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                facecolor=fc, edgecolor=ec, linewidth=0.9))


def add_text(x, y, text, *, ha="center", va="center", fontsize=6.5,
             color="#222222", fontweight="normal", rotation=0,
             linespacing=1.25):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            fontweight=fontweight, rotation=rotation, linespacing=linespacing)


def quadrant(cx, cy, header, label_color, body_lines, footer, footer_color):
    """Place content inside one quadrant, anchored at quadrant centre."""
    # Header label (short, fits within box)
    add_text(cx, cy + 15, header, fontsize=6.4, color=label_color,
             fontweight="bold")
    # Body text — wrapped
    add_text(cx, cy - 0.5, "\n".join(body_lines), fontsize=5.5,
             color="#222222", linespacing=1.25)
    # Footer action
    add_text(cx, cy - 16.5, footer, fontsize=6.0, color=footer_color,
             fontweight="bold")


# ---- Top header ---------------------------------------------------------
add_text(50, 95, "CF-Verify's organising distinction",
         fontsize=7.5, color="#1f2a44", fontweight="bold")

# ---- Column headers (above the grid) ------------------------------------
add_text((GX0 + MIDX) / 2, 87, "Dependent", fontsize=6.5,
         color="#1f2a44", fontweight="bold")
add_text((MIDX + GX1) / 2, 87, "Independent", fontsize=6.5,
         color="#1f2a44", fontweight="bold")

# ---- Row labels (left of grid) ------------------------------------------
add_text(GX0 - 7.5, (MIDY + GY1) / 2, "Evidence\ncompatible",
         ha="center", va="center", fontsize=6.5, color="#1f2a44",
         fontweight="bold", rotation=90, linespacing=1.2)
add_text(GX0 - 7.5, (GY0 + MIDY) / 2, "Evidence\nincompatible",
         ha="center", va="center", fontsize=6.5, color="#1f2a44",
         fontweight="bold", rotation=90, linespacing=1.2)

# ---- Quadrant contents --------------------------------------------------
# Short headers (one line), body (short wrapped lines), action footer.

quadrant((GX0 + MIDX) / 2, (MIDY + GY1) / 2,
         "TRUE grounded", "#2f6b2f",
         ["Compatible + Dependent",
          "Q: capital of Germany?",
          "E: Berlin -> ans. Berlin",
          "delete E -> ans. flips"],
         "ACCEPT", "#2f6b2f")

quadrant((MIDX + GX1) / 2, (MIDY + GY1) / 2,
         "post-rationalization", "#a03030",
         ["Compatible + Independent",
          "Q: capital of Germany?",
          "E: Berlin -> ans. Berlin",
          "delete E -> ans. Berlin",
          "(parametric answer)"],
         "flag this", "#a03030")

quadrant((GX0 + MIDX) / 2, (GY0 + MIDY) / 2,
         "follows evidence", "#8a6508",
         ["Incompatible + Dependent",
          "rewrite breaks the fact;",
          "ans. changed because",
          "evidence changed",
          "delete E -> ans. flips"],
         "REVISE / RE-RETRIEVE", "#8a6508")

quadrant((MIDX + GX1) / 2, (GY0 + MIDY) / 2,
         "ignores evidence", "#3f4756",
         ["Incompatible + Independent",
          "ans. disagrees with E",
          "delete E -> ans. unchanged",
          "collapse point:",
          " true->changed;",
          " false->unchanged"],
         "ABSTAIN", "#3f4756")

# ---- Tiny caption / route legend at bottom -----------------------------
add_text(50, 4,
         "rows: compatibility  |  cols: dependence",
         fontsize=5.6, color="#666666")

# Save with tight bbox so no white-space eats the column.
out_dir = Path("/root/gcy/cf/paper_acl/figures")
out_dir.mkdir(parents=True, exist_ok=True)
pdf_path = out_dir / "concept_2x2.pdf"
png_path = out_dir / "concept_2x2.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
print(f"Wrote {pdf_path}")
print(f"Wrote {png_path}")
