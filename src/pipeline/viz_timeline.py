"""
Make a timeline figure showing all 9 years' land use side by side.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA, FIG, YEARS, LU_NAMES, LU_COLORS

# Build colormap (0-6 + 7 for outside)
COLORS = [LU_COLORS[i] for i in range(7)] + ['#ffffff']
cmap = ListedColormap(COLORS)
norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), cmap.N)

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes_flat = axes.flatten()

for ax, y in zip(axes_flat, YEARS):
    a = np.load(DATA / f'lu_clean_{y}.npy').astype(int)
    a[a == 255] = 7   # 255 → 7 (white)
    ax.imshow(a, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title(f'{y}', fontsize=18, fontweight='bold')
    ax.axis('off')

# Legend
legend = [Patch(facecolor=LU_COLORS[i], label=f'{i}: {LU_NAMES[i]}') for i in range(7)]
fig.legend(handles=legend, loc='lower center', ncol=7, fontsize=11,
           bbox_to_anchor=(0.5, 0.01), frameon=False)

plt.suptitle(f'北京 9 年用地时序 ({YEARS[0]}–{YEARS[-1]}, +2020)',
             fontsize=20, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
out = FIG / 'timeline_9years.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close()
print(f'→ {out}')
