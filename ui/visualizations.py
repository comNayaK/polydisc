# =============================================================
# ui/visualizations.py — All chart generation
# =============================================================

import io, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image as PILImage
from rdkit import Chem
from rdkit.Chem import Draw

from scoring.green_score import PROPERTY_COLS
from config import tmp_path

DARK_BG   = '#0e1419'
PANEL_BG  = '#141c24'
ACCENT    = '#00b894'
ACCENT2   = '#0984e3'
TEXT_C    = '#e8f0f8'
SUBTEXT   = '#a0b0c0'
GRID_C    = '#2d3748'
COLORS7   = ['#00b894','#0984e3','#e17055','#6c5ce7','#fdcb6e','#fd79a8','#55efc4']


def _buf_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig); buf.seek(0)
    return PILImage.open(buf).copy()


def _dark_ax(ax, fig=None):
    if fig: fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=SUBTEXT)
    ax.spines[:].set_color(GRID_C)
    return ax


# ── Radar chart ───────────────────────────────────────────────
def make_radar(top, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    cats   = PROPERTY_COLS
    n      = len(cats)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    for ring in [0.25,0.5,0.75,1.0]:
        ax.plot(angles, [ring]*n+[ring], color=(0.5,0.5,0.5,0.15), lw=0.5, linestyle='--')

    for i, (_,row) in enumerate(top.head(3).iterrows()):
        vals = [float(row[c]) for c in cats] + [float(row[cats[0]])]
        ax.plot(angles, vals, 'o-', lw=2.5, color=COLORS7[i],
                label=f'#{i+1} GS={row["Green_score"]:.3f}', alpha=0.95)
        ax.fill(angles, vals, alpha=0.10, color=COLORS7[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, size=9, color=text)
    ax.set_ylim(0,1); ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(['0.25','0.5','0.75','1.0'], size=7, color=SUBTEXT)
    ax.spines['polar'].set_color(GRID_C); ax.grid(color=GRID_C, lw=0.8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4,1.15),
              fontsize=8, framealpha=0, labelcolor=text)
    ax.set_title('Property Radar — Top 3', color=text, fontsize=12, pad=22, fontweight='bold')
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── Horizontal bar chart ──────────────────────────────────────
def make_bar(top, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    fig, ax = plt.subplots(figsize=(9,4))
    fig.patch.set_facecolor(bg); ax.set_facecolor(PANEL_BG if theme=='dark' else '#ffffff')

    labels = [f'#{i+1}  {s[:22]}{"…" if len(s)>22 else ""}' for i,s in enumerate(top['SMILES'])]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))
    bars   = ax.barh(labels[::-1], top['Green_score'].values[::-1],
                     color=colors[::-1], edgecolor=GRID_C, linewidth=0.8, height=0.55)
    for bar, v in zip(bars, top['Green_score'].values[::-1]):
        ax.text(v+0.01, bar.get_y()+bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9, color=text, fontfamily='monospace')

    ax.set_xlim(0,1.15); ax.set_xlabel('Green Score', color=SUBTEXT, fontsize=10)
    ax.set_title('Top 5 — Green Score Ranking', color=text, fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(colors=SUBTEXT); ax.spines[['top','right','bottom']].set_visible(False)
    ax.spines['left'].set_color(GRID_C)
    for lbl in ax.get_yticklabels(): lbl.set_color(SUBTEXT); lbl.set_fontsize(9)
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── Property heatmap ──────────────────────────────────────────
def make_heatmap(top, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    cats     = PROPERTY_COLS
    data     = top[cats].values
    labels_y = [f'#{i+1} {s[:16]}{"…" if len(s)>16 else ""}' for i,s in enumerate(top['SMILES'])]

    fig, ax = plt.subplots(figsize=(10,4))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, color=SUBTEXT, fontsize=10, rotation=15)
    ax.set_yticks(range(len(top)));  ax.set_yticklabels(labels_y, color=SUBTEXT, fontsize=9)
    for i in range(len(top)):
        for j,c in enumerate(cats):
            v = data[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='#111' if 0.25<v<0.8 else 'white',
                    fontfamily='monospace', fontweight='bold')
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cb.ax.tick_params(colors=SUBTEXT, labelsize=8); cb.set_label('Score', color=SUBTEXT, fontsize=9)
    ax.set_title('Property Heatmap — Top 5', color=text, fontsize=12, fontweight='bold', pad=12)
    ax.spines[:].set_color(GRID_C); ax.tick_params(colors=SUBTEXT)
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── Score breakdown stacked bar ───────────────────────────────
def make_breakdown(top, weights, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    cats   = PROPERTY_COLS
    labels = [f'#{i+1}' for i in range(len(top))]
    x      = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(9,4))
    fig.patch.set_facecolor(bg); ax.set_facecolor(PANEL_BG if theme=='dark' else '#fff')

    weight_keys = ['biodeg','synth','tox','recycl','strength','tg','solubility']
    total_w = sum(weights.get(k,1) for k in weight_keys)
    bottom  = np.zeros(len(top))
    for cat, col, wk in zip(cats, COLORS7, weight_keys):
        w    = weights.get(wk, 1/7) / total_w
        vals = top[cat].values * w
        ax.bar(x, vals, 0.6, bottom=bottom, label=f'{cat} (w={w:.2f})',
               color=col, alpha=0.85, edgecolor=DARK_BG, lw=0.5)
        bottom += vals

    ax2 = ax.twinx()
    ax2.plot(x, top['Green_score'].values, 'o--', color='white', lw=2.5, ms=8,
             label='Green Score', zorder=5)
    ax2.set_ylim(0,1.25); ax2.set_ylabel('Green Score', color=text, fontsize=9)
    ax2.tick_params(colors=SUBTEXT); ax2.spines[:].set_color(GRID_C)

    ax.set_xticks(x); ax.set_xticklabels(labels, color=SUBTEXT, fontsize=11)
    ax.set_ylabel('Weighted Score', color=SUBTEXT, fontsize=9)
    ax.set_ylim(0,1.25); ax.tick_params(colors=SUBTEXT); ax.spines[:].set_color(GRID_C)
    ax.set_title('Green Score Component Breakdown', color=text, fontsize=12, fontweight='bold')

    l1,lb1 = ax.get_legend_handles_labels()
    l2,lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, loc='upper right', fontsize=7,
              framealpha=0.25, labelcolor=text, facecolor=PANEL_BG, ncol=2)
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── Distribution + scatter ────────────────────────────────────
def make_distribution(df, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    fig, axes = plt.subplots(1,2, figsize=(12,4))
    fig.patch.set_facecolor(bg)

    # Histogram
    ax = axes[0]; ax.set_facecolor(PANEL_BG if theme=='dark' else '#fff')
    n, bins, patches = ax.hist(df['Green_score'], bins=35, edgecolor=DARK_BG, alpha=0.9, lw=0.4)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(plt.cm.RdYlGn(left))
    ax.axvline(df['Green_score'].mean(), color='white', lw=2, linestyle='--',
               label=f'Mean={df["Green_score"].mean():.3f}')
    ax.axvline(df['Green_score'].max(),  color='#fdcb6e', lw=2, linestyle='--',
               label=f'Best={df["Green_score"].max():.3f}')
    ax.set_xlabel('Green Score', color=SUBTEXT, fontsize=10)
    ax.set_ylabel('Count', color=SUBTEXT, fontsize=10)
    ax.set_title('Score Distribution', color=text, fontsize=11, fontweight='bold')
    ax.tick_params(colors=SUBTEXT); ax.spines[:].set_color(GRID_C)
    ax.legend(fontsize=8, framealpha=0.2, labelcolor=text, facecolor=PANEL_BG)

    # Scatter: Biodeg vs Tox
    ax2 = axes[1]; ax2.set_facecolor(PANEL_BG if theme=='dark' else '#fff')
    sc  = ax2.scatter(df['Biodeg'], df['Tox'], c=df['Green_score'],
                      cmap='RdYlGn', s=18, alpha=0.65, vmin=0, vmax=1)
    top5 = df.head(5)
    ax2.scatter(top5['Biodeg'], top5['Tox'], c='white', s=100, marker='*', zorder=6, label='Top 5')
    cb  = plt.colorbar(sc, ax=ax2, fraction=0.035, pad=0.03)
    cb.ax.tick_params(colors=SUBTEXT, labelsize=7); cb.set_label('Green Score', color=SUBTEXT, fontsize=8)
    ax2.set_xlabel('Biodegradability', color=SUBTEXT, fontsize=10)
    ax2.set_ylabel('Low Toxicity',     color=SUBTEXT, fontsize=10)
    ax2.set_title('Biodeg vs Toxicity Space', color=text, fontsize=11, fontweight='bold')
    ax2.tick_params(colors=SUBTEXT); ax2.spines[:].set_color(GRID_C)
    ax2.legend(fontsize=8, framealpha=0.2, labelcolor=text, facecolor=PANEL_BG)

    plt.tight_layout()
    return _buf_to_pil(fig)


# ── 2D chemical structures ────────────────────────────────────
def make_structs(top):
    try:
        mols = [Chem.MolFromSmiles(s) for s in top['SMILES']]
        mols = [m for m in mols if m is not None]
        grid = Draw.MolsToGridImage(
            mols, molsPerRow=3, subImgSize=(340,280),
            legends=[f'#{i+1}  GS={top.iloc[i]["Green_score"]:.4f}' for i in range(len(mols))],
            returnPNG=False)
        buf = io.BytesIO(); grid.save(buf,'PNG'); buf.seek(0)
        return PILImage.open(buf).copy()
    except: return None


# ── Pairwise property correlation ────────────────────────────
def make_correlation(df, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    corr = df[PROPERTY_COLS+['Green_score']].corr()
    fig, ax = plt.subplots(figsize=(9,7))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    import seaborn as sns
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, False)
    sns.heatmap(corr, ax=ax, cmap='coolwarm', vmin=-1, vmax=1,
                annot=True, fmt='.2f', linewidths=0.5,
                linecolor=GRID_C, cbar_kws={'shrink':0.8},
                annot_kws={'size':8, 'color':text})
    ax.set_title('Property Correlation Matrix', color=text, fontsize=12, fontweight='bold', pad=14)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    for spine in ax.spines.values(): spine.set_color(GRID_C)
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── Property violin distribution ─────────────────────────────
def make_violin(df, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    fig, axes = plt.subplots(1, len(PROPERTY_COLS), figsize=(14,4), sharey=True)
    fig.patch.set_facecolor(bg)

    for ax, col, color in zip(axes, PROPERTY_COLS, COLORS7):
        ax.set_facecolor(PANEL_BG if theme=='dark' else '#fff')
        data = df[col].dropna().values
        vp   = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in vp['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.75)
        vp['cmedians'].set_color('white'); vp['cmedians'].set_lw(2)
        vp['cbars'].set_color(SUBTEXT);    vp['cmaxes'].set_color(SUBTEXT)
        vp['cmins'].set_color(SUBTEXT)
        ax.set_title(col, color=text, fontsize=9, fontweight='bold')
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        ax.spines[:].set_color(GRID_C)

    fig.suptitle('Property Distribution across Candidates', color=text, fontsize=12, fontweight='bold')
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── Comparison chart for 2 molecules ─────────────────────────
def make_comparison(row_a, row_b, theme='dark'):
    bg   = DARK_BG if theme=='dark' else '#f8f9fa'
    text = TEXT_C  if theme=='dark' else '#2d3436'

    cats = PROPERTY_COLS
    x    = np.arange(len(cats)); width = 0.35

    fig, ax = plt.subplots(figsize=(11,4))
    fig.patch.set_facecolor(bg); ax.set_facecolor(PANEL_BG if theme=='dark' else '#fff')

    vals_a = [float(row_a.get(c,0.5)) for c in cats]
    vals_b = [float(row_b.get(c,0.5)) for c in cats]

    ax.bar(x-width/2, vals_a, width, label=f'A  GS={row_a.get("Green_score",0):.4f}',
           color=ACCENT, alpha=0.85, edgecolor=DARK_BG)
    ax.bar(x+width/2, vals_b, width, label=f'B  GS={row_b.get("Green_score",0):.4f}',
           color=ACCENT2, alpha=0.85, edgecolor=DARK_BG)

    ax.set_xticks(x); ax.set_xticklabels(cats, color=SUBTEXT, fontsize=10)
    ax.set_ylim(0,1.15); ax.set_ylabel('Score', color=SUBTEXT, fontsize=10)
    ax.set_title('Side-by-Side Molecule Comparison', color=text, fontsize=12, fontweight='bold')
    ax.tick_params(colors=SUBTEXT); ax.spines[:].set_color(GRID_C)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor=text, facecolor=PANEL_BG)
    ax.grid(axis='y', color=GRID_C, lw=0.5, alpha=0.5)
    plt.tight_layout()
    return _buf_to_pil(fig)


# ── UMAP chemical space (optional) ───────────────────────────
def make_umap(df, theme='dark'):
    """Project candidates into 2D using UMAP on Morgan fingerprints."""
    try:
        from umap import UMAP
        from features.featurizer import smiles_to_fp
        import numpy as np

        bg   = DARK_BG if theme=='dark' else '#f8f9fa'
        text = TEXT_C  if theme=='dark' else '#2d3436'

        fps  = np.vstack([smiles_to_fp(s) for s in df['SMILES']])
        emb  = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(fps)

        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_facecolor(bg); ax.set_facecolor(PANEL_BG if theme=='dark' else '#fff')

        sc  = ax.scatter(emb[:,0], emb[:,1], c=df['Green_score'],
                         cmap='RdYlGn', s=20, alpha=0.7, vmin=0, vmax=1)
        top5 = df.head(5)
        idx5 = top5.index.tolist()
        ax.scatter(emb[idx5,0], emb[idx5,1], c='white', s=120, marker='*', zorder=6, label='Top 5')
        cb  = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.03)
        cb.ax.tick_params(colors=SUBTEXT, labelsize=7); cb.set_label('Green Score', color=SUBTEXT)
        ax.set_title('UMAP Chemical Space', color=text, fontsize=12, fontweight='bold')
        ax.set_xlabel('UMAP-1', color=SUBTEXT); ax.set_ylabel('UMAP-2', color=SUBTEXT)
        ax.tick_params(colors=SUBTEXT); ax.spines[:].set_color(GRID_C)
        ax.legend(fontsize=8, framealpha=0.2, labelcolor=text, facecolor=PANEL_BG)
        plt.tight_layout()
        return _buf_to_pil(fig)
    except ImportError:
        return None
    except Exception as e:
        print(f'UMAP error: {e}')
        return None


def save_img(img, name):
    path = tmp_path(f'{name}.png')
    if img: img.save(path)
    return path
