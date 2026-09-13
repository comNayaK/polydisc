# =============================================================
# reports/pdf_report.py — Multi-page PDF report builder
# =============================================================

import datetime
import os
from fpdf import FPDF
from config import tmp_path
from scoring.green_score import PROPERTY_COLS


class PolymerReport(FPDF):
    ACCENT  = (0, 184, 148)
    DARK    = (14, 20, 25)
    WHITE   = (255, 255, 255)
    LIGHT   = (240, 255, 252)
    SUBTEXT = (100, 120, 140)

    def header(self):
        self.set_fill_color(*self.ACCENT)
        self.rect(0, 0, 210, 30, 'F')
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(*self.WHITE)
        self.set_y(7)
        self.cell(0, 9, 'Generative AI — Sustainable Polymer Discovery', ln=True, align='C')
        self.set_font('Helvetica', '', 8)
        self.cell(0, 5,
                  f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
                  f'XGBoost + Random Forest + GNN Ensemble  |  7-Property Green Score',
                  ln=True, align='C')
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-13)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(*self.SUBTEXT)
        self.cell(0, 5,
                  f'Page {self.page_no()}  |  Polymer Discovery AI  |  '
                  'XGB + RF + GNN Ensemble  |  Confidential', align='C')

    def section(self, title, color=None):
        c = color or self.ACCENT
        self.set_font('Helvetica', 'B', 11)
        self.set_fill_color(*c)
        self.set_text_color(*self.WHITE)
        self.cell(0, 9, f'  {title}', ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def subsection(self, title):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*self.ACCENT)
        self.cell(0, 6, title, ln=True)
        self.set_text_color(0, 0, 0)

    def kv_row(self, key, val, fill=False):
        self.set_fill_color(*self.LIGHT)
        self.set_font('Helvetica', 'B', 9)
        self.cell(80, 7, f'  {key}', border='B', fill=fill)
        self.set_font('Helvetica', '', 9)
        self.cell(0,  7, f'  {val}', border='B', fill=fill, ln=True)

    def _table_header(self, cols, widths):
        self.set_fill_color(*self.ACCENT)
        self.set_text_color(*self.WHITE)
        self.set_font('Helvetica', 'B', 8)
        for col, w in zip(cols, widths):
            self.cell(w, 8, col, border=1, align='C', fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)

    def _table_row(self, vals, widths, idx):
        self.set_fill_color(245, 255, 252) if idx%2==0 else self.set_fill_color(255,255,255)
        self.set_font('Helvetica', 'B' if idx==0 else '', 7)
        for v, w in zip(vals, widths):
            self.cell(w, 6, str(v), border=1, fill=True)
        self.ln()
        if self.get_y() > 265:
            self.add_page()


def build_pdf(top, all_df, weights, img_paths,
              total_generated, valid_novel, mode, temp, n_gen):
    """
    Build a comprehensive multi-page PDF report.
    Returns the PDF file path.
    """
    pdf = PolymerReport()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── PAGE 1: Executive Summary ─────────────────────────────
    pdf.add_page()
    pdf.section('1. Executive Summary')
    pdf.set_font('Helvetica','',9)

    summary = [
        ('Discovery mode',            mode),
        ('Candidates generated',      f'{total_generated:,}'),
        ('Valid novel structures',     f'{valid_novel:,}'),
        ('Validity rate',             f'{valid_novel/max(total_generated,1)*100:.1f}%'),
        ('VAE temperature',           f'{temp:.2f}'),
        ('Candidates requested',      f'{n_gen:,}'),
        ('Top Green Score',           f'{top["Green_score"].iloc[0]:.4f}'),
        ('Mean Green Score',          f'{all_df["Green_score"].mean():.4f}'),
        ('Std Green Score',           f'{all_df["Green_score"].std():.4f}'),
        ('Models used',               'XGBoost + Random Forest + GNN (ensemble)'),
        ('Properties predicted',      ', '.join(PROPERTY_COLS)),
    ]
    for i,(k,v) in enumerate(summary):
        pdf.kv_row(k, v, fill=(i%2==0))
    pdf.ln(6)

    # Weight table
    pdf.subsection('Green Score Weights:')
    wnames = ['Biodeg','Synth','Tox','Recycl','Strength','Tg','Solubility']
    wkeys  = ['biodeg','synth','tox','recycl','strength','tg','solubility']
    total_w = sum(weights.get(k,1) for k in wkeys)
    pdf._table_header(wnames, [27]*7)
    vals = [f'{weights.get(k,1)/total_w:.3f}' for k in wkeys]
    pdf._table_row(vals, [27]*7, 0)
    pdf.ln(6)

    # ── PAGE 1 cont: Top 5 table ──────────────────────────────
    pdf.section('2. Top 5 Polymer Candidates')
    cols   = ['#','SMILES','Biodeg','Synth','Tox','Recycl','Str','Tg','Sol','GS']
    widths = [8, 62, 14, 12, 12, 13, 12, 12, 12, 18]
    pdf._table_header(cols, widths)

    for idx,(_, row) in enumerate(top.iterrows()):
        s = row['SMILES'][:28]+'…' if len(row['SMILES'])>28 else row['SMILES']
        vals = [str(idx+1), s,
                f'{row["Biodeg"]:.3f}',       f'{row["Synth"]:.3f}',
                f'{row["Tox"]:.3f}',          f'{row["Recyclability"]:.3f}',
                f'{row["Strength"]:.3f}',     f'{row["Tg"]:.3f}',
                f'{row["Solubility"]:.3f}',   f'{row["Green_score"]:.4f}']
        pdf._table_row(vals, widths, idx)
    pdf.ln(4)

    pdf.subsection('Full SMILES Strings:')
    pdf.set_font('Courier','',7)
    for i,(_,row) in enumerate(top.iterrows()):
        pdf.set_x(12)
        pdf.multi_cell(0, 5, f'#{i+1}: {row["SMILES"]}')
    pdf.ln(3)

    # ── PAGE 2: Visualisations ────────────────────────────────
    pdf.add_page()
    pdf.section('3. Visualisations')

    viz_order = [
        ('struct',    '2D Chemical Structures (Top 5)'),
        ('radar',     'Property Radar Chart (Top 3)'),
        ('bar',       'Green Score Ranking (Top 5)'),
        ('heatmap',   'Property Heatmap (Top 5)'),
        ('breakdown', 'Green Score Component Breakdown'),
        ('dist',      'Score Distribution & Chemical Space'),
        ('violin',    'Property Distributions — All Candidates'),
        ('corr',      'Property Correlation Matrix'),
        ('umap',      'UMAP Chemical Space Projection'),
    ]
    for key, title in viz_order:
        path = img_paths.get(key)
        if path and os.path.exists(path):
            pdf.subsection(title)
            pdf.image(path, w=183)
            pdf.ln(4)
            if pdf.get_y() > 235:
                pdf.add_page()

    # ── PAGE 3: Full candidate table ─────────────────────────
    pdf.add_page()
    pdf.section('4. All Generated Candidates (sorted by Green Score)')
    cols2   = ['SMILES','Biodeg','Synth','Tox','Recycl','Str','Tg','Sol','GS']
    widths2 = [62,14,12,12,13,12,12,12,18]
    pdf._table_header(cols2, widths2)

    for idx,(_,row) in enumerate(all_df.head(50).iterrows()):
        s = row['SMILES'][:28]+'…' if len(row['SMILES'])>28 else row['SMILES']
        vals = [s, f'{row["Biodeg"]:.3f}', f'{row["Synth"]:.3f}',
                f'{row["Tox"]:.3f}',       f'{row["Recyclability"]:.3f}',
                f'{row["Strength"]:.3f}',  f'{row["Tg"]:.3f}',
                f'{row["Solubility"]:.3f}',f'{row["Green_score"]:.4f}']
        pdf._table_row(vals, widths2, idx)

    # ── PAGE 4: Statistical analysis ─────────────────────────
    pdf.add_page()
    pdf.section('5. Statistical Analysis')

    pdf.subsection('Property Statistics (all candidates):')
    stat_cols = PROPERTY_COLS + ['Green_score']
    pdf._table_header(['Property','Mean','Std','Min','25%','Median','75%','Max'],
                      [32,18,16,16,16,18,16,18])
    for col in stat_cols:
        s = all_df[col].describe()
        pdf._table_row([col,
                        f'{s["mean"]:.3f}', f'{s["std"]:.3f}',
                        f'{s["min"]:.3f}',  f'{s["25%"]:.3f}',
                        f'{s["50%"]:.3f}',  f'{s["75%"]:.3f}',
                        f'{s["max"]:.3f}'],
                       [32,18,16,16,16,18,16,18], stat_cols.index(col))
    pdf.ln(5)

    pdf.subsection('Top 10 by individual property:')
    for prop in PROPERTY_COLS:
        pdf.set_font('Helvetica','B',8)
        pdf.set_text_color(*PolymerReport.ACCENT)
        pdf.cell(0, 6, f'  Top by {prop}:', ln=True)
        pdf.set_text_color(0,0,0)
        top_prop = all_df.nlargest(3, prop)[['SMILES', prop, 'Green_score']]
        for _,r in top_prop.iterrows():
            pdf.set_font('Courier','',7)
            s = r['SMILES'][:40]+'…' if len(r['SMILES'])>40 else r['SMILES']
            pdf.set_x(14)
            pdf.cell(0, 5, f'{s}  →  {prop}={r[prop]:.4f}  GS={r["Green_score"]:.4f}', ln=True)
        pdf.ln(2)
        if pdf.get_y() > 265: pdf.add_page()

    # ── PAGE 5: Methodology ───────────────────────────────────
    pdf.add_page()
    pdf.section('6. Methodology')
    pdf.set_font('Helvetica','',9)
    methodology = [
        ('VAE Architecture',
         'Character-level GRU VAE (2-layer encoder, 2-layer decoder). '
         'Latent dim=32, hidden=512, embed=128. '
         'Trained on PI1M dataset (up to 50K polymers).'),
        ('Property Models',
         'XGBoost + Random Forest ensemble for each property. '
         'GNN (Graph Attention Network, 4 layers, hidden=256) trained where graph data available. '
         'Ensemble weights: XGB 45%, RF 35%, GNN 20%.'),
        ('Biodegradability',
         'Trained on tsudalab degradability ranking + DeepChem biodeg + curated ground truth. '
         'Normalised to [0,1]. Higher = more biodegradable.'),
        ('Toxicity',
         'Trained on Tox21 (7831 molecules) + ToxCast (8597 molecules). '
         'Score = 1 - mean_toxicity. Higher = safer.'),
        ('Synthesizability',
         'SA Score from RDKit (1-10 scale). Normalised: score = 1 - (SA-1)/9. Higher = easier to synthesise.'),
        ('Recyclability',
         'Binary classifier trained on structural proxy labels from polyVERSE. Higher = more recyclable.'),
        ('Strength',
         'Regression on mechanical strength proxies from polyVERSE (tensile, modulus, Young\'s). '
         'Normalised to [0,1].'),
        ('Tg (Glass Transition)',
         'Regression on Tg data from polyVERSE thermal columns. Normalised. Higher = higher Tg.'),
        ('Solubility',
         'Regression on ESOL dataset (logS). Normalised to [0,1]. Higher = more soluble.'),
        ('Green Score',
         'Weighted sum of all 7 properties. Weights are user-adjustable and auto-normalised. '
         'Range [0,1]. Higher = greener polymer.'),
        ('Generation',
         'Prior sampling: z ~ N(0,I) * temperature. Perturbation: z = mu + sigma * noise * eps. '
         'Interpolation: linear in latent space between two encoded SMILES.'),
    ]
    for title, body in methodology:
        pdf.set_font('Helvetica','B',9); pdf.set_text_color(*PolymerReport.ACCENT)
        pdf.cell(0, 6, f'  {title}:', ln=True)
        pdf.set_font('Helvetica','',8); pdf.set_text_color(0,0,0)
        pdf.set_x(14); pdf.multi_cell(0, 5, body)
        pdf.ln(2)

    out = tmp_path('polymer_report_full.pdf')
    pdf.output(out)
    return out
