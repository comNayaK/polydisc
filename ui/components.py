# =============================================================
# ui/components.py — Gradio tabbed UI
# =============================================================

import datetime
import pandas as pd
import gradio as gr

from features.featurizer import is_valid
from generation.generator import generate_smiles, perturb_smiles, interpolate_smiles
from scoring.green_score  import compute_green_score, PROPERTY_COLS
from ui.visualizations    import (make_radar, make_bar, make_heatmap, make_breakdown,
                                   make_distribution, make_structs, make_correlation,
                                   make_violin, make_comparison, make_umap, save_img)
from reports.pdf_report   import build_pdf
from config import DEFAULT_WEIGHTS, tmp_path

# ── Per-user session state ─────────────────────────────────────
# Gradio State objects hold independent data per user session.

def _empty_session():
    return {
        'history'    : [],   # list of dicts
        'favourites' : [],   # list of SMILES strings
        'last_df'    : None, # last results DataFrame
        'theme'      : 'dark',
    }


# ── Core discovery function ────────────────────────────────────
def run_discovery(user_input, mode,
                  w_bio, w_sa, w_tox, w_rec, w_str, w_tg, w_sol,
                  temp, n_gen, train_set, session):
    log          = []
    candidates   = []
    total_gen    = 0
    n_gen        = int(n_gen)
    theme        = session.get('theme', 'dark')
    weights      = dict(biodeg=w_bio, synth=w_sa, tox=w_tox,
                        recycl=w_rec, strength=w_str, tg=w_tg, solubility=w_sol)

    # ── Generation ─────────────────────────────────────────
    if mode in ('Generate new materials', 'Both'):
        log.append(f'[{_ts()}] Generating {n_gen} candidates (temp={temp:.2f})...')
        try:
            raw       = generate_smiles(n=n_gen, temp=float(temp))
            total_gen += len(raw)
            for s in raw:
                ok, c = is_valid(s)
                if ok and c not in train_set: candidates.append(c)
            log.append(f'  → {len(candidates)} valid novel structures')
        except Exception as e: log.append(f'  Generation error: {e}')

    if mode in ('Encode & perturb input', 'Both') and user_input.strip():
        log.append(f'[{_ts()}] Perturbing input in latent space...')
        try:
            raw2      = perturb_smiles(user_input.strip(), n=n_gen)
            total_gen += len(raw2)
            for s in raw2:
                ok, c = is_valid(s)
                if ok: candidates.append(c)
            candidates = list(dict.fromkeys(candidates))
            log.append(f'  → {len(candidates)} unique candidates')
        except Exception as e: log.append(f'  Perturbation error: {e}')

    valid_novel = len(candidates)
    if not candidates:
        return (_err('No valid candidates — try higher N or temperature.'),
                *([None]*10), session)

    # ── Scoring ─────────────────────────────────────────────
    log.append(f'[{_ts()}] Scoring {len(candidates)} candidates...')
    try:
        df  = compute_green_score(candidates, weights=weights)
        top = df.head(5)
        log.append(f'  Top GS   : {top["Green_score"].iloc[0]:.4f}')
        log.append(f'  Mean GS  : {df["Green_score"].mean():.4f}')
        log.append(f'  Std  GS  : {df["Green_score"].std():.4f}')
        log.append(f'  Candidates: {len(df):,}')
        session['last_df'] = df
    except Exception as e:
        return (_err(f'Scoring error: {e}'), *([None]*10), session)

    # ── Visualisations ───────────────────────────────────────
    log.append(f'[{_ts()}] Generating visualisations...')
    img_paths = {}

    def _vis(name, fn, *args):
        try:
            img = fn(*args)
            if img:
                save_img(img, name)
                img_paths[name] = tmp_path(f'{name}.png')
            return img
        except Exception as e:
            log.append(f'  {name} error: {e}')
            return None

    radar_img     = _vis('radar',     make_radar,        top, theme)
    bar_img       = _vis('bar',       make_bar,          top, theme)
    heatmap_img   = _vis('heatmap',   make_heatmap,      top, theme)
    breakdown_img = _vis('breakdown', make_breakdown,    top, weights, theme)
    dist_img      = _vis('dist',      make_distribution, df,  theme)
    violin_img    = _vis('violin',    make_violin,       df,  theme)
    corr_img      = _vis('corr',      make_correlation,  df,  theme)
    struct_img    = _vis('struct',    make_structs,      top)
    umap_img      = _vis('umap',      make_umap,         df,  theme)

    # ── PDF & CSV ────────────────────────────────────────────
    log.append(f'[{_ts()}] Building PDF report...')
    pdf_path = csv_path = None
    try:
        pdf_path = build_pdf(top, df, weights, img_paths,
                             total_gen, valid_novel, mode, temp, n_gen)
        log.append('  PDF ready ✓')
    except Exception as e: log.append(f'  PDF error: {e}')

    try:
        csv_path = tmp_path('all_candidates.csv')
        df.to_csv(csv_path, index=False)
        log.append('  CSV ready ✓')
    except Exception as e: log.append(f'  CSV error: {e}')

    # ── Update session history ────────────────────────────────
    session['history'].append({
        'time'       : _ts(),
        'mode'       : mode,
        'n_candidates': len(df),
        'top_gs'     : round(float(top['Green_score'].iloc[0]), 4),
        'top_smiles' : top['SMILES'].iloc[0],
        'input'      : user_input.strip() or '—',
    })
    if len(session['history']) > 200:
        session['history'] = session['history'][-200:]

    log.append(f'\n[{_ts()}] Done ✓')
    table = top[['SMILES','Biodeg','Synth','Tox','Recyclability',
                 'Strength','Tg','Solubility','Green_score']].to_string(
        float_format='{:.4f}'.format)
    status = '\n'.join(log) + '\n\n' + table

    return (status,
            radar_img, bar_img, heatmap_img, breakdown_img,
            dist_img, violin_img, corr_img, struct_img, umap_img,
            pdf_path, csv_path,
            session)


# ── Interpolation function ─────────────────────────────────────
def run_interpolation(smi_a, smi_b, steps, session):
    theme = session.get('theme','dark')
    results = interpolate_smiles(smi_a, smi_b, steps=int(steps))
    valid   = [(s,c) for s in results for ok,c in [is_valid(s)] if ok]
    if not valid:
        return 'No valid structures along interpolation path.', None, None

    smiles_list = [c for _,c in valid]
    df  = compute_green_score(smiles_list)
    top = df.head(min(5, len(df)))

    bar   = make_bar(top, theme=theme)
    radar = make_radar(top, theme=theme)

    lines = ['Interpolation results:\n']
    for i, row in df.iterrows():
        lines.append(f'  Step {i+1}: {row["SMILES"]}  GS={row["Green_score"]:.4f}')
    return '\n'.join(lines), bar, radar


# ── Comparison function ────────────────────────────────────────
def run_comparison(smi_a, smi_b, session):
    theme = session.get('theme','dark')
    ok_a, ca = is_valid(smi_a)
    ok_b, cb = is_valid(smi_b)
    if not ok_a: return 'Invalid SMILES A', None
    if not ok_b: return 'Invalid SMILES B', None

    df = compute_green_score([ca, cb])
    row_a = df.iloc[0].to_dict() if df.iloc[0]['SMILES']==ca else df.iloc[1].to_dict()
    row_b = df.iloc[1].to_dict() if df.iloc[1]['SMILES']==cb else df.iloc[0].to_dict()

    chart = make_comparison(row_a, row_b, theme=theme)

    lines = ['=== Molecule A ===']
    for p in PROPERTY_COLS + ['Green_score']:
        lines.append(f'  {p:15s}: {row_a.get(p,0):.4f}')
    lines.append('\n=== Molecule B ===')
    for p in PROPERTY_COLS + ['Green_score']:
        lines.append(f'  {p:15s}: {row_b.get(p,0):.4f}')
    lines.append(f'\nWinner: {"A" if row_a["Green_score"]>=row_b["Green_score"] else "B"} '
                 f'(GS difference: {abs(row_a["Green_score"]-row_b["Green_score"]):.4f})')

    return '\n'.join(lines), chart


# ── Favourites ────────────────────────────────────────────────
def add_favourite(smi, session):
    ok, canon = is_valid(smi)
    if not ok: return 'Invalid SMILES — not added.', session
    if canon not in session['favourites']:
        session['favourites'].append(canon)
    return f'★ Added to favourites ({len(session["favourites"])} total)', session

def get_favourites_table(session):
    favs = session.get('favourites', [])
    if not favs: return 'No favourites yet.'
    df = compute_green_score(favs)
    return df[['SMILES','Biodeg','Synth','Tox','Recyclability',
               'Strength','Tg','Solubility','Green_score']].to_string(
        float_format='{:.4f}'.format)

def clear_favourites(session):
    session['favourites'] = []
    return 'Favourites cleared.', session

def export_favourites_csv(session):
    favs = session.get('favourites',[])
    if not favs: return None
    df  = compute_green_score(favs)
    out = tmp_path('favourites.csv')
    df.to_csv(out, index=False)
    return out


# ── History ───────────────────────────────────────────────────
def get_history_table(session):
    hist = session.get('history', [])
    if not hist: return 'No runs yet this session.'
    rows = []
    for h in reversed(hist[-30:]):
        rows.append(f'{h["time"]}  |  {h["mode"][:20]:20s}  |  '
                    f'n={h["n_candidates"]:4d}  |  '
                    f'Top GS={h["top_gs"]:.4f}  |  {h["top_smiles"][:30]}')
    return '\n'.join(rows)

def clear_history(session):
    session['history'] = []
    return 'History cleared.', session


# ── Theme toggle ──────────────────────────────────────────────
def toggle_theme(session):
    session['theme'] = 'light' if session['theme']=='dark' else 'dark'
    label = '☀ Light mode ON' if session['theme']=='light' else '🌙 Dark mode ON'
    return label, session


# ── Helpers ───────────────────────────────────────────────────
def _ts():
    return datetime.datetime.now().strftime('%H:%M:%S')

def _err(msg):
    return f'❌ {msg}'

def _w_slider(label, key, default=None):
    v = default if default is not None else DEFAULT_WEIGHTS.get(key, 0.15)
    return gr.Slider(0, 1, value=v, step=0.01, label=label)


# ── Build full Gradio app ─────────────────────────────────────
def build_app(train_set):
    """Constructs and returns the Gradio Blocks app."""

    CSS = """
    .gradio-container { max-width: 1300px !important; margin: auto; }
    .tab-nav button { font-size: 14px; font-weight: 600; padding: 10px 20px; }
    .gr-button-primary {
        background: linear-gradient(135deg, #00b894, #007a60) !important;
        font-size: 15px !important; font-weight: 700 !important;
    }
    footer { display: none !important; }
    .status-box textarea { font-family: monospace !important; font-size: 12px !important; }
    """

    HEADER_HTML = """
    <div style="text-align:center;padding:30px 0 18px;
                background:linear-gradient(135deg,#0a1628,#0e2018);
                border-radius:12px;margin-bottom:10px;
                border:1px solid rgba(0,184,148,0.25)">
      <h1 style="color:#00b894;font-size:28px;margin:0 0 8px;
                 font-family:Georgia,serif;letter-spacing:-0.5px">
        🧬 Generative AI for Sustainable Polymer Discovery
      </h1>
      <p style="color:#6b8f80;margin:0;font-size:13px">
        VAE generation &nbsp;·&nbsp;
        XGBoost + RF + GNN Ensemble &nbsp;·&nbsp;
        7-Property Green Score &nbsp;·&nbsp;
        Multi-user hosted service
      </p>
      <p style="color:#3d6b5a;margin:6px 0 0;font-size:11px;font-family:monospace">
        PI1M · tsudalab · Tox21 · ToxCast · ESOL · polyVERSE
      </p>
    </div>
    """

    with gr.Blocks(
        title='Polymer Discovery AI',
        theme=gr.themes.Soft(primary_hue='green', neutral_hue='slate'),
        css=CSS
    ) as app:

        # ── Session state ─────────────────────────────────
        session  = gr.State(_empty_session)
        train_st = gr.State(train_set)

        gr.HTML(HEADER_HTML)

        # ── Theme toggle ──────────────────────────────────
        with gr.Row():
            theme_btn = gr.Button('🌙 Toggle Dark / Light', size='sm', scale=0)
            theme_lbl = gr.Textbox(value='Dark mode ON', show_label=False,
                                   interactive=False, scale=0, max_lines=1)

        # ═══════════════════════════════════════════════
        # TAB 1 — Discovery
        # ═══════════════════════════════════════════════
        with gr.Tab('🚀 Discovery'):
            with gr.Row():

                # LEFT PANEL
                with gr.Column(scale=1, min_width=340):
                    with gr.Group():
                        gr.Markdown('### 🔬 Input')
                        user_input = gr.Textbox(
                            label='Seed SMILES (optional)',
                            placeholder='Leave blank to generate fresh, or paste a polymer SMILES',
                            lines=2)
                        mode = gr.Radio(
                            ['Generate new materials',
                             'Encode & perturb input',
                             'Both'],
                            value='Generate new materials',
                            label='Discovery mode')

                    with gr.Accordion('⚖️ Green Score Weights (auto-normalised)', open=True):
                        w_bio = _w_slider('🌿 Biodegradability', 'biodeg', 0.22)
                        w_sa  = _w_slider('⚗  Synthesizability', 'synth',  0.13)
                        w_tox = _w_slider('🛡  Low Toxicity',     'tox',    0.18)
                        w_rec = _w_slider('♻  Recyclability',   'recycl',  0.13)
                        w_str = _w_slider('💪 Mech. Strength',  'strength', 0.10)
                        w_tg  = _w_slider('🌡 Glass Trans. Tg', 'tg',       0.12)
                        w_sol = _w_slider('💧 Solubility',      'solubility',0.12)

                    with gr.Accordion('⚙️ Generation Settings', open=True):
                        temp  = gr.Slider(0.4, 2.0, value=1.0, step=0.05,
                                          label='VAE Temperature  (↑ = more diverse)')
                        n_gen = gr.Slider(100, 3000, value=500, step=100,
                                          label='N candidates')

                    run_btn = gr.Button('🚀 Run Discovery', variant='primary', size='lg')

                    with gr.Accordion('⭐ Favourites', open=False):
                        fav_input  = gr.Textbox(label='SMILES to star', lines=1)
                        fav_btn    = gr.Button('★ Add to Favourites', size='sm')
                        fav_status = gr.Textbox(label='Status', interactive=False, lines=1)
                        fav_show   = gr.Button('📋 Show Favourites', size='sm')
                        fav_table  = gr.Textbox(label='Starred Molecules',
                                                interactive=False, lines=6)
                        fav_csv    = gr.File(label='Export Favourites CSV')
                        with gr.Row():
                            fav_export_btn = gr.Button('⬇ Export CSV', size='sm')
                            fav_clear_btn  = gr.Button('🗑 Clear', size='sm')

                # RIGHT PANEL
                with gr.Column(scale=2):
                    status_box = gr.Textbox(
                        label='📋 Status & Top 5 Results',
                        lines=14, interactive=False, elem_classes=['status-box'])

                    gr.Markdown('### 📊 Visualisations')

                    with gr.Row():
                        radar_out = gr.Image(label='Radar (Top 3)',    type='pil', height=350)
                        bar_out   = gr.Image(label='GS Ranking',       type='pil', height=350)

                    with gr.Row():
                        heatmap_out   = gr.Image(label='Property Heatmap', type='pil', height=300)
                        breakdown_out = gr.Image(label='GS Breakdown',     type='pil', height=300)

                    with gr.Row():
                        dist_out   = gr.Image(label='Distribution',  type='pil', height=300)
                        violin_out = gr.Image(label='Violin Plots',  type='pil', height=300)

                    with gr.Row():
                        corr_out   = gr.Image(label='Correlation',   type='pil', height=320)
                        struct_out = gr.Image(label='2D Structures', type='pil', height=320)

                    umap_out = gr.Image(label='UMAP Chemical Space', type='pil', height=360)

                    gr.Markdown('### 💾 Downloads')
                    with gr.Row():
                        pdf_out = gr.File(label='📄 Full PDF Report (6 pages)')
                        csv_out = gr.File(label='📊 All Candidates CSV')

            gr.Examples(
                examples=[
                    ['CC(=O)Oc1ccc(cc1)OC(C)=O',
                     'Encode & perturb input',
                     0.30,0.15,0.18,0.13,0.10,0.08,0.06, 1.0, 400],
                    ['', 'Generate new materials',
                     0.22,0.13,0.18,0.13,0.10,0.12,0.12, 1.2, 600],
                    ['C(=O)(O)c1ccc(cc1)C(=O)OCCO',
                     'Both',
                     0.35,0.10,0.20,0.10,0.10,0.10,0.05, 0.9, 500],
                ],
                inputs=[user_input, mode,
                        w_bio, w_sa, w_tox, w_rec, w_str, w_tg, w_sol,
                        temp, n_gen],
                label='Example queries'
            )

        # ═══════════════════════════════════════════════
        # TAB 2 — Analysis
        # ═══════════════════════════════════════════════
        with gr.Tab('🔬 Analysis'):
            gr.Markdown('### ↔ Side-by-Side Comparison')
            with gr.Row():
                cmp_a = gr.Textbox(label='Molecule A (SMILES)', lines=1,
                                   placeholder='CC(=O)Oc1ccc(cc1)OC(C)=O')
                cmp_b = gr.Textbox(label='Molecule B (SMILES)', lines=1,
                                   placeholder='OC(=O)CCCCC(=O)OCCO')
            cmp_btn  = gr.Button('Compare', variant='primary')
            cmp_txt  = gr.Textbox(label='Comparison Results', lines=14, interactive=False)
            cmp_img  = gr.Image(label='Property Comparison Chart', type='pil', height=350)

            gr.Markdown('### 🧪 Latent Space Interpolation')
            gr.Markdown('Generate molecules along the path between two SMILES in latent space.')
            with gr.Row():
                int_a   = gr.Textbox(label='Start SMILES', lines=1)
                int_b   = gr.Textbox(label='End SMILES',   lines=1)
                int_steps = gr.Slider(4, 16, value=8, step=1, label='Steps')
            int_btn  = gr.Button('Interpolate', variant='primary')
            int_txt  = gr.Textbox(label='Interpolation Results', lines=10, interactive=False)
            with gr.Row():
                int_bar   = gr.Image(label='GS Along Path', type='pil', height=300)
                int_radar = gr.Image(label='Property Radar', type='pil', height=300)

        # ═══════════════════════════════════════════════
        # TAB 3 — History
        # ═══════════════════════════════════════════════
        with gr.Tab('📜 History'):
            gr.Markdown('### Session Run History')
            gr.Markdown('All runs from your current session are logged below.')
            hist_refresh = gr.Button('🔄 Refresh History')
            hist_box     = gr.Textbox(label='Recent Runs (newest first)',
                                      lines=20, interactive=False)
            hist_clear   = gr.Button('🗑 Clear History', size='sm')

        # ═══════════════════════════════════════════════
        # TAB 4 — About
        # ═══════════════════════════════════════════════
        with gr.Tab('ℹ️ About'):
            gr.Markdown("""
## About This App

**Generative AI for Sustainable Polymer Discovery** uses a combination of deep generative
modelling and ensemble machine learning to discover novel, green polymers.

---

### Architecture

| Component | Details |
|---|---|
| **Generator** | Character-level GRU VAE (latent dim=32, hidden=512) |
| **Property models** | XGBoost + Random Forest + Graph Attention Network (ensemble) |
| **Properties** | Biodegradability, Synthesizability, Toxicity, Recyclability, Strength, Tg, Solubility |
| **Green Score** | Weighted sum of 7 normalised properties |

---

### Datasets

| Dataset | Source | Used for |
|---|---|---|
| PI1M | RUIMINMA1996/PI1M | VAE training corpus |
| tsudalab | tsudalab/Polymer-degradability-ranking | Biodegradability |
| Tox21 | DeepChem | Toxicity |
| ToxCast | DeepChem | Toxicity |
| ESOL | DeepChem | Aqueous Solubility |
| polyVERSE | Ramprasad-Group/polyVERSE | Mechanical strength, Tg, Recyclability |

---

### Features

- **Dark / Light theme** toggle
- **Tabbed interface**: Discovery, Analysis, History, About
- **SMILES history** with timestamps (per session)
- **Favourites**: star any molecule and export as CSV
- **Side-by-side comparison** of two molecules
- **Latent space interpolation** between two SMILES
- **9 visualisations**: Radar, Bar, Heatmap, Breakdown, Distribution, Violin, Correlation, Structures, UMAP
- **6-page PDF report** with methodology section
- **Multi-user safe**: each user has independent session state

---

### How to Use

1. **Discovery tab**: Set weights, choose mode, click Run
2. **Analysis tab**: Compare two molecules or interpolate between them
3. **History tab**: Review past runs from this session
4. Star molecules you like using the ⭐ Favourites panel
            """)

        # ── Wiring ────────────────────────────────────────────

        disc_outputs = [
            status_box,
            radar_out, bar_out, heatmap_out, breakdown_out,
            dist_out, violin_out, corr_out, struct_out, umap_out,
            pdf_out, csv_out,
            session,
        ]

        run_btn.click(
            fn=run_discovery,
            inputs=[user_input, mode,
                    w_bio, w_sa, w_tox, w_rec, w_str, w_tg, w_sol,
                    temp, n_gen, train_st, session],
            outputs=disc_outputs,
        )

        theme_btn.click(fn=toggle_theme,
                        inputs=[session], outputs=[theme_lbl, session])

        # Favourites
        fav_btn.click(fn=add_favourite,
                      inputs=[fav_input, session],
                      outputs=[fav_status, session])
        fav_show.click(fn=get_favourites_table,
                       inputs=[session], outputs=[fav_table])
        fav_export_btn.click(fn=export_favourites_csv,
                             inputs=[session], outputs=[fav_csv])
        fav_clear_btn.click(fn=clear_favourites,
                            inputs=[session],
                            outputs=[fav_status, session])

        # Comparison
        cmp_btn.click(fn=run_comparison,
                      inputs=[cmp_a, cmp_b, session],
                      outputs=[cmp_txt, cmp_img])

        # Interpolation
        int_btn.click(fn=run_interpolation,
                      inputs=[int_a, int_b, int_steps, session],
                      outputs=[int_txt, int_bar, int_radar])

        # History
        hist_refresh.click(fn=get_history_table,
                           inputs=[session], outputs=[hist_box])
        hist_clear.click(fn=clear_history,
                         inputs=[session],
                         outputs=[hist_box, session])

    return app
