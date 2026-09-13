# =============================================================
# api/routes.py — FastAPI REST endpoints
# =============================================================

import datetime
import base64
import io
import uuid
from typing import Optional
from fastapi import APIRouter, Request, Response, Cookie
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import (
    DiscoveryRequest, DiscoveryResponse, CandidateRow,
    CompareRequest, CompareResponse,
    InterpolateRequest, InterpolateResponse,
    FavouriteRequest, FavouritesResponse,
    HistoryEntry, HistoryResponse,
    ConfigResponse,
    ChatRequest, ChatResponse,
)
from features.featurizer import is_valid
from generation.generator import generate_smiles, perturb_smiles, interpolate_smiles
from scoring.green_score import compute_green_score, PROPERTY_COLS
from ui.visualizations import (
    make_radar, make_bar, make_heatmap, make_breakdown,
    make_distribution, make_structs, make_correlation,
    make_violin, make_comparison, make_umap, save_img,
)
from reports.pdf_report import build_pdf
from config import DEFAULT_WEIGHTS, tmp_path
from api.ai_assistant import chat_with_groq

router = APIRouter(prefix="/api")

# ── In-memory session store ───────────────────────────────────
_sessions = {}

def _get_session(session_id: Optional[str]) -> tuple[str, dict]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "history": [],
        "favourites": [],
        "last_df": None,
        "last_pdf": None,
        "last_csv": None,
        "theme": "light",
    }
    return sid, _sessions[sid]

def _ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def _img_to_base64(img) -> Optional[str]:
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Config ────────────────────────────────────────────────────
@router.get("/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(
        default_weights=DEFAULT_WEIGHTS,
        property_cols=PROPERTY_COLS,
        modes=["Generate new materials", "Encode & perturb input", "Both"],
    )


# ── Discovery ─────────────────────────────────────────────────
@router.post("/discovery")
def run_discovery_endpoint(req: DiscoveryRequest, session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)

    # Import train_set from the global app state
    from api import _train_set

    log = []
    candidates = []
    total_gen = 0
    weights = dict(
        biodeg=req.w_bio, synth=req.w_sa, tox=req.w_tox,
        recycl=req.w_rec, strength=req.w_str, tg=req.w_tg, solubility=req.w_sol,
    )

    # ── Generation
    if req.mode in ("Generate new materials", "Both"):
        log.append(f"[{_ts()}] Generating {req.n_candidates} candidates (temp={req.temperature:.2f})...")
        try:
            raw = generate_smiles(n=req.n_candidates, temp=float(req.temperature))
            total_gen += len(raw)
            for s in raw:
                ok, c = is_valid(s)
                if ok and c not in _train_set:
                    candidates.append(c)
            log.append(f"  → {len(candidates)} valid novel structures")
        except Exception as e:
            log.append(f"  Generation error: {e}")

    if req.mode in ("Encode & perturb input", "Both") and req.smiles_input.strip():
        log.append(f"[{_ts()}] Perturbing input in latent space...")
        try:
            raw2 = perturb_smiles(req.smiles_input.strip(), n=req.n_candidates)
            total_gen += len(raw2)
            for s in raw2:
                ok, c = is_valid(s)
                if ok:
                    candidates.append(c)
            candidates = list(dict.fromkeys(candidates))
            log.append(f"  → {len(candidates)} unique candidates")
        except Exception as e:
            log.append(f"  Perturbation error: {e}")

    valid_novel = len(candidates)
    if not candidates:
        resp = DiscoveryResponse(
            status="error",
            log=["❌ No valid candidates — try higher N or temperature."],
            candidates=[],
            charts={},
        )
        response = JSONResponse(content=resp.model_dump())
        response.set_cookie("session_id", sid)
        return response

    # ── Scoring
    log.append(f"[{_ts()}] Scoring {len(candidates)} candidates...")
    try:
        df = compute_green_score(candidates, weights=weights)
        top = df.head(5)
        log.append(f"  Top GS   : {top['Green_score'].iloc[0]:.4f}")
        log.append(f"  Mean GS  : {df['Green_score'].mean():.4f}")
        log.append(f"  Std  GS  : {df['Green_score'].std():.4f}")
        log.append(f"  Candidates: {len(df):,}")
        session["last_df"] = df
    except Exception as e:
        resp = DiscoveryResponse(
            status="error",
            log=log + [f"❌ Scoring error: {e}"],
            candidates=[],
            charts={},
        )
        response = JSONResponse(content=resp.model_dump())
        response.set_cookie("session_id", sid)
        return response

    # ── Visualisations
    log.append(f"[{_ts()}] Generating visualisations...")
    theme = session.get("theme", "dark")
    charts = {}
    img_paths = {}

    def _vis(name, fn, *args):
        try:
            img = fn(*args)
            if img:
                save_img(img, name)
                img_paths[name] = tmp_path(f"{name}.png")
                charts[name] = _img_to_base64(img)
            else:
                charts[name] = None
        except Exception as e:
            log.append(f"  {name} error: {e}")
            charts[name] = None

    _vis("radar", make_radar, top, theme)
    _vis("bar", make_bar, top, theme)
    _vis("heatmap", make_heatmap, top, theme)
    _vis("breakdown", make_breakdown, top, weights, theme)
    _vis("dist", make_distribution, df, theme)
    _vis("violin", make_violin, df, theme)
    _vis("corr", make_correlation, df, theme)
    _vis("struct", make_structs, top)
    _vis("umap", make_umap, df, theme)

    # ── PDF & CSV
    log.append(f"[{_ts()}] Building PDF report...")
    pdf_available = False
    csv_available = False
    try:
        pdf_path = build_pdf(top, df, weights, img_paths,
                             total_gen, valid_novel, req.mode, req.temperature, req.n_candidates)
        session["last_pdf"] = pdf_path
        pdf_available = True
        log.append("  PDF ready ✓")
    except Exception as e:
        log.append(f"  PDF error: {e}")

    try:
        csv_path = tmp_path("all_candidates.csv")
        df.to_csv(csv_path, index=False)
        session["last_csv"] = csv_path
        csv_available = True
        log.append("  CSV ready ✓")
    except Exception as e:
        log.append(f"  CSV error: {e}")

    # ── Update session history
    session["history"].append({
        "time": _ts(),
        "mode": req.mode,
        "n_candidates": len(df),
        "top_gs": round(float(top["Green_score"].iloc[0]), 4),
        "top_smiles": top["SMILES"].iloc[0],
        "input": req.smiles_input.strip() or "—",
    })
    if len(session["history"]) > 200:
        session["history"] = session["history"][-200:]

    log.append(f"\n[{_ts()}] Done ✓")

    # Build candidate rows
    cand_rows = []
    for _, row in top.iterrows():
        cand_rows.append(CandidateRow(
            smiles=row["SMILES"],
            biodeg=round(float(row["Biodeg"]), 4),
            synth=round(float(row["Synth"]), 4),
            tox=round(float(row["Tox"]), 4),
            recyclability=round(float(row["Recyclability"]), 4),
            strength=round(float(row["Strength"]), 4),
            tg=round(float(row["Tg"]), 4),
            solubility=round(float(row["Solubility"]), 4),
            green_score=round(float(row["Green_score"]), 4),
        ))

    resp = DiscoveryResponse(
        status="success",
        log=log,
        candidates=cand_rows,
        charts=charts,
        pdf_available=pdf_available,
        csv_available=csv_available,
    )
    response = JSONResponse(content=resp.model_dump())
    response.set_cookie("session_id", sid)
    return response


# ── Compare ───────────────────────────────────────────────────
@router.post("/compare", response_model=CompareResponse)
def run_compare(req: CompareRequest, session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    theme = session.get("theme", "dark")

    ok_a, ca = is_valid(req.smiles_a)
    ok_b, cb = is_valid(req.smiles_b)
    if not ok_a:
        return CompareResponse(text="❌ Invalid SMILES A", chart=None)
    if not ok_b:
        return CompareResponse(text="❌ Invalid SMILES B", chart=None)

    df = compute_green_score([ca, cb])
    row_a = df.iloc[0].to_dict() if df.iloc[0]["SMILES"] == ca else df.iloc[1].to_dict()
    row_b = df.iloc[1].to_dict() if df.iloc[1]["SMILES"] == cb else df.iloc[0].to_dict()

    chart = make_comparison(row_a, row_b, theme=theme)

    lines = ["=== Molecule A ==="]
    for p in PROPERTY_COLS + ["Green_score"]:
        lines.append(f"  {p:15s}: {row_a.get(p, 0):.4f}")
    lines.append("\n=== Molecule B ===")
    for p in PROPERTY_COLS + ["Green_score"]:
        lines.append(f"  {p:15s}: {row_b.get(p, 0):.4f}")
    lines.append(
        f'\nWinner: {"A" if row_a["Green_score"] >= row_b["Green_score"] else "B"} '
        f'(GS difference: {abs(row_a["Green_score"] - row_b["Green_score"]):.4f})'
    )

    return CompareResponse(text="\n".join(lines), chart=_img_to_base64(chart))


# ── Interpolation ─────────────────────────────────────────────
@router.post("/interpolate", response_model=InterpolateResponse)
def run_interpolate(req: InterpolateRequest, session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    theme = session.get("theme", "dark")

    results = interpolate_smiles(req.smiles_a, req.smiles_b, steps=req.steps)
    valid = [(s, c) for s in results for ok, c in [is_valid(s)] if ok]
    if not valid:
        return InterpolateResponse(text="No valid structures along interpolation path.")

    smiles_list = [c for _, c in valid]
    df = compute_green_score(smiles_list)
    top = df.head(min(5, len(df)))

    bar = make_bar(top, theme=theme)
    radar = make_radar(top, theme=theme)

    lines = ["Interpolation results:\n"]
    for i, row in df.iterrows():
        lines.append(f'  Step {i + 1}: {row["SMILES"]}  GS={row["Green_score"]:.4f}')

    return InterpolateResponse(
        text="\n".join(lines),
        bar_chart=_img_to_base64(bar),
        radar_chart=_img_to_base64(radar),
    )


# ── Favourites ────────────────────────────────────────────────
@router.post("/favourite")
def add_favourite(req: FavouriteRequest, session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    ok, canon = is_valid(req.smiles)
    if not ok:
        response = JSONResponse(content={"message": "Invalid SMILES — not added.", "count": len(session["favourites"])})
        response.set_cookie("session_id", sid)
        return response
    if canon not in session["favourites"]:
        session["favourites"].append(canon)
    response = JSONResponse(content={
        "message": f"★ Added to favourites ({len(session['favourites'])} total)",
        "count": len(session["favourites"]),
    })
    response.set_cookie("session_id", sid)
    return response


@router.get("/favourites")
def get_favourites(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    favs = session.get("favourites", [])
    if not favs:
        response = JSONResponse(content={"favourites": [], "count": 0})
        response.set_cookie("session_id", sid)
        return response
    df = compute_green_score(favs)
    rows = []
    for _, row in df.iterrows():
        rows.append(CandidateRow(
            smiles=row["SMILES"],
            biodeg=round(float(row["Biodeg"]), 4),
            synth=round(float(row["Synth"]), 4),
            tox=round(float(row["Tox"]), 4),
            recyclability=round(float(row["Recyclability"]), 4),
            strength=round(float(row["Strength"]), 4),
            tg=round(float(row["Tg"]), 4),
            solubility=round(float(row["Solubility"]), 4),
            green_score=round(float(row["Green_score"]), 4),
        ).model_dump())
    response = JSONResponse(content={"favourites": rows, "count": len(rows)})
    response.set_cookie("session_id", sid)
    return response


@router.delete("/favourites/clear")
def clear_favourites(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    session["favourites"] = []
    response = JSONResponse(content={"message": "Favourites cleared.", "count": 0})
    response.set_cookie("session_id", sid)
    return response


@router.get("/favourites/export")
def export_favourites(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    favs = session.get("favourites", [])
    if not favs:
        return JSONResponse(content={"error": "No favourites to export"}, status_code=400)
    df = compute_green_score(favs)
    out = tmp_path("favourites.csv")
    df.to_csv(out, index=False)
    return FileResponse(out, filename="favourites.csv", media_type="text/csv")


# ── History ───────────────────────────────────────────────────
@router.get("/history")
def get_history(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    entries = []
    for h in reversed(session.get("history", [])[-30:]):
        entries.append(HistoryEntry(**h).model_dump())
    response = JSONResponse(content={"entries": entries})
    response.set_cookie("session_id", sid)
    return response


@router.delete("/history/clear")
def clear_history(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    session["history"] = []
    response = JSONResponse(content={"message": "History cleared.", "entries": []})
    response.set_cookie("session_id", sid)
    return response


# ── Downloads ─────────────────────────────────────────────────
@router.get("/download/pdf")
def download_pdf(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    path = session.get("last_pdf")
    if not path:
        return JSONResponse(content={"error": "No PDF available. Run discovery first."}, status_code=404)
    return FileResponse(path, filename="polymer_report.pdf", media_type="application/pdf")


@router.get("/download/csv")
def download_csv(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    path = session.get("last_csv")
    if not path:
        return JSONResponse(content={"error": "No CSV available. Run discovery first."}, status_code=404)
    return FileResponse(path, filename="all_candidates.csv", media_type="text/csv")


# ── Theme ─────────────────────────────────────────────────────
@router.post("/theme/toggle")
def toggle_theme(session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    session["theme"] = "light" if session["theme"] == "dark" else "dark"
    response = JSONResponse(content={"theme": session["theme"]})
    response.set_cookie("session_id", sid)
    return response


# ── AI Chat ───────────────────────────────────────────────────
@router.post("/chat")
def chat_endpoint(req: ChatRequest, session_id: Optional[str] = Cookie(None)):
    sid, session = _get_session(session_id)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    result = chat_with_groq(messages)
    response = JSONResponse(content={
        "reply": result["reply"],
        "config": result.get("config"),
    })
    response.set_cookie("session_id", sid)
    return response
