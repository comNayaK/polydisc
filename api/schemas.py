# =============================================================
# api/schemas.py — Pydantic request/response models
# =============================================================

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ── Discovery ─────────────────────────────────────────────────
class DiscoveryRequest(BaseModel):
    smiles_input: str = ""
    mode: str = "Generate new materials"
    w_bio: float = 0.22
    w_sa: float = 0.13
    w_tox: float = 0.18
    w_rec: float = 0.13
    w_str: float = 0.10
    w_tg: float = 0.12
    w_sol: float = 0.12
    temperature: float = 1.0
    n_candidates: int = 500


class CandidateRow(BaseModel):
    smiles: str
    biodeg: float
    synth: float
    tox: float
    recyclability: float
    strength: float
    tg: float
    solubility: float
    green_score: float


class DiscoveryResponse(BaseModel):
    status: str
    log: List[str]
    candidates: List[CandidateRow]
    charts: Dict[str, Optional[str]]   # name → base64 PNG
    pdf_available: bool = False
    csv_available: bool = False


# ── Compare ───────────────────────────────────────────────────
class CompareRequest(BaseModel):
    smiles_a: str
    smiles_b: str


class CompareResponse(BaseModel):
    text: str
    chart: Optional[str] = None  # base64


# ── Interpolate ───────────────────────────────────────────────
class InterpolateRequest(BaseModel):
    smiles_a: str
    smiles_b: str
    steps: int = 8


class InterpolateResponse(BaseModel):
    text: str
    bar_chart: Optional[str] = None
    radar_chart: Optional[str] = None


# ── Favourites ────────────────────────────────────────────────
class FavouriteRequest(BaseModel):
    smiles: str


class FavouritesResponse(BaseModel):
    favourites: List[CandidateRow]
    count: int


# ── History ───────────────────────────────────────────────────
class HistoryEntry(BaseModel):
    time: str
    mode: str
    n_candidates: int
    top_gs: float
    top_smiles: str
    input: str


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]


# ── Config ────────────────────────────────────────────────────
class ConfigResponse(BaseModel):
    default_weights: Dict[str, float]
    property_cols: List[str]
    modes: List[str]


# ── AI Chat ───────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    config: Optional[Dict[str, Any]] = None
