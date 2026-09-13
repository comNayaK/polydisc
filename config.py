# =============================================================
# config.py — Central configuration for Polymer Discovery App
# =============================================================

import os, torch

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR  = os.path.join(BASE_DIR, 'saved_models')
DATA_DIR  = os.path.join(BASE_DIR, 'data', 'raw')
TMP_DIR   = '/tmp/polymer_app'

for d in [SAVE_DIR, DATA_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)

def save_path(fname): return os.path.join(SAVE_DIR, fname)
def data_path(fname): return os.path.join(DATA_DIR,  fname)
def tmp_path(fname):  return os.path.join(TMP_DIR,   fname)

# ── Device ────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XGB_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Training flags — set False after first successful run ──────
FORCE_RETRAIN_PREDICTORS = False
FORCE_RETRAIN_GNN        = False
FORCE_RETRAIN_VAE        = False

# ── Fingerprint ───────────────────────────────────────────────
FP_RADIUS = 2
FP_BITS   = 2048

# ── VAE ───────────────────────────────────────────────────────
VAE_MAX_LEN      = 120
VAE_NUM_EPOCHS   = 100
VAE_BATCH_SIZE   = 256
VAE_LATENT_DIM   = 32
VAE_HIDDEN_DIM   = 512
VAE_EMBED_DIM    = 128
VAE_BETA_MAX     = 0.40
VAE_FREE_BITS    = 0.10
VAE_CYCLE_EPOCHS = 10
VAE_MAX_PATIENCE = 20
VAE_LR           = 3e-4

# ── GNN ───────────────────────────────────────────────────────
GNN_HIDDEN_DIM  = 256
GNN_NUM_LAYERS  = 4
GNN_DROPOUT     = 0.2
GNN_LR          = 1e-3
GNN_EPOCHS      = 80
GNN_BATCH_SIZE  = 64
GNN_PATIENCE    = 15

# ── Model files ───────────────────────────────────────────────
MODEL_FILES = [
    'sa_model','bio_model','tox_model','recycle_model',
    'strength_model','tg_model','solubility_model',
    'bio_clf','tox_clf',
]

# ── Green Score default weights (must sum to 1) ───────────────
DEFAULT_WEIGHTS = dict(
    biodeg=0.22, synth=0.13, tox=0.18,
    recycl=0.13, strength=0.10,
    tg=0.12, solubility=0.12,
)

# ── Server ────────────────────────────────────────────────────
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 7860
SHARE       = False

# ── Session ───────────────────────────────────────────────────
MAX_HISTORY = 200
