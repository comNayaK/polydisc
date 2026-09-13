#!/usr/bin/env python3
# =============================================================
# app.py — Entry point for Polymer Discovery AI
#
# HOW TO RUN (once, forever):
#   pip install -r requirements.txt
#   python app.py
#
# On first run: downloads datasets + trains models → saved to ./saved_models/
# On every run after: loads models instantly (no retraining).
# Run as a systemd service to keep it alive permanently.
# =============================================================

import sys, os, warnings
warnings.filterwarnings('ignore')

# ── Make all submodules importable ─────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import DEVICE

print('=' * 60)
print('  Polymer Discovery AI — Starting up')
print('=' * 60)
print(f'  PyTorch : {torch.__version__}')
print(f'  CUDA    : {torch.cuda.is_available()}')
print(f'  Device  : {DEVICE}')
print('=' * 60)

# ── Step 1: Build FastAPI app ─────────────────────────────────
print('\n[STARTUP] Building FastAPI app...')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import api

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load datasets
    print('\n' + '='*40)
    print(' [STARTUP] STAGE 1: LOADING DATASETS')
    print('='*40)
    from data.loader import load_all_data
    data = load_all_data()

    # Load/train property models
    print('\n' + '='*40)
    print(' [STARTUP] STAGE 2: PREDICTIVE MODELS')
    print('='*40)
    print(' This may take several minutes if training from scratch...')
    from models.predictors import load_or_train_models
    ensembles = load_or_train_models(data)

    from scoring.green_score import register_models
    register_models(ensembles)
    print(' >> Predictive models ready.')

    # Load/train VAE
    print('\n' + '='*40)
    print(' [STARTUP] STAGE 3: GENERATIVE VAE')
    print('='*40)
    from models.vae import load_or_train_vae
    vae, char_to_idx, idx_to_char, train_tensor = load_or_train_vae(data['df_struct'])

    from generation.generator import register_vae
    register_vae(vae, char_to_idx, idx_to_char)
    print(' >> VAE ready.')

    # Store train_set for routes
    api._train_set = data['train_set']
    
    print('\n' + '='*40)
    print(' [STARTUP] COMPLETE: SERVER READY')
    print('='*40 + '\n')
    yield

app = FastAPI(
    title='Polymer Discovery AI',
    description='Generative AI for Sustainable Polymer Discovery — REST API',
    version='2.0.0',
    lifespan=lifespan
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.routes import router
app.include_router(router)

print('  API ready.')

# ── Step 2: Launch ────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8000

    print('\n' + '=' * 60)
    print(f'  Launching API on http://{SERVER_HOST}:{SERVER_PORT}')
    print(f'  API docs at http://{SERVER_HOST}:{SERVER_PORT}/docs')
    print('  Models will NOT retrain — loaded from ./saved_models/')
    print('  To retrain: set FORCE_RETRAIN_* = True in config.py')
    print('=' * 60 + '\n')

    uvicorn.run("app:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
