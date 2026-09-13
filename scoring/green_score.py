# =============================================================
# scoring/green_score.py — Property prediction & Green Score
# =============================================================

import numpy as np
import pandas as pd
from features.featurizer import smiles_to_fp
from config import DEFAULT_WEIGHTS

# Global model registry — set by app.py at startup
_MODELS = {}

def register_models(ensembles):
    global _MODELS
    _MODELS = ensembles

def predict_properties(smi):
    """
    Returns dict of all 7 normalised property scores in [0,1].
    Higher = better for all properties.
    """
    fp = smiles_to_fp(smi).reshape(1,-1)
    props = {}

    try:
        sa_raw = float(_MODELS['sa'].predict(fp)[0])
        props['Synth'] = float(np.clip(1-(sa_raw-1)/9, 0, 1))
    except: props['Synth'] = 0.5

    try:
        props['Biodeg'] = float(np.clip(_MODELS['bio'].predict(fp)[0], 0, 1))
    except: props['Biodeg'] = 0.5

    try:
        props['Tox'] = float(np.clip(1-_MODELS['tox'].predict(fp)[0], 0, 1))
    except: props['Tox'] = 0.5

    try:
        props['Recyclability'] = float(_MODELS['recycl'].predict_proba(fp)[0][1])
    except: props['Recyclability'] = 0.5

    try:
        props['Strength'] = float(np.clip(_MODELS['strength'].predict(fp)[0], 0, 1))
    except: props['Strength'] = 0.5

    try:
        props['Tg'] = float(np.clip(_MODELS['tg'].predict(fp)[0], 0, 1))
    except: props['Tg'] = 0.5

    try:
        props['Solubility'] = float(np.clip(_MODELS['solubility'].predict(fp)[0], 0, 1))
    except: props['Solubility'] = 0.5

    return props


def compute_green_score(candidates, weights=None):
    """
    Score a list of SMILES and return a sorted DataFrame.
    weights: dict with keys matching DEFAULT_WEIGHTS.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    # Normalise weights to sum to 1
    total = sum(weights.values())
    w = {k: v/total for k,v in weights.items()}

    rows = []
    for smi in candidates:
        p  = predict_properties(smi)
        gs = (w['biodeg']     * p['Biodeg']       +
              w['synth']      * p['Synth']         +
              w['tox']        * p['Tox']           +
              w['recycl']     * p['Recyclability'] +
              w['strength']   * p['Strength']      +
              w['tg']         * p['Tg']            +
              w['solubility'] * p['Solubility'])
        rows.append({'SMILES':smi, **p, 'Green_score': float(gs)})

    df = pd.DataFrame(rows).sort_values('Green_score', ascending=False).reset_index(drop=True)
    return df


PROPERTY_COLS = ['Biodeg','Synth','Tox','Recyclability','Strength','Tg','Solubility']
