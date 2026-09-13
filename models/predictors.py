# =============================================================
# models/predictors.py — XGB + RF + GNN ensemble
# =============================================================

import os, pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from config import (save_path, XGB_DEVICE,
                    FORCE_RETRAIN_PREDICTORS, FORCE_RETRAIN_GNN,
                    MODEL_FILES)
from features.featurizer import smiles_to_fp, is_valid
from models.gnn import train_gnn, HAS_PYG


# ── Helpers ───────────────────────────────────────────────────
def _pkl_save(obj, name):
    pickle.dump(obj, open(save_path(f'{name}.pkl'), 'wb'))

def _pkl_load(name):
    return pickle.load(open(save_path(f'{name}.pkl'), 'rb'))

def _all_exist():
    return all(os.path.exists(save_path(f'{m}.pkl')) for m in MODEL_FILES
               if not m.startswith('gnn'))


# ── XGBoost wrappers ──────────────────────────────────────────
def _xgb_reg(**kw):
    params = dict(n_estimators=400, max_depth=6, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  tree_method='hist', device=XGB_DEVICE)
    params.update(kw)
    return xgb.XGBRegressor(**params)

def _xgb_clf(**kw):
    params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  tree_method='hist', device=XGB_DEVICE,
                  eval_metric='logloss')
    params.update(kw)
    return xgb.XGBClassifier(**params)

def _rf_reg(**kw):
    params = dict(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    params.update(kw)
    return RandomForestRegressor(**params)

def _rf_clf(**kw):
    params = dict(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    params.update(kw)
    return RandomForestClassifier(**params)


# ── Ensemble predictor ────────────────────────────────────────
class EnsembleRegressor:
    """Weighted average of XGB, RF, (optionally GNN) predictions."""
    def __init__(self, xgb_m, rf_m, gnn_m=None, w=(0.45, 0.35, 0.20)):
        self.xgb = xgb_m; self.rf = rf_m; self.gnn = gnn_m
        self.w   = w

    def predict(self, X, smiles=None):
        p_xgb = np.clip(self.xgb.predict(X), 0, 1)
        p_rf  = np.clip(self.rf.predict(X),  0, 1)
        if self.gnn is not None and smiles is not None:
            from models.gnn import gnn_predict
            p_gnn = gnn_predict(self.gnn, smiles)
            if p_gnn is not None:
                return self.w[0]*p_xgb + self.w[1]*p_rf + self.w[2]*p_gnn
        return (self.w[0]/(self.w[0]+self.w[1]))*p_xgb + \
               (self.w[1]/(self.w[0]+self.w[1]))*p_rf

class EnsembleClassifier:
    """Soft-vote ensemble of XGB + RF classifiers."""
    def __init__(self, xgb_m, rf_m, gnn_m=None, w=(0.45, 0.35, 0.20)):
        self.xgb = xgb_m; self.rf = rf_m; self.gnn = gnn_m
        self.w   = w

    def predict_proba(self, X, smiles=None):
        p_xgb = self.xgb.predict_proba(X)[:,1]
        p_rf  = self.rf.predict_proba(X)[:,1]
        if self.gnn is not None and smiles is not None:
            from models.gnn import gnn_predict
            p_gnn = gnn_predict(self.gnn, smiles)
            if p_gnn is not None:
                proba = self.w[0]*p_xgb + self.w[1]*p_rf + self.w[2]*p_gnn
                return np.stack([1-proba, proba], axis=1)
        w0 = self.w[0]/(self.w[0]+self.w[1])
        w1 = self.w[1]/(self.w[0]+self.w[1])
        proba = w0*p_xgb + w1*p_rf
        return np.stack([1-proba, proba], axis=1)

    def predict(self, X, smiles=None):
        return (self.predict_proba(X, smiles)[:,1] >= 0.5).astype(int)


# ── Master train / load ───────────────────────────────────────
def load_or_train_models(data):
    """
    Load all models from disk if they exist,
    otherwise train from scratch and save.
    Returns dict of model objects.
    """
    if _all_exist() and not FORCE_RETRAIN_PREDICTORS:
        print('\nLoading all models from disk...')
        models = {n: _pkl_load(n) for n in MODEL_FILES if not n.startswith('gnn')
                  and os.path.exists(save_path(f'{n}.pkl'))}
        print(f'  Loaded {len(models)} sklearn/xgb models.')
        # Load GNN models
        gnn_models = _load_gnn_models(data)
        models.update(gnn_models)
        return _build_ensembles(models)

    print('\nTraining all models from scratch...')
    models = {}
    d = data

    # ── 1. SA Score regressor ────────────────────────────────
    vm = d['df_struct']['SA_score'].notna().values
    Xsa = d['X_struct'][vm]; ysa = d['y_sa'][vm]
    Xtr,Xte,ytr,yte = train_test_split(Xsa, ysa, test_size=0.2, random_state=42)
    xgb_sa = _xgb_reg(n_estimators=300, max_depth=6)
    xgb_sa.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_sa  = _rf_reg(n_estimators=200, max_depth=10)
    rf_sa.fit(Xtr, ytr)
    r2 = r2_score(yte, 0.6*xgb_sa.predict(Xte)+0.4*rf_sa.predict(Xte))
    print(f'[1] SA Score      Ensemble R2: {r2:.4f}')
    models['sa_model_xgb'] = xgb_sa; models['sa_model_rf'] = rf_sa
    _pkl_save(xgb_sa, 'sa_model_xgb'); _pkl_save(rf_sa, 'sa_model_rf')

    # ── 2. Biodegradability regressor ────────────────────────
    Xtr,Xte,ytr,yte = train_test_split(d['X_bio'], d['y_deg'], test_size=0.2, random_state=42)
    xgb_bio = _xgb_reg(n_estimators=500, max_depth=4, learning_rate=0.03)
    xgb_bio.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_bio  = _rf_reg(n_estimators=300)
    rf_bio.fit(Xtr, ytr)
    r2 = r2_score(yte, 0.55*xgb_bio.predict(Xte)+0.45*rf_bio.predict(Xte))
    print(f'[2] Biodeg        Ensemble R2: {r2:.4f}')
    models['bio_model_xgb'] = xgb_bio; models['bio_model_rf'] = rf_bio
    _pkl_save(xgb_bio, 'bio_model_xgb'); _pkl_save(rf_bio, 'bio_model_rf')

    # ── 3. Biodeg classifier ─────────────────────────────────
    y_bin = (d['y_deg'] >= 0.5).astype(int)
    ub,cb = np.unique(y_bin, return_counts=True)
    xgb_bioc = _xgb_clf()
    rf_bioc  = _rf_clf(n_estimators=500, max_depth=4)
    if len(cb)>1 and cb.min()>=2:
        rf_bioc.fit(d['X_bio'], y_bin)
        xgb_bioc.fit(d['X_bio'], y_bin, verbose=False)
    models['bio_clf_xgb'] = xgb_bioc; models['bio_clf_rf'] = rf_bioc
    _pkl_save(xgb_bioc, 'bio_clf_xgb'); _pkl_save(rf_bioc, 'bio_clf_rf')
    print(f'[3] Biodeg Clf    trained')

    # ── 4. Toxicity regressor ────────────────────────────────
    Xtr,Xte,ytr,yte = train_test_split(d['X_tox'], d['y_tox'], test_size=0.2, random_state=42)
    xgb_tox = _xgb_reg(n_estimators=400, max_depth=6)
    xgb_tox.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_tox  = _rf_reg(n_estimators=300)
    rf_tox.fit(Xtr, ytr)
    r2 = r2_score(yte, 0.55*xgb_tox.predict(Xte)+0.45*rf_tox.predict(Xte))
    print(f'[4] Toxicity      Ensemble R2: {r2:.4f}')
    models['tox_model_xgb'] = xgb_tox; models['tox_model_rf'] = rf_tox
    _pkl_save(xgb_tox, 'tox_model_xgb'); _pkl_save(rf_tox, 'tox_model_rf')

    # ── 5. Tox classifier (SR-p53) ───────────────────────────
    df21 = d['df_tox21']
    tox21_sc = 'smiles' if 'smiles' in df21.columns else 'SMILES'
    if 'SR-p53' in df21.columns:
        df_tc  = df21[[tox21_sc,'SR-p53']].dropna()
        vr     = df_tc[tox21_sc].apply(is_valid)
        vm     = vr.apply(lambda x: x[0])
        df_tc  = df_tc[vm].copy()
        Xtc    = np.vstack([smiles_to_fp(s) for s in df_tc[tox21_sc]])
        ytc    = df_tc['SR-p53'].astype(int).values
        uc,cc  = np.unique(ytc, return_counts=True)
        if len(cc)>1 and cc.min()>=6:
            sm = SMOTE(random_state=42, k_neighbors=min(5,cc.min()-1))
            Xtc, ytc = sm.fit_resample(Xtc, ytc)
        Xtr,Xte,ytr,yte = train_test_split(Xtc, ytc, test_size=0.2, random_state=42, stratify=ytc)
        xgb_toxc = _xgb_clf(scale_pos_weight=float(cc[0])/float(cc[1]) if len(cc)>1 else 1.0)
        xgb_toxc.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
        rf_toxc  = _rf_clf(n_estimators=300)
        rf_toxc.fit(Xtr, ytr)
        acc = accuracy_score(yte, xgb_toxc.predict(Xte))
        print(f'[5] Tox Clf       Acc: {acc*100:.2f}%')
        models['tox_clf_xgb'] = xgb_toxc; models['tox_clf_rf'] = rf_toxc
        _pkl_save(xgb_toxc, 'tox_clf_xgb'); _pkl_save(rf_toxc, 'tox_clf_rf')
    else:
        models['tox_clf_xgb'] = xgb_tox; models['tox_clf_rf'] = rf_tox
        print('[5] Tox Clf       SR-p53 not found — using regressor fallback')

    # ── 6. Recyclability classifier ──────────────────────────
    ur,cr = np.unique(d['y_rec'].astype(int), return_counts=True)
    Xtr,Xte,ytr,yte = train_test_split(
        d['X_rec'], d['y_rec'].astype(int), test_size=0.2, random_state=42,
        stratify=d['y_rec'].astype(int) if len(cr)>1 else None)
    xgb_rec = _xgb_clf()
    xgb_rec.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_rec  = _rf_clf(n_estimators=300, max_depth=8, min_samples_leaf=5)
    rf_rec.fit(Xtr, ytr)
    acc = accuracy_score(yte, xgb_rec.predict(Xte))
    print(f'[6] Recyclability Acc: {acc*100:.2f}%')
    models['recycle_model_xgb'] = xgb_rec; models['recycle_model_rf'] = rf_rec
    _pkl_save(xgb_rec, 'recycle_model_xgb'); _pkl_save(rf_rec, 'recycle_model_rf')

    # ── 7. Strength regressor ────────────────────────────────
    Xtr,Xte,ytr,yte = train_test_split(d['X_str'], d['y_str'], test_size=0.2, random_state=42)
    xgb_str = _xgb_reg(n_estimators=300)
    xgb_str.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_str  = _rf_reg(n_estimators=200)
    rf_str.fit(Xtr, ytr)
    r2 = r2_score(yte, 0.55*xgb_str.predict(Xte)+0.45*rf_str.predict(Xte))
    print(f'[7] Strength      Ensemble R2: {r2:.4f}')
    models['strength_model_xgb'] = xgb_str; models['strength_model_rf'] = rf_str
    _pkl_save(xgb_str, 'strength_model_xgb'); _pkl_save(rf_str, 'strength_model_rf')

    # ── 8. Tg regressor ─────────────────────────────────────
    Xtr,Xte,ytr,yte = train_test_split(d['X_tg'], d['y_tg'], test_size=0.2, random_state=42)
    xgb_tg = _xgb_reg(n_estimators=400)
    xgb_tg.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_tg  = _rf_reg(n_estimators=250)
    rf_tg.fit(Xtr, ytr)
    r2 = r2_score(yte, 0.55*xgb_tg.predict(Xte)+0.45*rf_tg.predict(Xte))
    print(f'[8] Tg            Ensemble R2: {r2:.4f}')
    models['tg_model_xgb'] = xgb_tg; models['tg_model_rf'] = rf_tg
    _pkl_save(xgb_tg, 'tg_model_xgb'); _pkl_save(rf_tg, 'tg_model_rf')

    # ── 9. Solubility regressor ──────────────────────────────
    Xtr,Xte,ytr,yte = train_test_split(d['X_sol'], d['y_sol'], test_size=0.2, random_state=42)
    xgb_sol = _xgb_reg(n_estimators=400)
    xgb_sol.fit(Xtr, ytr, eval_set=[(Xte,yte)], verbose=False)
    rf_sol  = _rf_reg(n_estimators=250)
    rf_sol.fit(Xtr, ytr)
    r2 = r2_score(yte, 0.55*xgb_sol.predict(Xte)+0.45*rf_sol.predict(Xte))
    print(f'[9] Solubility    Ensemble R2: {r2:.4f}')
    models['solubility_model_xgb'] = xgb_sol; models['solubility_model_rf'] = rf_sol
    _pkl_save(xgb_sol, 'solubility_model_xgb'); _pkl_save(rf_sol, 'solubility_model_rf')

    # ── GNN models ───────────────────────────────────────────
    gnn_models = _train_gnn_models(data)
    models.update(gnn_models)

    print(f'\nAll models saved to saved_models/')
    return _build_ensembles(models)


def _train_gnn_models(data):
    gnn = {}
    if not HAS_PYG:
        return gnn
    d = data
    print('\nTraining GNN models...')
    gnn['gnn_bio'] = train_gnn(
        d['df_bio']['SMILES'].tolist(), d['y_deg'].tolist(),
        task='regression', model_name='gnn_bio')
    gnn['gnn_tox'] = train_gnn(
        d['df_tox']['SMILES'].tolist(), d['y_tox'].tolist(),
        task='regression', model_name='gnn_tox')
    # Tg & sol GNNs only if enough data
    if len(d['y_tg']) > 50:
        gnn['gnn_tg'] = train_gnn(
            d['df_struct']['SMILES'].sample(len(d['y_tg']), random_state=42).tolist(),
            d['y_tg'].tolist(), task='regression', model_name='gnn_tg')
    if len(d['y_sol']) > 50:
        gnn['gnn_sol'] = train_gnn(
            d['df_struct']['SMILES'].sample(len(d['y_sol']), random_state=42).tolist(),
            d['y_sol'].tolist(), task='regression', model_name='gnn_sol')
    return gnn


def _load_gnn_models(data):
    from models.gnn import train_gnn
    gnn = {}
    if not HAS_PYG: return gnn
    d = data
    for name, smiles, labels in [
        ('gnn_bio', d['df_bio']['SMILES'].tolist(), d['y_deg'].tolist()),
        ('gnn_tox', d['df_tox']['SMILES'].tolist(), d['y_tox'].tolist()),
    ]:
        gnn[name] = train_gnn(smiles, labels, model_name=name)
    return gnn


def _build_ensembles(models):
    """Wrap raw models into EnsembleRegressor / EnsembleClassifier."""
    def _get(name_xgb, name_rf, fallback_xgb=None, fallback_rf=None):
        xm = models.get(name_xgb) or models.get(fallback_xgb)
        rm = models.get(name_rf)  or models.get(fallback_rf)
        return xm, rm

    ensembles = {}

    xm,rm = _get('sa_model_xgb','sa_model_rf')
    if xm and rm:
        ensembles['sa'] = EnsembleRegressor(xm, rm)

    xm,rm = _get('bio_model_xgb','bio_model_rf')
    if xm and rm:
        ensembles['bio'] = EnsembleRegressor(xm, rm, models.get('gnn_bio'))

    xm,rm = _get('bio_clf_xgb','bio_clf_rf')
    if xm and rm:
        ensembles['bio_clf'] = EnsembleClassifier(xm, rm)

    xm,rm = _get('tox_model_xgb','tox_model_rf')
    if xm and rm:
        ensembles['tox'] = EnsembleRegressor(xm, rm, models.get('gnn_tox'))

    xm,rm = _get('tox_clf_xgb','tox_clf_rf')
    if xm and rm:
        ensembles['tox_clf'] = EnsembleClassifier(xm, rm)

    xm,rm = _get('recycle_model_xgb','recycle_model_rf')
    if xm and rm:
        ensembles['recycl'] = EnsembleClassifier(xm, rm)

    xm,rm = _get('strength_model_xgb','strength_model_rf')
    if xm and rm:
        ensembles['strength'] = EnsembleRegressor(xm, rm)

    xm,rm = _get('tg_model_xgb','tg_model_rf')
    if xm and rm:
        ensembles['tg'] = EnsembleRegressor(xm, rm, models.get('gnn_tg'))

    xm,rm = _get('solubility_model_xgb','solubility_model_rf')
    if xm and rm:
        ensembles['solubility'] = EnsembleRegressor(xm, rm, models.get('gnn_sol'))

    print(f'Ensembles built: {list(ensembles.keys())}')
    return ensembles
