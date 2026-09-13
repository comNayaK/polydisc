# =============================================================
# data/loader.py — Dataset downloading & preprocessing
# =============================================================

import os, glob, urllib.request, gzip, shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed

from config import data_path, save_path
from features.featurizer import is_valid, smiles_to_fp, filter_df

# ── Helpers ───────────────────────────────────────────────────
def _wget(url, out):
    if not os.path.exists(out):
        try:
            print(f'Downloading {url} to {out}')
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print(f'Download failed: {e}')

def _clone(url, dest):
    if not os.path.exists(dest):
        os.system(f'git clone "{url}" "{dest}"')

# ── PI1M ──────────────────────────────────────────────────────
def load_pi1m():
    dest = data_path('PI1M')
    _clone('https://github.com/RUIMINMA1996/PI1M.git', dest)
    csvs = [f for f in os.listdir(dest) if f.endswith('.csv')]
    pi1m_file = next((f for f in ['PI1M_v2.csv','PI1M.csv'] if f in csvs), None)
    if not pi1m_file:
        pi1m_file = csvs[0] if csvs else ''
    df = pd.read_csv(os.path.join(dest, pi1m_file))
    df.columns = [c.strip().replace('SA Score','SA_score') for c in df.columns]
    if 'smiles' in df.columns: df = df.rename(columns={'smiles':'SMILES'})
    keep = ['SMILES'] + (['SA_score'] if 'SA_score' in df.columns else [])
    df   = df[keep].dropna(subset=['SMILES'])
    if 'SA_score' not in df.columns: df['SA_score'] = np.nan
    print(f'PI1M raw: {len(df):,}')
    return df

# ── Biodegradability ──────────────────────────────────────────
def load_biodeg():
    frames = []

    # tsudalab
    dest = data_path('Polymer-degradability-ranking')
    _clone('https://github.com/tsudalab/Polymer-degradability-ranking.git', dest)
    for fname in ['exp1.xlsx','exp2.xlsx','literature.xlsx']:
        path = os.path.join(dest,'Data',fname)
        if not os.path.exists(path): continue
        try:
            df = pd.read_excel(path, engine='openpyxl')
            df.columns = [str(c).strip() for c in df.columns]
            sc = next((c for c in df.columns if 'smiles' in c.lower()), None)
            dc = next((c for c in df.columns if 'degrad' in c.lower()), None)
            if sc and dc:
                tmp = df[[sc,dc]].rename(columns={sc:'SMILES',dc:'Degradability'}).dropna()
                tmp['Degradability'] = pd.to_numeric(tmp['Degradability'], errors='coerce')
                frames.append(tmp.dropna())
                print(f'  tsudalab {fname}: {len(tmp)} rows')
        except Exception as e: print(f'  tsudalab {fname} error: {e}')

    # DeepChem biodeg
    out = data_path('biodeg_deepchem.csv')
    _wget('https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/biodeg.csv', out)
    if os.path.exists(out) and os.path.getsize(out) > 500:
        try:
            df = pd.read_csv(out)
            df.columns = [c.strip() for c in df.columns]
            sc = next((c for c in df.columns if 'smiles' in c.lower()), None)
            dc = next((c for c in df.columns if any(k in c.lower() for k in ['biodeg','label','activity','class'])), None)
            if sc and dc:
                tmp = df[[sc,dc]].dropna()
                tmp.columns = ['SMILES','Degradability']
                tmp['Degradability'] = pd.to_numeric(tmp['Degradability'], errors='coerce') * 100
                frames.append(tmp.dropna())
                print(f'  DeepChem biodeg: {len(tmp)} rows')
        except Exception as e: print(f'  DeepChem error: {e}')

    # Curated ground truth
    KNOWN = [
        ('OC(=O)CCCCC(=O)OCCO',95.0),('OC(=O)CC(=O)OCCO',92.0),
        ('OC(=O)CCCC(=O)OCCO',90.0), ('CC(O)C(=O)O',98.0),
        ('OCC(O)C(O)C(O)C(O)CO',97.0),('OC(=O)c1ccc(O)cc1',45.0),
        ('NCCCC(=O)O',88.0),          ('OC(=O)CCCCCCC(=O)OCCO',85.0),
        ('O=C1CCCCO1',78.0),          ('OC(=O)CC(N)=O',91.0),
        ('O=C(O)c1ccccc1C(=O)OCCO',35.0),('CC(=O)Oc1ccc(cc1)OC(C)=O',12.0),
        ('c1ccc2ccccc2c1',2.0),        ('ClC(Cl)(Cl)Cl',1.0),
        ('FC(F)(F)C(F)(F)F',0.5),      ('C(C)(C)(C)C',3.0),
        ('c1ccc(Cl)cc1',5.0),          ('CC(C)CC(C)CC(C)C',4.0),
        ('c1ccc(cc1)c1ccccc1',3.0),    ('ClCCCl',8.0),
        ('CCCCCC(=O)O',55.0),          ('CCCC(=O)O',72.0),
        ('OC(=O)CCC(=O)O',82.0),       ('OC(=O)CCCC(=O)O',78.0),
        ('OC(=O)CCCCC(=O)O',70.0),     ('CCCCCCCC(=O)O',48.0),
        ('NC(CCCC[NH3+])C(=O)O',85.0), ('OC(=O)Cc1ccccc1',42.0),
        ('OCC1OC(O)C(O)C(O)C1O',96.0), ('OC(=O)C(O)C(O)C(=O)O',93.0),
    ]
    frames.append(pd.DataFrame(KNOWN, columns=['SMILES','Degradability']))

    df_bio = pd.concat(frames, ignore_index=True).drop_duplicates(subset='SMILES')
    df_bio['Degradability'] = pd.to_numeric(df_bio['Degradability'], errors='coerce')
    df_bio = df_bio.dropna(subset=['Degradability'])
    print(f'Biodegradability combined: {len(df_bio):,}')
    return df_bio

# ── Toxicity ──────────────────────────────────────────────────
def load_toxicity():
    tox21_path = data_path('tox21.csv')
    if not os.path.exists(tox21_path):
        gz_path = f"{tox21_path}.gz"
        _wget('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz', gz_path)
        if os.path.exists(gz_path):
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tox21_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
    df21 = pd.read_csv(tox21_path)
    print(f'Tox21: {df21.shape}')

    tc_path = data_path('toxcast_data.csv')
    if not os.path.exists(tc_path):
        gz_path = f"{tc_path}.gz"
        _wget('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz', gz_path)
        if os.path.exists(gz_path):
            with gzip.open(gz_path, 'rb') as f_in:
                with open(tc_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gz_path)
    have_tc = os.path.exists(tc_path) and os.path.getsize(tc_path) > 1000
    dftc    = pd.read_csv(tc_path) if have_tc else pd.DataFrame()
    if have_tc: print(f'ToxCast: {dftc.shape}')

    def process(df):
        sc = 'smiles' if 'smiles' in df.columns else 'SMILES'
        if sc not in df.columns: return pd.DataFrame()
        tgt = [c for c in df.columns if c not in [sc,'mol_id','ID','smiles','SMILES']
               and pd.api.types.is_numeric_dtype(df[c])]
        if not tgt: return pd.DataFrame()
        df2 = df[[sc]+tgt].copy()
        df2['tox_mean'] = df2[tgt].clip(0,1).mean(axis=1)
        return df2[[sc,'tox_mean']].rename(columns={sc:'SMILES'}).dropna()

    parts = [process(df21)]
    if have_tc and len(dftc): parts.append(process(dftc))
    combined = pd.concat([p for p in parts if len(p)], ignore_index=True).drop_duplicates('SMILES')
    print(f'Toxicity combined: {len(combined):,}')
    return combined, df21

# ── Glass Transition Temperature (Tg) ────────────────────────
def load_tg():
    """
    Tg data from polyVERSE (thermal decomposition / glass transition cols)
    + synthetic proxy for polymers without real data.
    """
    frames = []
    pv_dir = data_path('polyVERSE')
    if os.path.exists(pv_dir):
        TG_KEYS = ['glass transition','tg ','tg,','glass temp']
        for fpath in glob.glob(os.path.join(pv_dir,'**','*.csv'), recursive=True):
            try:
                df = pd.read_csv(fpath, on_bad_lines='skip')
                df.columns = [str(c).strip().lower() for c in df.columns]
                sc = next((c for c in df.columns if 'smiles' in c), None)
                if sc is None: continue
                for pc in df.columns:
                    if any(k in pc for k in TG_KEYS):
                        tmp = df[[sc,pc]].dropna()
                        tmp.columns = ['SMILES','Tg']
                        tmp['Tg'] = pd.to_numeric(tmp['Tg'], errors='coerce')
                        tmp = tmp.dropna()
                        if len(tmp) >= 5:
                            frames.append(tmp)
            except: pass

    if frames:
        df_tg = pd.concat(frames, ignore_index=True).drop_duplicates('SMILES')
        print(f'Tg real data: {len(df_tg):,}')
        return df_tg, True
    else:
        print('No real Tg data — will use proxy labels')
        return pd.DataFrame(), False

# ── Solubility ────────────────────────────────────────────────
def load_solubility():
    """ESOL dataset from DeepChem for logS solubility."""
    out = data_path('delaney-processed.csv')
    _wget('https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv', out)
    if os.path.exists(out) and os.path.getsize(out) > 500:
        try:
            df = pd.read_csv(out)
            df.columns = [c.strip() for c in df.columns]
            sc  = next((c for c in df.columns if 'smiles' in c.lower()), None)
            sol = next((c for c in df.columns if 'solubility' in c.lower() or 'logs' in c.lower()), None)
            if sc and sol:
                tmp = df[[sc,sol]].rename(columns={sc:'SMILES',sol:'LogS'}).dropna()
                tmp['LogS'] = pd.to_numeric(tmp['LogS'], errors='coerce')
                tmp = tmp.dropna()
                print(f'Solubility (ESOL): {len(tmp):,}')
                return tmp, True
        except Exception as e: print(f'Solubility load error: {e}')
    print('No solubility data — will use proxy labels')
    return pd.DataFrame(), False

# ── polyVERSE strength & recyclability ───────────────────────
def load_polyverse():
    pv_dir = data_path('polyVERSE')
    _clone('https://github.com/Ramprasad-Group/polyVERSE.git', pv_dir)

    STR_KEYS  = ['modulus','tensile','young','elastic','strength']
    REC_KEYS  = ['decomp','recycl','crystallin','tg','glass']
    str_frames, rec_frames = [], []

    for fpath in glob.glob(os.path.join(pv_dir,'**','*.csv'), recursive=True):
        try:
            df = pd.read_csv(fpath, on_bad_lines='skip')
            df.columns = [str(c).strip().lower() for c in df.columns]
            sc = next((c for c in df.columns if 'smiles' in c), None)
            if sc is None: continue
            for pc in [c for c in df.columns if c != sc]:
                tmp = df[[sc,pc]].dropna()
                tmp.columns = ['SMILES','val']
                tmp['val'] = pd.to_numeric(tmp['val'], errors='coerce')
                tmp = tmp.dropna()
                if len(tmp) < 5: continue
                if any(k in pc for k in STR_KEYS):
                    str_frames.append(tmp.rename(columns={'val':'Strength'}))
                if any(k in pc for k in REC_KEYS):
                    rec_frames.append(tmp.rename(columns={'val':'RecyclScore'}))
        except: pass

    print(f'polyVERSE — strength: {len(str_frames)} | recycl: {len(rec_frames)}')
    return str_frames, rec_frames

# ── Master load function ──────────────────────────────────────
def load_all_data():
    """
    Returns a dict of all cleaned dataframes and feature matrices
    ready for model training.
    """
    from sklearn.preprocessing import MinMaxScaler

    print('\n[1/6] Loading PI1M...')
    df_pi1m = load_pi1m()
    df_pi1m_s = df_pi1m.sample(n=min(50_000, len(df_pi1m)), random_state=42)
    df_struct = filter_df(df_pi1m_s, 'PI1M')
    df_struct = df_struct[['SMILES','SA_score']].drop_duplicates('SMILES').reset_index(drop=True)

    print('\n[2/6] Loading biodegradability...')
    df_bio_raw = load_biodeg()
    df_bio = filter_df(df_bio_raw, 'Biodegradability')
    scaler_deg = MinMaxScaler()
    df_bio['Degradability_norm'] = scaler_deg.fit_transform(
        df_bio['Degradability'].values.reshape(-1,1)).flatten()

    print('\n[3/6] Loading toxicity...')
    df_tox_raw, df_tox21 = load_toxicity()
    tox_res  = df_tox_raw['SMILES'].apply(is_valid)
    tox_mask = tox_res.apply(lambda x: x[0])
    df_tox   = df_tox_raw[tox_mask].copy()
    df_tox['SMILES'] = tox_res[tox_mask].apply(lambda x: x[1]).values
    print(f'  Toxicity valid: {len(df_tox):,}')

    print('\n[4/6] Loading Tg & Solubility...')
    df_tg,  have_tg  = load_tg()
    df_sol, have_sol = load_solubility()

    print('\n[5/6] Loading polyVERSE...')
    str_frames, rec_frames = load_polyverse()

    print('\n[6/6] Building fingerprints...')
    X_struct = np.vstack(Parallel(n_jobs=-1)(
        delayed(smiles_to_fp)(s) for s in df_struct['SMILES']))
    X_bio    = np.vstack(Parallel(n_jobs=-1)(
        delayed(smiles_to_fp)(s) for s in df_bio['SMILES']))
    X_tox    = np.vstack(Parallel(n_jobs=-1)(
        delayed(smiles_to_fp)(s) for s in df_tox['SMILES']))

    y_sa  = df_struct['SA_score'].fillna(df_struct['SA_score'].median()).values
    y_deg = df_bio['Degradability_norm'].values
    y_tox = df_tox['tox_mean'].values

    # Tg features & labels
    X_tg = y_tg = None
    if have_tg and len(df_tg) > 10:
        df_tg_c = filter_df(df_tg, 'Tg')
        if len(df_tg_c) > 10:
            X_tg = np.vstack(Parallel(n_jobs=-1)(
                delayed(smiles_to_fp)(s) for s in df_tg_c['SMILES']))
            scaler_tg = MinMaxScaler()
            y_tg = scaler_tg.fit_transform(
                df_tg_c['Tg'].values.reshape(-1,1)).flatten().astype(np.float32)

    # Solubility features & labels
    X_sol = y_sol = None
    if have_sol and len(df_sol) > 10:
        df_sol_c = filter_df(df_sol, 'Solubility')
        if len(df_sol_c) > 10:
            X_sol = np.vstack(Parallel(n_jobs=-1)(
                delayed(smiles_to_fp)(s) for s in df_sol_c['SMILES']))
            scaler_sol = MinMaxScaler()
            y_sol = scaler_sol.fit_transform(
                df_sol_c['LogS'].values.reshape(-1,1)).flatten().astype(np.float32)

    # Proxy labels for strength & recyclability
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    N_PROXY   = 3000
    sub_smi   = df_struct['SMILES'].sample(N_PROXY, random_state=42).tolist()
    _e1 = Chem.MolFromSmarts('[#6]C(=O)O')
    _am = Chem.MolFromSmarts('[#6]C(=O)N')
    _et = Chem.MolFromSmarts('[#6]O[#6]')
    mwl,aroml,rotl,brl,esl,aml,etl,fpl = [],[],[],[],[],[],[],[]
    for smi in sub_smi:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: raise ValueError
            n_at = mol.GetNumHeavyAtoms()
            mwl.append(Descriptors.MolWt(mol))
            aroml.append(rdMolDescriptors.CalcNumAromaticRings(mol))
            rotl.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
            brl.append(sum(1 for a in mol.GetAtoms() if a.GetDegree()>2)/max(n_at,1))
            esl.append(int(mol.HasSubstructMatch(_e1)))
            aml.append(int(mol.HasSubstructMatch(_am)))
            etl.append(int(mol.HasSubstructMatch(_et)))
            fpl.append(smiles_to_fp(smi))
        except:
            mwl.append(200); aroml.append(0); rotl.append(3); brl.append(0.2)
            esl.append(0);   aml.append(0);   etl.append(0)
            fpl.append(np.zeros(2048, dtype=np.float32))

    mw_a   = np.array(mwl);   ar_a = np.array(aroml)
    rot_a  = np.array(rotl);  br_a = np.array(brl)
    est_a  = np.array(esl);   am_a = np.array(aml)
    et_a   = np.array(etl);   FP   = np.vstack(fpl)

    y_str = np.clip(
        0.4*np.clip(ar_a/5.0,0,1) +
        0.3*(1-np.clip(rot_a/20.0,0,1)) +
        0.2*np.clip((mw_a-100)/900,0,1) +
        0.1*(1-br_a), 0, 1).astype(np.float32)

    sc    = (2*est_a + am_a + et_a +
             (mw_a<300).astype(int) + (rot_a>=4).astype(int) -
             2*(ar_a>=3).astype(int) - (mw_a>800).astype(int) -
             (br_a>0.4).astype(int))
    y_rec = (sc >= 2).astype(int)

    # Proxy Tg if no real data
    if X_tg is None:
        y_tg  = np.clip(
            0.5*np.clip(ar_a/5.0,0,1) +
            0.3*np.clip((mw_a-100)/900,0,1) +
            0.2*(1-np.clip(rot_a/20.0,0,1)), 0, 1).astype(np.float32)
        X_tg  = FP

    # Proxy solubility if no real data
    if X_sol is None:
        y_sol = np.clip(
            0.4*(1-np.clip(mw_a/1000,0,1)) +
            0.3*np.clip(et_a.astype(float),0,1) +
            0.3*(1-np.clip(ar_a/5.0,0,1)), 0, 1).astype(np.float32)
        X_sol = FP

    print('\n=== Dataset Summary ===')
    print(f'  VAE corpus    : {len(df_struct):,}')
    print(f'  Biodeg        : {len(df_bio):,}')
    print(f'  Toxicity      : {len(df_tox):,}')
    print(f'  Tg            : {len(y_tg):,} ({"real" if have_tg else "proxy"})')
    print(f'  Solubility    : {len(y_sol):,} ({"real" if have_sol else "proxy"})')
    print(f'  X_struct      : {X_struct.shape}')

    return dict(
        df_struct=df_struct, df_bio=df_bio, df_tox=df_tox, df_tox21=df_tox21,
        X_struct=X_struct, X_bio=X_bio, X_tox=X_tox,
        X_str=FP, X_rec=FP, X_tg=X_tg, X_sol=X_sol,
        y_sa=y_sa, y_deg=y_deg, y_tox=y_tox,
        y_str=y_str, y_rec=y_rec, y_tg=y_tg, y_sol=y_sol,
        train_set=set(df_struct['SMILES'].tolist()),
    )
