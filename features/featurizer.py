# =============================================================
# features/featurizer.py — SMILES → features
# =============================================================

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from config import FP_RADIUS, FP_BITS, DEVICE

# ── Validity ──────────────────────────────────────────────────
def is_valid(smi):
    """Returns (bool, canonical_smiles_or_None)"""
    if not isinstance(smi, str) or not smi.strip(): return False, None
    try:
        clean = smi.strip().replace('[*]','[H]').replace('*','[H]')
        mol   = Chem.MolFromSmiles(clean)
        if mol is None: return False, None
        real  = [a for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
        if len(real) < 3: return False, None
        mw    = Descriptors.MolWt(mol)
        if not (28 <= mw <= 5000): return False, None
        if sum(1 for a in real if a.GetAtomicNum() == 6) < 1: return False, None
        Chem.SanitizeMol(mol)
        canon = Chem.MolToSmiles(mol)
        return (True, canon) if canon else (False, None)
    except: return False, None

# ── Morgan fingerprint ────────────────────────────────────────
def smiles_to_fp(smi, radius=FP_RADIUS, n_bits=FP_BITS):
    try:
        mol = Chem.MolFromSmiles(smi.replace('[*]','[H]').replace('*','[H]'))
        if mol is None: return np.zeros(n_bits, dtype=np.float32)
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits),
            dtype=np.float32)
    except: return np.zeros(n_bits, dtype=np.float32)

# ── Rich polymer descriptors ──────────────────────────────────
def polymer_descriptors(smi):
    try:
        mol   = Chem.MolFromSmiles(smi.replace('[*]','[H]').replace('*','[H]'))
        if mol is None: return np.zeros(20, dtype=np.float32)
        n_at  = mol.GetNumHeavyAtoms()
        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        tpsa  = Descriptors.TPSA(mol)
        hbd   = rdMolDescriptors.CalcNumHBD(mol)
        hba   = rdMolDescriptors.CalcNumHBA(mol)
        rot   = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        arom  = rdMolDescriptors.CalcNumAromaticRings(mol)
        br    = sum(1 for a in mol.GetAtoms() if a.GetDegree()>2) / max(n_at,1)
        ester = int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]C(=O)O')))
        amide = int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]C(=O)N')))
        ether = int(mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]O[#6]')))
        amid2 = int(mol.HasSubstructMatch(Chem.MolFromSmarts('NC(=O)')))
        hal   = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9,17,35,53])
        n_cnt = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        o_cnt = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        s_cnt = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
        frac_csp3 = Descriptors.FractionCSP3(mol)
        qed   = Descriptors.qed(mol) if hasattr(Descriptors,'qed') else 0.5
        return np.array([
            mw, logp, tpsa, hbd, hba, rot, rings, arom,
            br, n_at, ester, amide, ether, amid2, hal,
            n_cnt, o_cnt, s_cnt, frac_csp3, qed
        ], dtype=np.float32)
    except: return np.zeros(20, dtype=np.float32)

# ── Combined feature vector ───────────────────────────────────
def combined_features(smi):
    return np.concatenate([polymer_descriptors(smi), smiles_to_fp(smi)])

# ── Graph features for GNN ────────────────────────────────────
ATOM_FEATURES = 74   # total atom feature dim

def atom_features(atom):
    from rdkit.Chem import rdchem
    def one_hot(val, choices):
        return [int(val == c) for c in choices]

    feat = (
        one_hot(atom.GetAtomicNum(), [6,7,8,9,15,16,17,35,53,0]) +
        one_hot(atom.GetDegree(), [0,1,2,3,4,5]) +
        one_hot(atom.GetFormalCharge(), [-2,-1,0,1,2]) +
        one_hot(int(atom.GetHybridization()), [
            int(rdchem.HybridizationType.SP),
            int(rdchem.HybridizationType.SP2),
            int(rdchem.HybridizationType.SP3),
            int(rdchem.HybridizationType.SP3D),
            int(rdchem.HybridizationType.SP3D2),
        ]) +
        [int(atom.GetIsAromatic())] +
        one_hot(atom.GetTotalNumHs(), [0,1,2,3,4]) +
        [atom.GetMass() / 100.0,
         atom.GetTotalValence() / 6.0,
         int(atom.IsInRing())]
    )
    # pad / trim to ATOM_FEATURES
    if len(feat) < ATOM_FEATURES:
        feat += [0] * (ATOM_FEATURES - len(feat))
    return feat[:ATOM_FEATURES]

def bond_features(bond):
    from rdkit.Chem import rdchem
    bt = bond.GetBondTypeAsDouble()
    return [
        int(bt == 1.0), int(bt == 1.5),
        int(bt == 2.0), int(bt == 3.0),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]

BOND_FEATURES = 6

def smiles_to_graph(smi):
    """Convert SMILES to PyTorch Geometric Data object."""
    try:
        from torch_geometric.data import Data
        mol = Chem.MolFromSmiles(smi.replace('[*]','[H]').replace('*','[H]'))
        if mol is None: return None

        x = torch.tensor(
            [atom_features(a) for a in mol.GetAtoms()],
            dtype=torch.float)

        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf    = bond_features(bond)
            edge_index += [[i, j], [j, i]]
            edge_attr  += [bf, bf]

        if not edge_index:
            edge_index = torch.zeros((2,0), dtype=torch.long)
            edge_attr  = torch.zeros((0, BOND_FEATURES), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except: return None

def filter_df(df, label, smiles_col='SMILES'):
    from joblib import Parallel, delayed
    res  = df[smiles_col].apply(is_valid)
    mask = res.apply(lambda x: x[0])
    out  = df[mask].copy()
    out[smiles_col] = res[mask].apply(lambda x: x[1]).values
    print(f'{label}: {mask.sum():,}/{len(df):,} valid ({mask.mean()*100:.1f}%)')
    return out.reset_index(drop=True)
