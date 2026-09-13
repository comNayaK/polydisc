# =============================================================
# generation/generator.py — SMILES generation via VAE
# =============================================================

import torch
import torch.nn.functional as F
import numpy as np

from config import DEVICE, VAE_MAX_LEN, VAE_LATENT_DIM
from features.featurizer import is_valid
from models.vae import smiles_to_tensor, decode_tensor, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN

# Global VAE state — set at startup
_VAE = None
_CHAR_TO_IDX = None
_IDX_TO_CHAR = None

def register_vae(vae, char_to_idx, idx_to_char):
    global _VAE, _CHAR_TO_IDX, _IDX_TO_CHAR
    _VAE = vae
    _CHAR_TO_IDX = char_to_idx
    _IDX_TO_CHAR = idx_to_char


def _autoregressive_decode(z, model, temp=1.0, max_len=VAE_MAX_LEN):
    """Shared decoder used by both generate and perturb."""
    bs = z.size(0)
    h0 = model.z_to_h(z).unsqueeze(0).repeat(2,1,1)
    seq = torch.full((bs,1), SOS_TOKEN, dtype=torch.long, device=DEVICE)

    for _ in range(max_len-1):
        emb   = model.embed(seq[:,-1:])
        z_exp = z.unsqueeze(1)
        inp   = torch.cat([emb, z_exp], dim=-1)
        out, h0 = model.decoder(inp, h0)
        logits   = model.fc_out(out[:,0,:]) / max(float(temp), 0.1)
        probs    = F.softmax(logits, dim=-1)
        nxt      = torch.multinomial(probs, 1)
        seq      = torch.cat([seq, nxt], dim=1)
        if (nxt.squeeze(-1) == EOS_TOKEN).all(): break

    return seq


def generate_smiles(n=500, temp=1.0):
    """
    Sample n SMILES from the VAE prior.
    Returns list of raw decoded strings (not validated).
    """
    _VAE.eval()
    results = []
    with torch.no_grad():
        for start in range(0, n, 64):
            bs  = min(64, n-len(results))
            if bs <= 0: break
            z   = torch.randn(bs, VAE_LATENT_DIM, device=DEVICE) * temp
            seq = _autoregressive_decode(z, _VAE, temp=temp)
            for i in range(bs):
                results.append(decode_tensor(seq[i][1:], _IDX_TO_CHAR))
    return results


def perturb_smiles(smi, n=200, noise=0.3):
    """
    Encode a SMILES to latent space, add noise, decode.
    Returns list of raw decoded strings.
    """
    ok, canon = is_valid(smi)
    if not ok: return []

    t = smiles_to_tensor([canon], _CHAR_TO_IDX)
    _VAE.eval()
    results = []
    with torch.no_grad():
        mu, lv = _VAE.encode(t)
        for start in range(0, n, 64):
            bs  = min(64, n-len(results))
            if bs <= 0: break
            z   = (mu.expand(bs,-1) +
                   torch.exp(0.5*lv).expand(bs,-1) *
                   torch.randn(bs, VAE_LATENT_DIM, device=DEVICE) * noise)
            seq = _autoregressive_decode(z, _VAE, temp=noise+0.7)
            for i in range(bs):
                results.append(decode_tensor(seq[i][1:], _IDX_TO_CHAR))
    return results


def interpolate_smiles(smi_a, smi_b, steps=8):
    """
    Linearly interpolate between two SMILES in latent space.
    Returns list of decoded strings along the path.
    """
    ok_a, ca = is_valid(smi_a)
    ok_b, cb = is_valid(smi_b)
    if not ok_a or not ok_b: return []

    ta = smiles_to_tensor([ca], _CHAR_TO_IDX)
    tb = smiles_to_tensor([cb], _CHAR_TO_IDX)
    _VAE.eval()
    results = []
    with torch.no_grad():
        mu_a, _ = _VAE.encode(ta)
        mu_b, _ = _VAE.encode(tb)
        for alpha in np.linspace(0, 1, steps):
            z   = (1-alpha)*mu_a + alpha*mu_b
            seq = _autoregressive_decode(z, _VAE, temp=0.8)
            results.append(decode_tensor(seq[0][1:], _IDX_TO_CHAR))
    return results
