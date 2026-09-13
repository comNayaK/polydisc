# =============================================================
# models/vae.py — Character-level VAE for SMILES generation
# =============================================================

import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (DEVICE, save_path, FORCE_RETRAIN_VAE,
                    VAE_MAX_LEN, VAE_NUM_EPOCHS, VAE_BATCH_SIZE,
                    VAE_LATENT_DIM, VAE_HIDDEN_DIM, VAE_EMBED_DIM,
                    VAE_BETA_MAX, VAE_FREE_BITS, VAE_CYCLE_EPOCHS,
                    VAE_MAX_PATIENCE, VAE_LR)

PAD_TOKEN = 0; SOS_TOKEN = 1; EOS_TOKEN = 2


# ── Tokenizer ─────────────────────────────────────────────────
def build_vocab(smiles_list):
    all_chars = set()
    for smi in smiles_list: all_chars.update(smi)
    char_to_idx = {c: i+3 for i,c in enumerate(sorted(all_chars))}
    char_to_idx.update({'<PAD>':PAD_TOKEN,'<SOS>':SOS_TOKEN,'<EOS>':EOS_TOKEN})
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    return char_to_idx, idx_to_char

def save_vocab(char_to_idx, idx_to_char):
    with open(save_path('vocab.json'), 'w') as f:
        json.dump({'char_to_idx': char_to_idx,
                   'idx_to_char': {str(k):v for k,v in idx_to_char.items()}}, f)

def load_vocab():
    with open(save_path('vocab.json')) as f:
        v = json.load(f)
    c2i = v['char_to_idx']
    i2c = {int(k):v for k,v in v['idx_to_char'].items()}
    return c2i, i2c

def smiles_to_tensor(smiles_list, char_to_idx, max_len=VAE_MAX_LEN):
    out = []
    for smi in smiles_list:
        idxs   = [SOS_TOKEN] + [char_to_idx.get(c, PAD_TOKEN) for c in smi[:max_len-2]] + [EOS_TOKEN]
        padded = idxs + [PAD_TOKEN]*(max_len - len(idxs))
        out.append(padded[:max_len])
    return torch.tensor(out, dtype=torch.long).to(DEVICE)

def decode_tensor(seq, idx_to_char):
    chars = []
    for idx in seq.tolist():
        if idx == EOS_TOKEN: break
        c = idx_to_char.get(idx, '')
        if c not in ['<PAD>','<SOS>','<EOS>','']: chars.append(c)
    return ''.join(chars)


# ── Model ─────────────────────────────────────────────────────
class PolymerVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=VAE_EMBED_DIM,
                 hidden_dim=VAE_HIDDEN_DIM, latent_dim=VAE_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=2,
                               batch_first=True, bidirectional=True, dropout=0.1)
        self.fc_mu     = nn.Linear(hidden_dim*2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim*2, latent_dim)
        nn.init.constant_(self.fc_logvar.bias, -3.0)

        self.z_to_h  = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(embed_dim+latent_dim, hidden_dim,
                               num_layers=2, batch_first=True, dropout=0.1)
        self.fc_out  = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        _, h   = self.encoder(self.embed(x))
        h_top  = torch.cat([h[-2],h[-1]], dim=-1)
        mu     = self.fc_mu(h_top)
        logvar = torch.clamp(self.fc_logvar(h_top), -8.0, 2.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5*logvar)*torch.randn_like(mu)

    def decode(self, z, target):
        h0    = self.z_to_h(z).unsqueeze(0).repeat(2,1,1)
        emb   = self.embed(target)
        z_exp = z.unsqueeze(1).expand(-1, emb.size(1), -1)
        out,_ = self.decoder(torch.cat([emb,z_exp],dim=-1), h0)
        return self.fc_out(out)

    def forward(self, x, x_noisy):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, x_noisy[:,:-1])
        return recon, mu, logvar


# ── Training helpers ──────────────────────────────────────────
def _word_dropout(x, rate=0.15):
    mask = (torch.rand_like(x.float()) < rate) & (x != PAD_TOKEN)
    return x.masked_fill(mask, PAD_TOKEN)

def _get_beta(epoch):
    cycle = epoch % VAE_CYCLE_EPOCHS
    return VAE_BETA_MAX * min(cycle / (VAE_CYCLE_EPOCHS * 0.5), 1.0)

def _compute_kl(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl.mean(0), min=VAE_FREE_BITS)


# ── Load or train ─────────────────────────────────────────────
def load_or_train_vae(df_struct):
    """
    Returns (vae_model, char_to_idx, idx_to_char, train_tensor).
    Loads from disk if available and FORCE_RETRAIN_VAE=False.
    """
    vocab_path = save_path('vocab.json')
    vae_path   = save_path('polymer_vae_best.pth')

    subset_size  = min(50_000, len(df_struct))
    train_smiles = df_struct['SMILES'].sample(subset_size, random_state=42).tolist()

    # Build vocab (always — needed to define model size)
    if os.path.exists(vocab_path) and not FORCE_RETRAIN_VAE:
        char_to_idx, idx_to_char = load_vocab()
        print(f'Vocab loaded: {len(char_to_idx)} tokens')
    else:
        char_to_idx, idx_to_char = build_vocab(train_smiles)
        save_vocab(char_to_idx, idx_to_char)
        print(f'Vocab built: {len(char_to_idx)} tokens')

    vocab_size   = len(char_to_idx)
    train_tensor = smiles_to_tensor(train_smiles, char_to_idx)
    train_tensor = train_tensor[torch.randperm(len(train_tensor))]
    print(f'Train tensor: {train_tensor.shape}')

    vae = PolymerVAE(vocab_size).to(DEVICE)
    print(f'VAE params: {sum(p.numel() for p in vae.parameters()):,}')

    # ── Load ─────────────────────────────────────────────────
    if os.path.exists(vae_path) and not FORCE_RETRAIN_VAE:
        ck = torch.load(vae_path, map_location=DEVICE)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            vae.load_state_dict(ck['model_state_dict'])
            ep = ck.get('epoch','?'); bl = ck.get('best_loss','?')
            print(f'VAE loaded — epochs: {ep} | best loss: {bl:.4f}' if isinstance(bl,float)
                  else f'VAE loaded — epochs: {ep}')
        else:
            vae.load_state_dict(ck)
            print('VAE loaded (old format)')
        vae.eval()
        return vae, char_to_idx, idx_to_char, train_tensor

    # ── Train ─────────────────────────────────────────────────
    print(f'\nTraining VAE for up to {VAE_NUM_EPOCHS} epochs...')
    optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=VAE_NUM_EPOCHS)
    best_loss, patience_cnt = float('inf'), 0

    print(f'{"Ep":>3} | {"Total":>7} | {"Recon":>7} | {"KL/d":>7} | {"Beta":>6} | {"Active":>8}')
    print('-'*55)

    for epoch in range(VAE_NUM_EPOCHS):
        vae.train()
        total_l = recon_t = kl_t = 0.0; n_b = 0
        beta = _get_beta(epoch)
        perm = torch.randperm(len(train_tensor), device=DEVICE)

        for i in range(0, len(train_tensor), VAE_BATCH_SIZE):
            batch = train_tensor[perm[i:i+VAE_BATCH_SIZE]]
            if len(batch) < 16: continue
            optimizer.zero_grad()
            recon, mu, logvar = vae(batch, _word_dropout(batch))
            r_loss = F.cross_entropy(recon.view(-1,vocab_size),
                                      batch[:,1:].reshape(-1),
                                      ignore_index=PAD_TOKEN)
            kl     = _compute_kl(mu, logvar).sum()
            loss   = r_loss + beta*kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            total_l += loss.item(); recon_t += r_loss.item()
            kl_t    += kl.item();   n_b += 1

        scheduler.step()
        avg = total_l / n_b
        vae.eval()
        with torch.no_grad():
            mu_p, lv_p = vae.encode(train_tensor[:512])
            kl_p   = -0.5*(1+lv_p-mu_p.pow(2)-lv_p.exp()).mean(0)
            active = (kl_p > 0.05).sum().item()
        vae.train()

        print(f'{epoch+1:>3}  | {avg:>7.4f} | {recon_t/n_b:>7.4f} | '
              f'{kl_t/n_b/VAE_LATENT_DIM:>7.4f} | {beta:>6.4f} | {active:>4}/{VAE_LATENT_DIM}')

        if avg < best_loss:
            best_loss = avg; patience_cnt = 0
            torch.save({'epoch': epoch+1,
                        'model_state_dict': vae.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss}, vae_path)
        else:
            patience_cnt += 1
            if patience_cnt >= VAE_MAX_PATIENCE:
                print(f'Early stop at epoch {epoch+1}'); break

    ck = torch.load(vae_path, map_location=DEVICE)
    vae.load_state_dict(ck['model_state_dict'])
    vae.eval()
    print(f'\nVAE saved — epochs: {ck["epoch"]} | best loss: {ck["best_loss"]:.4f}')
    return vae, char_to_idx, idx_to_char, train_tensor
