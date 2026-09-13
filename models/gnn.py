# =============================================================
# models/gnn.py — GNN for property prediction
# =============================================================

import os, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import (DEVICE, GNN_HIDDEN_DIM, GNN_NUM_LAYERS,
                    GNN_DROPOUT, GNN_LR, GNN_EPOCHS,
                    GNN_BATCH_SIZE, GNN_PATIENCE, save_path)
from features.featurizer import ATOM_FEATURES, BOND_FEATURES, smiles_to_graph

# ── Try importing PyG (optional dependency) ───────────────────
try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import DataLoader as GeoDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print('torch_geometric not found — GNN disabled, ensemble will use XGB+RF only')


class PolymerGNN(nn.Module):
    """
    Graph Attention Network for polymer property prediction.
    Combines node-level GAT layers with global pooling and
    an MLP head. Works for both regression and classification.
    """
    def __init__(self, out_dim=1, task='regression',
                 hidden=GNN_HIDDEN_DIM, n_layers=GNN_NUM_LAYERS,
                 dropout=GNN_DROPOUT):
        super().__init__()
        self.task    = task
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(ATOM_FEATURES, hidden)

        # GAT layers
        self.convs = nn.ModuleList([
            GATConv(hidden, hidden // 4, heads=4,
                    dropout=dropout, concat=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(n_layers)
        ])

        # Edge feature projection (used as attention bias)
        self.edge_proj = nn.Linear(BOND_FEATURES, 4)  # 4 = n_heads

        # Readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

        if task == 'regression':
            self.out_act = nn.Sigmoid()   # outputs in [0,1]
        else:
            self.out_act = nn.Identity()  # logits for BCE

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch          = data.batch if hasattr(data,'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device)

        x = F.silu(self.input_proj(x))

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x     = norm(x + x_new)
            x     = F.dropout(x, p=self.dropout, training=self.training)

        # Global readout: mean + max pooling concatenated
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x,  batch)
        x_out  = torch.cat([x_mean, x_max], dim=-1)

        out = self.readout(x_out)
        return self.out_act(out)


def _build_graph_dataset(smiles_list, labels):
    """Build list of PyG Data objects with labels."""
    from torch_geometric.data import Data
    dataset = []
    for smi, y in zip(smiles_list, labels):
        g = smiles_to_graph(smi)
        if g is None: continue
        g.y = torch.tensor([y], dtype=torch.float)
        dataset.append(g)
    return dataset


def train_gnn(smiles_list, labels, task='regression', model_name='gnn'):
    """
    Train or load a GNN model.
    Returns trained PolymerGNN or None if PyG not available.
    """
    if not HAS_PYG:
        return None

    path = save_path(f'{model_name}.pth')
    out_dim = 1

    model = PolymerGNN(out_dim=out_dim, task=task).to(DEVICE)

    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        print(f'  GNN loaded: {model_name} (epoch {ck.get("epoch","?")})')
        return model

    print(f'  Training GNN: {model_name} on {len(smiles_list)} samples...')
    dataset = _build_graph_dataset(smiles_list, labels)
    if len(dataset) < 20:
        print(f'  Too few valid graphs ({len(dataset)}) — skipping GNN')
        return None

    split    = int(0.85 * len(dataset))
    train_ds = dataset[:split]
    val_ds   = dataset[split:]
    train_loader = GeoDataLoader(train_ds, batch_size=GNN_BATCH_SIZE, shuffle=True)
    val_loader   = GeoDataLoader(val_ds,   batch_size=GNN_BATCH_SIZE)

    opt       = torch.optim.Adam(model.parameters(), lr=GNN_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    criterion = nn.MSELoss() if task=='regression' else nn.BCEWithLogitsLoss()

    best_val, patience_cnt = float('inf'), 0

    for epoch in range(GNN_EPOCHS):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            opt.zero_grad()
            pred = model(batch).squeeze()
            loss = criterion(pred, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch    = batch.to(DEVICE)
                pred     = model(batch).squeeze()
                val_loss += criterion(pred, batch.y.squeeze()).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss; patience_cnt = 0
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'best_val': best_val}, path)
        else:
            patience_cnt += 1
            if patience_cnt >= GNN_PATIENCE:
                print(f'    GNN early stop at epoch {epoch+1}')
                break

        if (epoch+1) % 10 == 0:
            print(f'    Epoch {epoch+1:>3} | val_loss={val_loss:.4f}')

    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    print(f'  GNN {model_name} done — best val loss: {best_val:.4f}')
    return model


def gnn_predict(model, smiles_list):
    """Batch-predict with GNN. Returns np.array of predictions."""
    if model is None or not HAS_PYG:
        return None
    from torch_geometric.data import DataLoader as GeoDataLoader
    graphs = []
    idx_map = []   # original index of each valid graph
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            idx_map.append(i)

    preds = np.full(len(smiles_list), 0.5, dtype=np.float32)
    if not graphs: return preds

    loader = GeoDataLoader(graphs, batch_size=128)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out   = model(batch).squeeze().cpu().numpy()
            if out.ndim == 0: out = out.reshape(1)
            results.extend(out.tolist())

    for orig_i, val in zip(idx_map, results):
        preds[orig_i] = float(val)
    return preds
