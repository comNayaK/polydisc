// =============================================================
// components/CandidatesTable.jsx — Displays top discovery results
// =============================================================

import { addFavourite } from '../api';
import { useState } from 'react';

export default function CandidatesTable({ candidates }) {
  const [faving, setFaving] = useState({});

  if (!candidates || candidates.length === 0) return null;

  const handleFav = async (smiles) => {
    setFaving({ ...faving, [smiles]: true });
    try {
      await addFavourite(smiles);
      // Optional: show toast notification
    } catch (e) {
      console.error('Failed to add favourite', e);
    }
    setFaving({ ...faving, [smiles]: false });
  };

  return (
    <div className="overflow-auto mt-4" style={{ maxHeight: '300px' }}>
      <table className="data-table">
        <thead>
          <tr>
            <th style={{ width: '40px' }}>#</th>
            <th>SMILES</th>
            <th>GS</th>
            <th>Bio</th>
            <th>Synth</th>
            <th>Tox</th>
            <th>Rec</th>
            <th>Str</th>
            <th>Tg</th>
            <th>Sol</th>
            <th style={{ width: '60px' }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c, i) => (
            <tr key={c.smiles}>
              <td className="text-muted">{i + 1}</td>
              <td title={c.smiles}>
                {c.smiles.length > 25 ? c.smiles.substring(0, 25) + '…' : c.smiles}
              </td>
              <td className="text-accent font-bold">{c.green_score.toFixed(4)}</td>
              <td>{c.biodeg.toFixed(3)}</td>
              <td>{c.synth.toFixed(3)}</td>
              <td>{c.tox.toFixed(3)}</td>
              <td>{c.recyclability.toFixed(3)}</td>
              <td>{c.strength.toFixed(3)}</td>
              <td>{c.tg.toFixed(3)}</td>
              <td>{c.solubility.toFixed(3)}</td>
              <td>
                <button
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '4px 8px' }}
                  onClick={() => handleFav(c.smiles)}
                  disabled={faving[c.smiles]}
                  title="Add to Favourites"
                >
                  Save
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
