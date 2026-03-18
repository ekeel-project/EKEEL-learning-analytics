#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WP4 Learning Analytics — Mode 4 Sequence Modeling (Discrete HMM).

This script provides a lightweight, reproducible implementation of the HMM-based
sequence modeling described in D4.2 for Mode 4 (IKCM).

It expects the per-session processed CSV logs exported by the Mode 4 pipeline:
  Report/ProcessedLogs/<SID>.csv
where each CSV contains at least an 'Action' column.

It also expects the Mode 4 feature table (Mode4Features.csv) to optionally split
sessions into 'high' vs 'low' performance groups using score_percent.

Usage
-----
python mode4_sequence_hmm.py \
  --processed_logs_dir <Report/ProcessedLogs> \
  --features_csv <Report/Mode4Features.csv> \
  --out_dir <Report/Predictive>

Notes
-----
- Uses CategoricalHMM when available; otherwise falls back to MultinomialHMM with one-hot observations.
- The goal is to provide a maintainable software counterpart of the HMM notebook;
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from hmmlearn.hmm import CategoricalHMM  # type: ignore
except Exception:  # pragma: no cover
    CategoricalHMM = None  # type: ignore

from hmmlearn.hmm import MultinomialHMM


def load_sequences(processed_logs_dir: Path) -> dict[str, list[str]]:
    seqs: dict[str, list[str]] = {}
    for fp in sorted(processed_logs_dir.glob('*.csv')):
        sid = fp.stem
        df = pd.read_csv(fp)
        if 'Action' not in df.columns:
            continue
        seq = df['Action'].astype(str).tolist()
        if len(seq) > 0:
            seqs[sid] = seq
    return seqs


def encode_sequences(seqs: dict[str, list[str]]):
    vocab = sorted({a for s in seqs.values() for a in s})
    tok2id = {t: i for i, t in enumerate(vocab)}

    X_list = []
    lengths = []
    sids = []
    for sid, seq in seqs.items():
        obs = np.array([tok2id[a] for a in seq], dtype=int).reshape(-1, 1)
        X_list.append(obs)
        lengths.append(len(obs))
        sids.append(sid)

    X = np.vstack(X_list) if X_list else np.empty((0, 1), dtype=int)
    return X, lengths, vocab, tok2id, sids


def fit_hmm(X: np.ndarray, lengths: list[int], n_states: int = 3, random_state: int = 42):
    """Fit a discrete HMM on an observation sequence.

    Robust across hmmlearn versions:
    - Preferred: `CategoricalHMM` (if available) with integer observations (shape [N, 1]).
    - Fallback: `MultinomialHMM` with one-hot/count vectors (shape [N, V]).

    Parameters
    ----------
    X:
        Encoded observations as integer ids in [0, V-1], shape [N, 1].
    lengths:
        Per-sequence lengths for concatenated sequences.
    """
    if X.size == 0:
        raise ValueError('Empty observation matrix.')

    n_features = int(X.max()) + 1

    if CategoricalHMM is not None:
        model = CategoricalHMM(
            n_components=n_states,
            random_state=random_state,
            n_iter=200,
            verbose=False,
        )
        if hasattr(model, 'n_features'):
            model.n_features = n_features
        model.fit(X, lengths)
        return model

    # Fallback: one-hot for MultinomialHMM
    X_oh = np.zeros((X.shape[0], n_features), dtype=int)
    X_oh[np.arange(X.shape[0]), X.reshape(-1)] = 1

    model = MultinomialHMM(
        n_components=n_states,
        random_state=random_state,
        n_iter=200,
        verbose=False,
    )
    if hasattr(model, 'n_features'):
        model.n_features = n_features
    model.fit(X_oh, lengths)
    model._wp4_onehot = True  # type: ignore[attr-defined]
    return model


def group_sessions_by_score(features_csv: Path, threshold: float | None = None):
    df = pd.read_csv(features_csv)
    if 'score_percent' not in df.columns:
        return {}, {}, None

    df = df[['session_id', 'score_percent']].copy() if 'session_id' in df.columns else df[['score_percent']].copy()
    if 'session_id' not in df.columns:
        # fallback: derive session ids as sequential labels
        df['session_id'] = [f'Userlogs{i+1:02d}_M4' for i in range(len(df))]

    if threshold is None:
        threshold = float(df['score_percent'].median())

    high = set(df.loc[df['score_percent'] >= threshold, 'session_id'].astype(str))
    low = set(df.loc[df['score_percent'] < threshold, 'session_id'].astype(str))
    return high, low, threshold


def export_hmm_report(model, vocab: list[str], out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    trans = pd.DataFrame(model.transmat_, columns=[f'S{j}' for j in range(model.n_components)])
    trans.index = [f'S{i}' for i in range(model.n_components)]
    trans.to_csv(out_dir / f'{prefix}_hmm_transition_matrix.csv')

    # Emission probabilities: rows=states, cols=observations
    emiss = pd.DataFrame(model.emissionprob_, columns=vocab)
    emiss.index = [f'S{i}' for i in range(model.n_components)]
    emiss.to_csv(out_dir / f'{prefix}_hmm_emissions.csv')

    # Simple plot: top emissions per state
    topn = 8
    plt.figure(figsize=(12, 4 * model.n_components))
    for i in range(model.n_components):
        row = emiss.loc[f'S{i}'].sort_values(ascending=False).head(topn)[::-1]
        plt.subplot(model.n_components, 1, i + 1)
        plt.barh(row.index, row.values)
        plt.title(f'{prefix} — State S{i} top-{topn} emissions')
        plt.xlabel('Probability')
    plt.tight_layout()
    plt.savefig(out_dir / f'{prefix}_hmm_top_emissions.png', dpi=200)
    plt.close()


def run(processed_logs_dir: Path, features_csv: Path, out_dir: Path, n_states: int = 3):
    seqs = load_sequences(processed_logs_dir)
    if not seqs:
        raise RuntimeError(f'No sequences found in {processed_logs_dir}.')

    high_set, low_set, thr = group_sessions_by_score(features_csv)

    # Fit on all sessions
    X, lengths, vocab, _, _ = encode_sequences(seqs)
    model_all = fit_hmm(X, lengths, n_states=n_states)
    export_hmm_report(model_all, vocab, out_dir, prefix='all')

    # Fit high vs low (if possible)
    if high_set and low_set:
        seqs_high = {k: v for k, v in seqs.items() if k in high_set}
        seqs_low = {k: v for k, v in seqs.items() if k in low_set}
        if len(seqs_high) >= 3 and len(seqs_low) >= 3:
            Xh, lh, vocab_h, _, _ = encode_sequences(seqs_high)
            Xl, ll, vocab_l, _, _ = encode_sequences(seqs_low)
            # Use same vocabulary as 'all' for comparability
            # Re-encode high/low using global mapping
            global_map = {t: i for i, t in enumerate(vocab)}
            def reenc(ss):
                X_list=[]; lens=[]
                for seq in ss.values():
                    obs=np.array([global_map[a] for a in seq],dtype=int).reshape(-1,1)
                    X_list.append(obs); lens.append(len(obs))
                return np.vstack(X_list), lens
            Xh, lh = reenc(seqs_high)
            Xl, ll = reenc(seqs_low)

            mh = fit_hmm(Xh, lh, n_states=n_states)
            ml = fit_hmm(Xl, ll, n_states=n_states)
            export_hmm_report(mh, vocab, out_dir, prefix='high')
            export_hmm_report(ml, vocab, out_dir, prefix='low')

    # HTML summary
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Mode 4 — HMM Sequence Modeling</title></head>
<body>
<h1>Mode 4 — HMM Sequence Modeling</h1>
<p><b>Processed logs</b>: {processed_logs_dir}</p>
<p><b>Features</b>: {features_csv}</p>
<p><b>n_states</b>: {n_states}</p>
<p><b>High/Low split</b>: median threshold on score_percent = {thr if thr is not None else 'N/A'}</p>
<ul>
  <li>all_hmm_transition_matrix.csv, all_hmm_emissions.csv, all_hmm_top_emissions.png</li>
  <li>(optional) high_* and low_* outputs if the split is available</li>
</ul>
</body></html>"""
    (out_dir / 'hmm_report.html').write_text(html, encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description='WP4 — Mode 4 sequence modeling (HMM)')
    p.add_argument('--processed_logs_dir', type=str, required=True, help='Path to Report/ProcessedLogs')
    p.add_argument('--features_csv', type=str, required=True, help='Path to Report/Mode4Features.csv')
    p.add_argument('--out_dir', type=str, required=True, help='Output folder')
    p.add_argument('--n_states', type=int, default=3, help='Number of hidden states (default: 3)')
    args = p.parse_args()

    run(Path(args.processed_logs_dir), Path(args.features_csv), Path(args.out_dir), n_states=args.n_states)


if __name__ == '__main__':
    main()
