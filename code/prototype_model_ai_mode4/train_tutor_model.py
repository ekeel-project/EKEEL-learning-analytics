#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -----------------------
# CONFIG
# -----------------------
COLS = ["Timestamp", "Action", "Field1", "Field2", "Field3", "Field4", "Field5"]

# -----------------------
# Normalizzazione token / concept (MUST MATCH training+inference)
# -----------------------
def norm_token(x: str) -> str:
    """
    Normalizza token tipo:
      - concept_On-Premises -> concept_on_premises
      - concept_IaaS -> concept_iaas
      - concept_SaaS -> concept_saas
    """
    x = str(x or "").strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"[^a-z0-9_]", "", x)
    return x

# -----------------------
# CONCEPT -> ANN mapping (RAW) + normalized (used in model/inference)
# -----------------------
concept_to_annotation = {
    "concept_cloud_computing": "ann1",
    "concept_servizio": "ann4",
    "concept_software": "ann5",
    "concept_risorse_hardware": "ann6",
    "concept_fornitore_del_servizio": "ann7",
    "concept_infrastruttura_cloud": "ann8",
    "concept_livello_astratto": "ann10",
    "concept_livello_fisico": "ann9",
    "concept_modello_di_cloud": "ann11",
    "concept_SaaS": "ann12",
    "concept_PaaS": "ann13",
    "concept_IaaS": "ann14",
    "concept_On-Premises": "ann15",
    "concept_caratteristica": "ann17"
}

# Versione normalizzata coerente con i log (questa è quella da usare in inferenza)
CONCEPT_TO_ANN = {norm_token(k): str(v).strip() for k, v in concept_to_annotation.items()}

# -----------------------
# ConceptSolve: concept può stare in Field1 o Field2 (robusto)
# -----------------------
def get_conceptsolve_concept(row: pd.Series) -> str:
    """
    Nei tuoi log tipici:
      ConceptSolve, concept_id, start, end, solve_time, attempts
      -> concept_id in Field1
    Ma supportiamo fallback su Field2.
    """
    c1 = norm_token(row.get("Field1", ""))
    c2 = norm_token(row.get("Field2", ""))
    if c1.startswith("concept_") and c1 != "concept_":
        return c1
    if c1:
        return c1
    return c2

# -----------------------
# Timestamp parsing (robusto per: 2025-6-12-9:29:22:289)
# -----------------------
def parse_timestamp_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("").str.strip()
    # HH:MM:SS:ms -> HH:MM:SS.ms (se ms è alla fine)
    s2 = s.str.replace(r"(\d{1,2}:\d{1,2}:\d{1,2}):(\d{1,6})$", r"\1.\2", regex=True)
    ts = pd.to_datetime(s2, errors="coerce")
    # formato con trattini: YYYY-M-D-HH:MM:SS:fff...
    ts2 = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d-%H:%M:%S:%f")
    return ts.fillna(ts2)

def read_log_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=COLS,
        sep=",",
        engine="python",
        na_values=["", "null", "NULL", "NaN", "nan", "None"]
    )

    df["Timestamp"] = parse_timestamp_series(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # pulizia base stringhe
    for c in ["Action", "Field1", "Field2", "Field3", "Field4", "Field5"]:
        df[c] = df[c].astype("string").fillna("").str.strip()

    # ActionID per Touch/Grab/Release: Field2 (concept_...)
    df["ActionID"] = ""
    m = df["Action"].isin(["Touch", "Grab", "Release"])
    df.loc[m, "ActionID"] = df.loc[m, "Field2"]

    # Direction (L/R) per Touch/Grab/Release in Field3 (non usata dal modello)
    df["Direction"] = ""
    df.loc[m, "Direction"] = df.loc[m, "Field3"]

    # -----------------------
    # NORMALIZZAZIONE coerente con training/inferenza
    # -----------------------
    df["ActionID"] = df["ActionID"].map(norm_token)
    df["Field1"] = df["Field1"].map(norm_token)
    df["Field2"] = df["Field2"].map(norm_token)

    # Field3: può essere L/R o True/False -> lower
    df["Field3"] = df["Field3"].astype("string").fillna("").str.strip().str.lower()

    return df

# -----------------------
# Score
# -----------------------
def get_final_score_percent(df: pd.DataFrame) -> float:
    ss = df[df["Action"] == "ScoreSummary"]
    if ss.empty:
        return np.nan
    last = ss.iloc[-1]
    val = pd.to_numeric(last["Field4"], errors="coerce")
    return float(val) if pd.notna(val) else np.nan

def get_current_score_percent(df_upto: pd.DataFrame) -> float:
    """
    Ultimo ScoreSummary osservato FINO A QUI.
    Informazione disponibile online: NON è leakage.
    """
    ss = df_upto[df_upto["Action"] == "ScoreSummary"]
    if ss.empty:
        return np.nan
    last = ss.iloc[-1]
    val = pd.to_numeric(last["Field4"], errors="coerce")
    return float(val) if pd.notna(val) else np.nan

# -----------------------
# Helpers
# -----------------------
def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def _max_consecutive(actions, value: str) -> int:
    max_run = 0
    run = 0
    for a in actions:
        if a == value:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)

def _idle_gaps(df: pd.DataFrame, thr_sec: float = 60.0) -> int:
    ts = df["Timestamp"].values.astype("datetime64[ns]").astype(np.int64) / 1e9
    if ts.size < 2:
        return 0
    gaps = np.diff(ts)
    return int(np.sum(gaps > thr_sec))

def _entropy_after_action(actions, action_name: str) -> float:
    nxt = []
    for i, a in enumerate(actions[:-1]):
        if a == action_name:
            nxt.append(actions[i + 1])
    if not nxt:
        return 0.0
    cnt = Counter(nxt)
    p = np.array(list(cnt.values()), dtype=float)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))

# -----------------------
# Online state features (NO leakage)
# -----------------------
def compute_state_features(df_upto: pd.DataFrame, t0: pd.Timestamp) -> dict:
    if df_upto.empty:
        return {}

    t_now = df_upto["Timestamp"].iloc[-1]
    elapsed = max(float((t_now - t0).total_seconds()), 0.0)

    current_score = get_current_score_percent(df_upto)

    counts = df_upto["Action"].value_counts().to_dict()
    n_touch = int(counts.get("Touch", 0))
    n_play  = int(counts.get("PlayAnnotation", 0))
    n_pause = int(counts.get("Pause", 0))
    n_tp    = int(counts.get("Teleport", 0))
    n_grab  = int(counts.get("Grab", 0))
    n_rel   = int(counts.get("Release", 0))

    # Touch -> next PlayAnnotation deltas
    touch_ts = df_upto.loc[df_upto["Action"] == "Touch", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    play_ts  = df_upto.loc[df_upto["Action"] == "PlayAnnotation", "Timestamp"].to_numpy(dtype="datetime64[ns]")

    t2p_deltas = np.array([], dtype=float)
    touch_to_play_count = 0
    if touch_ts.size and play_ts.size:
        idx = np.searchsorted(play_ts, touch_ts, side="left")
        valid = idx < play_ts.size
        if np.any(valid):
            deltas = (play_ts[idx[valid]] - touch_ts[valid]).astype("timedelta64[ns]").astype(np.int64) / 1e9
            deltas = deltas[deltas >= 0]
            if deltas.size:
                t2p_deltas = deltas
                touch_to_play_count = int(deltas.size)

    touch_to_play_rate = _safe_div(touch_to_play_count, n_touch)
    median_t2p = float(np.median(t2p_deltas)) if t2p_deltas.size else np.nan

    # concetti toccati
    touch_df = df_upto[df_upto["Action"] == "Touch"]
    unique_concepts = int(
        touch_df["ActionID"].astype("string").replace("", pd.NA).dropna().nunique()
    ) if not touch_df.empty else 0
    last_touch = touch_df["ActionID"].iloc[-1] if not touch_df.empty else ""

    # MatchAttempt
    ma = df_upto[df_upto["Action"] == "MatchAttempt"].copy()
    match_total = int(len(ma))
    match_wrong = 0
    if match_total:
        outcome = ma["Field3"].astype(str).str.strip().str.lower()
        wrong_mask = outcome.isin(["false", "0", "no", "f"])
        match_wrong = int(wrong_mask.sum())
    match_acc = _safe_div(match_total - match_wrong, match_total) if match_total else np.nan

    # ConceptSolve + last_solved
    cs = df_upto[df_upto["Action"] == "ConceptSolve"].copy()
    cs_count = int(len(cs))
    last_solved = ""
    if not cs.empty:
        last_solved = get_conceptsolve_concept(cs.iloc[-1])

        solve_time = pd.to_numeric(cs["Field4"], errors="coerce")
        attempts = pd.to_numeric(cs["Field5"], errors="coerce")

        conceptsolve_mean_time = float(solve_time.mean()) if solve_time.notna().any() else np.nan
        conceptsolve_mean_attempts = float(attempts.mean()) if attempts.notna().any() else np.nan
        conceptsolve_first_try = int((attempts == 1).sum())
        conceptsolve_multi_try = int((attempts > 1).sum())
    else:
        conceptsolve_mean_time = np.nan
        conceptsolve_mean_attempts = np.nan
        conceptsolve_first_try = 0
        conceptsolve_multi_try = 0

    minutes = max(elapsed / 60.0, 1e-6)
    pause_per_min = n_pause / minutes
    teleport_per_min = n_tp / minutes

    # map/table context
    count_touch_on_table = int(((df_upto["Action"] == "Touch") & (df_upto["Field1"] == "table")).sum())
    count_touch_on_map   = int(((df_upto["Action"] == "Touch") & (df_upto["Field1"] == "map")).sum())
    count_grab_on_table  = int(((df_upto["Action"] == "Grab")  & (df_upto["Field1"] == "table")).sum())
    count_grab_on_map    = int(((df_upto["Action"] == "Grab")  & (df_upto["Field1"] == "map")).sum())

    actions = df_upto["Action"].astype(str).tolist()
    idle_gaps_gt60 = _idle_gaps(df_upto, thr_sec=60.0)
    max_pause_streak = _max_consecutive(actions, "Pause")
    max_teleport_streak = _max_consecutive(actions, "Teleport")
    entropy_next_after_touch = _entropy_after_action(actions, "Touch")

    return {
        "t_elapsed_sec": elapsed,
        "current_score_percent": current_score,

        "count_Touch": n_touch,
        "count_PlayAnnotation": n_play,
        "count_Pause": n_pause,
        "count_Teleport": n_tp,
        "count_Grab": n_grab,
        "count_Release": n_rel,

        "touch_to_play_rate": touch_to_play_rate,
        "median_touch_to_play_sec": median_t2p,

        "unique_concepts_touched": unique_concepts,

        "match_attempts_total": match_total,
        "match_wrong": match_wrong,
        "match_accuracy": match_acc,

        "conceptsolve_count": cs_count,
        "conceptsolve_mean_time": conceptsolve_mean_time,
        "conceptsolve_mean_attempts": conceptsolve_mean_attempts,
        "conceptsolve_first_try": conceptsolve_first_try,
        "conceptsolve_multi_try": conceptsolve_multi_try,

        "pause_per_min": float(pause_per_min),
        "teleport_per_min": float(teleport_per_min),

        "last_touched_concept": str(last_touch),
        "last_solved_concept": str(last_solved),

        "count_Touch_on_table": count_touch_on_table,
        "count_Touch_on_map": count_touch_on_map,
        "count_Grab_on_table": count_grab_on_table,
        "count_Grab_on_map": count_grab_on_map,
        "idle_gaps_gt60": idle_gaps_gt60,
        "max_pause_streak": max_pause_streak,
        "max_teleport_streak": max_teleport_streak,
        "entropy_next_after_touch": entropy_next_after_touch,
    }

# -----------------------
# Training rows builder
# -----------------------
def build_training_rows(df: pd.DataFrame, session_id: str):
    rows = []

    final_score = get_final_score_percent(df)
    if np.isnan(final_score) or df.empty:
        return rows

    t0 = df["Timestamp"].iloc[0]

    play_idx = df.index[df["Action"] == "PlayAnnotation"].to_numpy()
    if play_idx.size == 0:
        return rows

    for j in play_idx:
        ann_id = str(df.at[j, "Field1"]).strip()
        if not ann_id:
            continue
        if j == 0:
            continue

        df_upto = df.loc[: j - 1].copy()
        state = compute_state_features(df_upto, t0)
        if not state:
            continue

        rows.append({
            "sessionID": session_id,
            "ann_id": ann_id,
            "target_score_percent": float(final_score),
            **state
        })

    return rows

def load_all_sessions(input_dir: str):
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt"))) + sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"Nessun .txt o .csv trovato in {input_dir}")

    sessions = []
    for p in files:
        sid = os.path.splitext(os.path.basename(p))[0]
        df = read_log_file(p)
        if df.empty:
            continue
        sessions.append((sid, df))

    if not sessions:
        raise RuntimeError("Nessuna sessione valida trovata.")
    return sessions

# -----------------------
# MAIN
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Cartella con log .txt/.csv")
    ap.add_argument("--out_model", required=True, help="Path output bundle .joblib (file o cartella)")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="Hold-out per sessioni (validation)")
    args = ap.parse_args()

    sessions = load_all_sessions(args.input_dir)

    all_rows = []
    for sid, df in sessions:
        all_rows.extend(build_training_rows(df, sid))

    data = pd.DataFrame(all_rows)
    if data.empty:
        raise RuntimeError(
            "Dataset vuoto: nessun PlayAnnotation valido trovato nei log.\n"
            "Controlla che PlayAnnotation abbia ann_id in Field1 e che ci sia ScoreSummary finale."
        )

    candidate_anns = sorted(data["ann_id"].astype(str).unique().tolist())

    # Split train/val by session
    sess_ids = data["sessionID"].unique().tolist()
    rng = np.random.default_rng(42)
    rng.shuffle(sess_ids)
    n_val = max(1, int(len(sess_ids) * args.test_ratio))
    val_sess = set(sess_ids[:n_val])

    train_df = data[~data["sessionID"].isin(val_sess)].reset_index(drop=True)
    val_df   = data[data["sessionID"].isin(val_sess)].reset_index(drop=True)

    target = "target_score_percent"
    cat_cols = ["ann_id", "last_touched_concept", "last_solved_concept"]
    drop_cols = ["sessionID", target]
    feature_cols = [c for c in data.columns if c not in drop_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target].astype(float).values
    X_val = val_df[feature_cols].copy()
    y_val = val_df[target].astype(float).values

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.08,
        max_iter=300
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)

    if len(val_df) >= 20:
        pred = pipe.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        print(f"[EVAL] val_sessions={len(val_sess)} val_rows={len(val_df)}  MAE={mae:.3f}  R2={r2:.3f}")
    else:
        print(f"[EVAL] Validation piccola (rows={len(val_df)}), salto metriche affidabili.")

    bundle = {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "candidate_anns": candidate_anns,
        # mappa concept->ann NORMALIZZATA (chiavi coerenti col log e con ANN_ORDER->concept_)
        "concept_to_ann": CONCEPT_TO_ANN,
        "notes": (
            "Tutor model: predicts final score given current state + ann_id (choice). "
            "Includes current_score_percent, normalized concepts, robust ConceptSolve concept extraction, "
            "and concept_to_ann mapping for deterministic next-ann inference."
        )
    }

    # out_model: file o cartella
    out_path = Path(args.out_model)
    if out_path.exists() and out_path.is_dir():
        out_path = out_path / "tutor_model.joblib"
    elif out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "tutor_model.joblib"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, out_path)

    print(f"[OK] Modello salvato in: {out_path.resolve()}")
    print(f"[OK] candidate_anns: {len(candidate_anns)}")
    print(f"[OK] concept_to_ann: {len(CONCEPT_TO_ANN)}")
    print(f"[INFO] Train rows: {len(train_df)}  |  Val rows: {len(val_df)}")

if __name__ == "__main__":
    main()
