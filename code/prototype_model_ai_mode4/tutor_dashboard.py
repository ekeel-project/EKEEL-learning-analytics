#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# -----------------------
# CONFIG
# -----------------------
COLS = ["Timestamp", "Action", "Field1", "Field2", "Field3", "Field4", "Field5"]

# Ordine dei concetti (curriculum). Il "prossimo" concept viene convertito in ann tramite bundle["concept_to_ann"].
ANN_ORDER = [
    "cloud_computing",
    "servizio",
    "software",
    "risorse_hardware",
    "fornitore_del_servizio",
    "infrastruttura_cloud",
    "livello_astratto",
    "livello_fisico",
    "modello_di_cloud",
    "saas",
    "paas",
    "iaas",
    "on_premises",
    "caratteristica",
]

# -----------------------
# Normalizzazione (DEVE combaciare col training)
# -----------------------
def norm_token(x: str) -> str:
    x = str(x or "").strip().lower()
    x = x.replace("-", "_").replace(" ", "_")
    x = re.sub(r"[^a-z0-9_]", "", x)
    return x

def get_conceptsolve_concept(row: pd.Series) -> str:
    """
    Robust: nei log concept di ConceptSolve può stare in Field1 o Field2.
    Dal tuo esempio tipico è in Field1.
    """
    c1 = norm_token(row.get("Field1", ""))
    c2 = norm_token(row.get("Field2", ""))
    if c1.startswith("concept_") and c1 != "concept_":
        return c1
    if c1:
        return c1
    return c2

# -----------------------
# Timestamp parsing (come training)
# -----------------------
def parse_timestamp_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("").str.strip()
    s2 = s.str.replace(r"(\d{1,2}:\d{1,2}:\d{1,2}):(\d{1,6})$", r"\1.\2", regex=True)
    ts = pd.to_datetime(s2, errors="coerce")
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

    # pulizia base
    for c in ["Action", "Field1", "Field2", "Field3", "Field4", "Field5"]:
        df[c] = df[c].astype("string").fillna("").str.strip()

    # ActionID (Touch/Grab/Release) in Field2
    df["ActionID"] = ""
    m = df["Action"].isin(["Touch", "Grab", "Release"])
    df.loc[m, "ActionID"] = df.loc[m, "Field2"]

    # Direction (debug)
    df["Direction"] = ""
    df.loc[m, "Direction"] = df.loc[m, "Field3"]

    # NORMALIZZAZIONE come training
    df["ActionID"] = df["ActionID"].map(norm_token)
    df["Field1"] = df["Field1"].map(norm_token)
    df["Field2"] = df["Field2"].map(norm_token)
    df["Field3"] = df["Field3"].astype("string").fillna("").str.strip().str.lower()

    return df

def list_log_files(folder: str):
    p = Path(folder)
    if not p.exists() or not p.is_dir():
        return []
    files = list(p.glob("*.txt")) + list(p.glob("*.csv"))
    files = sorted(files, key=lambda x: x.name.lower())
    return [str(f) for f in files]

# -----------------------
# Score corrente (come training)
# -----------------------
def get_current_score_percent(df_upto: pd.DataFrame) -> float:
    ss = df_upto[df_upto["Action"] == "ScoreSummary"]
    if ss.empty:
        return np.nan
    last = ss.iloc[-1]
    val = pd.to_numeric(last["Field4"], errors="coerce")
    return float(val) if pd.notna(val) else np.nan

# -----------------------
# NAV
# -----------------------
def find_first_matchattempt_index(df: pd.DataFrame) -> int:
    idx = df.index[df["Action"] == "MatchAttempt"].to_list()
    return int(idx[0]) if idx else 0

def find_next_matchattempt_index(df: pd.DataFrame, current_i: int) -> int:
    future = df.index[(df.index > current_i) & (df["Action"] == "MatchAttempt")].to_list()
    return int(future[0]) if future else current_i

# -----------------------
# Feature helpers (come training)
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
# State features (COPIA 1:1 del training)
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

    touch_df = df_upto[df_upto["Action"] == "Touch"]
    unique_concepts = int(
        touch_df["ActionID"].astype("string").replace("", pd.NA).dropna().nunique()
    ) if not touch_df.empty else 0
    last_touch = touch_df["ActionID"].iloc[-1] if not touch_df.empty else ""

    ma = df_upto[df_upto["Action"] == "MatchAttempt"].copy()
    match_total = int(len(ma))
    match_wrong = 0
    if match_total:
        outcome = ma["Field3"].astype(str).str.strip().str.lower()
        wrong_mask = outcome.isin(["false", "0", "no", "f"])
        match_wrong = int(wrong_mask.sum())
    match_acc = _safe_div(match_total - match_wrong, match_total) if match_total else np.nan

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
# Deterministic next-ann logic (your requirement)
# -----------------------
def next_concept_by_order(last_solved_concept: str) -> str:
    """
    Ritorna il prossimo concept_... (NORMALIZZATO) secondo ANN_ORDER.
    last_solved_concept è tipo 'concept_servizio' (già normalizzato).
    """
    order = [f"concept_{norm_token(x)}" for x in ANN_ORDER]
    last = norm_token(last_solved_concept)

    if not order:
        return ""

    if not last:
        return order[0]

    if last not in order:
        return order[0]

    i = order.index(last)
    return order[i + 1] if i + 1 < len(order) else ""

def suggest_next_ann(bundle: dict, df_upto: pd.DataFrame) -> dict:
    """
    Usa:
      - ultimo ConceptSolve -> last_concept
      - ANN_ORDER -> next_concept
      - bundle['concept_to_ann'] -> next_ann (ann_id)
    """
    cs = df_upto[df_upto["Action"] == "ConceptSolve"]
    if cs.empty:
        last_concept = ""
    else:
        last_concept = get_conceptsolve_concept(cs.iloc[-1])  # concept_...

    nxt_concept = next_concept_by_order(last_concept)

    c2a = bundle.get("concept_to_ann", {}) or {}
    nxt_ann = c2a.get(norm_token(nxt_concept), "")

    return {
        "last_concept": last_concept,
        "next_concept": nxt_concept,
        "next_ann": nxt_ann
    }

# -----------------------
# Prediction
# -----------------------
def make_feature_row(bundle, state_feats: dict, ann_id: str) -> pd.DataFrame:
    feature_cols = bundle["feature_cols"]
    row = {c: np.nan for c in feature_cols}
    for k, v in state_feats.items():
        if k in row:
            row[k] = v
    row["ann_id"] = ann_id
    return pd.DataFrame([row], columns=feature_cols)

def predict_for_ann(bundle, state_feats: dict, ann_id: str) -> float:
    X = make_feature_row(bundle, state_feats, ann_id)
    return float(bundle["pipeline"].predict(X)[0])

# -----------------------
# STREAMLIT APP
# -----------------------
st.set_page_config(page_title="Tutor Dashboard", layout="wide")
st.title("Tutor AI — Final Score Prediction")

colL, colR = st.columns(2)

with colL:
    tutor_model_path = st.text_input("Tutor model path (.joblib)", value="")
    log_path_in = st.text_input("Log path (.txt/.csv) OR logs folder", value="")

    log_path = log_path_in
    if log_path_in and os.path.isdir(log_path_in):
        files = list_log_files(log_path_in)
        if files:
            log_path = st.selectbox("Select a log file", files)
        else:
            st.warning("No .txt/.csv files found in the folder.")

    start_mode = st.radio("Starting point", ["First MatchAttempt", "Row 0"], index=0)
    load_btn = st.button("Load")

if "df" not in st.session_state:
    st.session_state.df = None
if "bundle" not in st.session_state:
    st.session_state.bundle = None
if "i" not in st.session_state:
    st.session_state.i = 0

if load_btn:
    if not tutor_model_path or not os.path.exists(tutor_model_path):
        st.error("Invalid tutor model path.")
    elif not log_path or not os.path.exists(log_path):
        st.error("Invalid log path.")
    else:
        st.session_state.bundle = joblib.load(tutor_model_path)
        st.session_state.df = read_log_file(log_path)

        if start_mode == "First MatchAttempt":
            st.session_state.i = find_first_matchattempt_index(st.session_state.df)
        else:
            st.session_state.i = 0

        st.success(
            f"Loaded: {Path(tutor_model_path).name} + {Path(log_path).name} "
            f"(rows={len(st.session_state.df)})"
        )

df = st.session_state.df
bundle = st.session_state.bundle

with colR:
    if df is None or bundle is None:
        st.info("Insert the paths and press **Load**.")
    else:
        # NAV
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1, 3])

        def clamp():
            st.session_state.i = int(max(0, min(st.session_state.i, len(df) - 1)))

        with c1:
            if st.button("⏮️"):
                st.session_state.i = 0
                clamp()
        with c2:
            if st.button("⬅️"):
                st.session_state.i -= 1
                clamp()
        with c3:
            if st.button("➡️"):
                st.session_state.i += 1
                clamp()
        with c4:
            if st.button("⏭️"):
                st.session_state.i = len(df) - 1
                clamp()
        with c5:
            if st.button("🎯", help="Go to the next MatchAttempt"):
                st.session_state.i = find_next_matchattempt_index(df, int(st.session_state.i))
                clamp()
        with c6:
            st.session_state.i = st.slider("Row index", 0, len(df) - 1, int(st.session_state.i))

        i = int(st.session_state.i)
        df_upto = df.iloc[: i + 1].copy()

        # Indicatori UI
        current_score = get_current_score_percent(df_upto)

        st.subheader("Indicators")
        m1, m2 = st.columns(2)
        m1.metric("Current score (ScoreSummary so far)", f"{current_score:.2f}" if pd.notna(current_score) else "N/A")

        # Stato
        t0 = df["Timestamp"].iloc[0]
        state = compute_state_features(df_upto, t0)

        st.subheader("Suggestion")
        if not state:
            m2.metric("Estimated final score", "N/A")
            st.warning("Empty state (too early in the log). Move forward a few rows.")
        else:
            info = suggest_next_ann(bundle, df_upto)
            last_concept = info["last_concept"]
            next_concept = info["next_concept"]
            next_ann = info["next_ann"]

            st.caption(f"Last solved concept: `{last_concept or 'N/A'}`")
            st.caption(f"Next concept (curriculum order): `{next_concept or 'N/A'}`")
            st.caption(f"Annotation associated with next concept: `{next_ann or 'N/A'}`")

            if not next_ann:
                m2.metric("Estimated final score", "N/A")
                st.warning("Next annotation not found (missing concept_to_ann mapping or concept not mapped).")
            else:
                pred = predict_for_ann(bundle, state, next_ann)
                m2.metric("Estimated final score", f"{pred:.2f}")
                st.success(f"✅ Suggested annotation: **{next_ann}**")

                # sanity check: ann deve esistere nel training
                cands = [str(x) for x in bundle.get("candidate_anns", [])]
                if next_ann not in cands:
                    st.warning("⚠️ This annotation was not present in the training candidate_anns and will be treated as unknown by the encoder.")

        # Evento corrente
        st.subheader("Current event")
        st.write(df.iloc[i][COLS].to_dict())

        with st.expander("Debug: state features so far"):
            st.dataframe(pd.DataFrame([state]).T.rename(columns={0: "value"}), use_container_width=True)

        with st.expander("Debug: feature row sent to model (with next_ann)"):
            if state:
                info = suggest_next_ann(bundle, df_upto)
                next_ann = info["next_ann"]
                if next_ann:
                    X_dbg = make_feature_row(bundle, state, next_ann)
                    st.dataframe(X_dbg, use_container_width=True)
                else:
                    st.info("No next_ann available: cannot build X.")
        
        with st.expander("Debug: concept_to_ann mapping in bundle"):
            st.write(bundle.get("concept_to_ann", {}))