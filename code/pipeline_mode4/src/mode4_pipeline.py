# -*- coding: utf-8 -*-

import os
import glob
import base64
from io import BytesIO
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config
# -----------------------
INPUT_DIR = None
OUT_DIR = None

FEATURES_CSV = None
RES_CSV = None
ASSIGN_CSV = None
PROFILES_CSV = None
REPORT_HTML = None

TIMESTAMP_FMT = "%Y-%m-%d-%H:%M:%S:%f"

COLS = ["Timestamp", "Action", "Field1", "Field2", "Field3", "Field4", "Field5"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PER_SESSION_LIMIT = 20
BIN_SECONDS_TIMELINE = 10
MAIN_ACTIONS = [
    "Touch", "Grab", "Release", "Teleport",
    "PlayAnnotation", "PlayVideo", "Pause", "Resume"
]


# -----------------------
# Feature whitelist
# -----------------------
DESIRED_FEATURES = [
    "session_duration",
    "count_Teleport",
    "count_Grab_on_table",
    "count_Grab_on_map",
    "count_PlayAnnotation",
    "unique_concepts",
    "avg_Play_per_concept",
    "count_Touch_on_table",
    "count_Touch_on_map",
    "sum_resolution_time_for_all_concepts",
    "number_of_concepts_solved",
    "avg_resolution_time_per_concept",

    "resolution_time_concept_cloud_computing",
    "resolution_time_concept_servizio",
    "resolution_time_concept_software",
    "resolution_time_concept_risorse_hardware",
    "resolution_time_concept_fornitore_del_servizio",
    "resolution_time_concept_infrastruttura_cloud",
    "resolution_time_concept_livello_astratto",
    "resolution_time_concept_livello_fisico",
    "resolution_time_concept_modello_di_cloud",
    "resolution_time_concept_SaaS",
    "resolution_time_concept_PaaS",
    "resolution_time_concept_IaaS",
    "resolution_time_concept_On-Premises",
    "resolution_time_concept_caratteristica",

    "number_of_match_attempts",
    "total_match_errors",

    "match_attempt_concept_cloud_computing",
    "match_attempt_concept_servizio",
    "match_attempt_concept_software",
    "match_attempt_concept_risorse_hardware",
    "match_attempt_concept_fornitore_del_servizio",
    "match_attempt_concept_infrastruttura_cloud",
    "match_attempt_concept_livello_fisico",
    "match_attempt_concept_livello_astratto",
    "match_attempt_concept_modello_di_cloud",
    "match_attempt_concept_SaaS",
    "match_attempt_concept_PaaS",
    "match_attempt_concept_IaaS",
    "match_attempt_concept_On_Premises",
    "match_attempt_concept_caratteristica",

    "match_attempts_total",
    "match_correct",
    "match_wrong",
    "match_accuracy",

    "conceptsolve_count",
    "conceptsolve_mean_time",
    "conceptsolve_mean_attempts",
    "conceptsolve_first_try",
    "conceptsolve_multi_try",

    "score_tot_solved",
    "score_tot_attempts",
    "score_tot_accuracy",
    "score_percent",

]

# -----------------------
# Utils grafici
# -----------------------
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"

def make_colors(n):
    if n <= 0:
        return []
    base = plt.cm.tab20(np.linspace(0, 1, 20)).tolist()
    if n <= 20:
        return base[:n]
    extra = plt.cm.hsv(np.linspace(0, 1, max(1, n - 20))).tolist()
    return (base + extra)[:n]

# -----------------------
# PREPROCESS
# -----------------------
def read_session_txt(path):
    """
    Legge i log nella nuova struttura (7 colonne) e ricostruisce:
    - Timestamp (uguale)
    - Action (uguale)
    - ActionID: per Touch/Grab/Release = concept_* (Field2)
    - Direction: 'R' quando presente in Field1 o Field3

    The Field* columns remain in the DataFrame in case they are needed later.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLS,
        sep=",",
        engine="python",
        na_values=["", "null", "NULL", "NaN", "nan", "None"]
    )

    # Timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Normalizza stringhe nei campi raw
    for c in ["Action", "Field1", "Field2", "Field3", "Field4", "Field5"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("").str.strip()

    # Ricostruzione ActionID
    df["ActionID"] = ""
    mask_touch = df["Action"] == "Touch"
    df.loc[mask_touch, "ActionID"] = df.loc[mask_touch, "Field2"]

    mask_grab_rel = df["Action"].isin(["Grab", "Release"])
    df.loc[mask_grab_rel, "ActionID"] = df.loc[mask_grab_rel, "Field2"]

    
    df["Direction"] = ""
    df.loc[df["Field1"] == "R", "Direction"] = "R"
    df.loc[df["Field3"] == "R", "Direction"] = "R"

    df["ActionID"] = df["ActionID"].astype("string").fillna("").str.strip()
    df["Direction"] = df["Direction"].astype("string").fillna("").str.strip()

    return df

def count_action(counts, name):
    return int(counts.get(name, 0))

def avg_touch_to_next_play_seconds(df):
    touch = df.loc[df["Action"] == "Touch", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    play = df.loc[df["Action"] == "PlayAnnotation", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    if touch.size == 0 or play.size == 0:
        return np.nan
    idx = np.searchsorted(play, touch, side="left")
    valid = idx < play.size
    if not np.any(valid):
        return np.nan
    deltas_ns = (play[idx[valid]] - touch[valid]).astype("timedelta64[ns]").astype(np.int64)
    deltas_ns = deltas_ns[deltas_ns >= 0]
    return float(np.mean(deltas_ns / 1e9)) if deltas_ns.size else np.nan

def count_touch_followed_by_play(df):
    touch = df.loc[df["Action"] == "Touch", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    play = df.loc[df["Action"] == "PlayAnnotation", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    if touch.size == 0 or play.size == 0:
        return 0
    idx = np.searchsorted(play, touch, side="left")
    return int(np.sum(idx < play.size))

def extract_features_from_df(df):
    """
    Extracts all Type A + B features.
    Ritorna:
      - base: dict with all scalar features
      - concept_counts: dict concept -> Touch count
      - match_target_counts: dict concetto -> nº MatchAttempt su quel target
      - resolution_time_per_concept: dict concetto -> tempo di risoluzione (sec) da ConceptSolve
    """

    dur = (df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds()
    action_counts = df["Action"].value_counts().to_dict()

    touch_df = df[df["Action"] == "Touch"]
    unique_concepts = int(touch_df["ActionID"].nunique()) if not touch_df.empty else 0
    concept_counts = touch_df["ActionID"].value_counts().to_dict() if not touch_df.empty else {}

    count_PlayAnnotation = count_action(action_counts, "PlayAnnotation")

    count_Grab_on_table = int(((df["Action"] == "Grab") & (df["Field1"] == "table")).sum())
    count_Grab_on_map   = int(((df["Action"] == "Grab") & (df["Field1"] == "map")).sum())

    count_Touch_on_table = int(((df["Action"] == "Touch") & (df["Field1"] == "table")).sum())
    count_Touch_on_map   = int(((df["Action"] == "Touch") & (df["Field1"] == "map")).sum())

    base = {
        "session_duration": float(dur),
        "count_Teleport": count_action(action_counts, "Teleport"),
        "count_Grab": count_action(action_counts, "Grab"),
        "count_Release": count_action(action_counts, "Release"),
        "count_PlayVideo": count_action(action_counts, "PlayVideo"),
        "count_PlayAnnotation": count_PlayAnnotation,
        "count_Pause": count_action(action_counts, "Pause"),
        "count_Resume": count_action(action_counts, "Resume"),
        "count_Touch": count_action(action_counts, "Touch"),
        "count_Grab_on_table": count_Grab_on_table,
        "count_Grab_on_map": count_Grab_on_map,
        "count_Touch_on_table": count_Touch_on_table,
        "count_Touch_on_map": count_Touch_on_map,
        "unique_concepts_touched": unique_concepts,
        "unique_concepts": unique_concepts,   # alias for list B
        "touch_play_count": count_touch_followed_by_play(df),
        "grab_release_count": int(
            min(count_action(action_counts, "Grab"),
                count_action(action_counts, "Release"))
        ),
        "avg_touch_play_duration": avg_touch_to_next_play_seconds(df),
    }

    # avg_Play_per_concept
    if unique_concepts > 0:
        base["avg_Play_per_concept"] = float(count_PlayAnnotation) / float(unique_concepts)
    else:
        base["avg_Play_per_concept"] = np.nan

    ma = df[df["Action"] == "MatchAttempt"].copy()

    match_attempts_total = int(len(ma))
    if match_attempts_total > 0:

        outcome = ma["Field3"].astype(str).str.strip().str.lower()
        correct_mask = outcome.isin(["true", "1", "t", "yes"])

        match_correct = int(correct_mask.sum())
        match_wrong = int(match_attempts_total - match_correct)
        match_accuracy = (
            float(match_correct / match_attempts_total)
            if match_attempts_total > 0 else np.nan
        )

        incorrect = ma[~correct_mask]
        if not incorrect.empty:
            # target concepts on which errors occurred
            error_target_concepts = int(
                incorrect["Field1"].astype(str).nunique()
            )
        else:
            error_target_concepts = 0

        # counts by concept 
        targets = ma["Field1"].astype(str).str.strip()
        match_target_counts = targets.value_counts().to_dict()
    else:
        match_correct = 0
        match_wrong = 0
        match_accuracy = np.nan
        error_target_concepts = 0
        match_target_counts = {}

    base.update({
        "match_attempts_total": match_attempts_total,
        "match_correct": match_correct,
        "match_wrong": match_wrong,
        "match_accuracy": match_accuracy,
        "error_target_concepts": error_target_concepts,
        # alias B
        "number_of_match_attempts": match_attempts_total,
        "total_match_errors": match_wrong,
    })


    cs = df[df["Action"] == "ConceptSolve"].copy()
    conceptsolve_count = int(len(cs))

    # per-concept resolution time (B)
    resolution_time_per_concept = {}
    sum_resolution_time_all = np.nan
    n_concepts_solved = 0
    avg_resolution_time_per_concept = np.nan

    if not cs.empty:
        cs_time = pd.to_numeric(cs["Field4"], errors="coerce")
        cs_attempts = pd.to_numeric(cs["Field5"], errors="coerce")

        valid_time = cs_time.dropna()
        valid_att = cs_attempts.dropna()

        conceptsolve_mean_time = float(valid_time.mean()) if not valid_time.empty else np.nan
        conceptsolve_median_time = float(valid_time.median()) if not valid_time.empty else np.nan
        conceptsolve_max_time = float(valid_time.max()) if not valid_time.empty else np.nan

        conceptsolve_mean_attempts = float(valid_att.mean()) if not valid_att.empty else np.nan
        conceptsolve_median_attempts = float(valid_att.median()) if not valid_att.empty else np.nan
        conceptsolve_max_attempts = float(valid_att.max()) if not valid_att.empty else np.nan

        conceptsolve_first_try = int((cs_attempts == 1).sum())
        conceptsolve_multi_try = int((cs_attempts > 1).sum())

        # per-concept resolution time
        cs_tmp = cs.copy()
        cs_tmp["solve_time"] = pd.to_numeric(cs_tmp["Field4"], errors="coerce")
        cs_tmp = cs_tmp.dropna(subset=["solve_time"])
        if not cs_tmp.empty:
            grp = cs_tmp.groupby("Field1")["solve_time"].last()
            resolution_time_per_concept = grp.to_dict()
            sum_resolution_time_all = float(grp.sum())
            n_concepts_solved = int(grp.size)
            avg_resolution_time_per_concept = float(grp.mean())
        else:
            resolution_time_per_concept = {}
    else:
        conceptsolve_mean_time = np.nan
        conceptsolve_median_time = np.nan
        conceptsolve_max_time = np.nan
        conceptsolve_mean_attempts = np.nan
        conceptsolve_median_attempts = np.nan
        conceptsolve_max_attempts = np.nan
        conceptsolve_first_try = 0
        conceptsolve_multi_try = 0

    base.update({
        "conceptsolve_count": conceptsolve_count,
        "conceptsolve_mean_time": conceptsolve_mean_time,
        "conceptsolve_median_time": conceptsolve_median_time,
        "conceptsolve_max_time": conceptsolve_max_time,
        "conceptsolve_mean_attempts": conceptsolve_mean_attempts,
        "conceptsolve_median_attempts": conceptsolve_median_attempts,
        "conceptsolve_max_attempts": conceptsolve_max_attempts,
        "conceptsolve_first_try": conceptsolve_first_try,
        "conceptsolve_multi_try": conceptsolve_multi_try,
        # B
        "sum_resolution_time_for_all_concepts": sum_resolution_time_all,
        "number_of_concepts_solved": n_concepts_solved,
        "avg_resolution_time_per_concept": avg_resolution_time_per_concept,
    })

    ss = df[df["Action"] == "ScoreSummary"].copy()

    if not ss.empty:
        last = ss.iloc[-1]
        score_tot_solved = pd.to_numeric(last["Field1"], errors="coerce")
        score_tot_attempts = pd.to_numeric(last["Field2"], errors="coerce")
        score_avg_metric = pd.to_numeric(last["Field3"], errors="coerce")
        score_percent = pd.to_numeric(last["Field4"], errors="coerce")

        if pd.notna(score_tot_solved) and pd.notna(score_tot_attempts) and score_tot_attempts > 0:
            score_tot_accuracy = float(score_tot_solved) / float(score_tot_attempts)
        else:
            score_tot_accuracy = np.nan

        score_tot_solved = float(score_tot_solved) if pd.notna(score_tot_solved) else np.nan
        score_tot_attempts = float(score_tot_attempts) if pd.notna(score_tot_attempts) else np.nan
        score_avg_metric = float(score_avg_metric) if pd.notna(score_avg_metric) else np.nan
        score_percent = float(score_percent) if pd.notna(score_percent) else np.nan
    else:
        score_tot_solved = np.nan
        score_tot_attempts = np.nan
        score_tot_accuracy = np.nan
        score_avg_metric = np.nan
        score_percent = np.nan

    if not np.isnan(score_tot_solved):
        score_consistency_solved_vs_conceptsolve = float(score_tot_solved) - float(conceptsolve_count)
    else:
        score_consistency_solved_vs_conceptsolve = np.nan

    if not np.isnan(score_tot_attempts):
        score_consistency_attempts_vs_match = float(score_tot_attempts) - float(match_attempts_total)
    else:
        score_consistency_attempts_vs_match = np.nan

    base.update({
        "score_tot_solved": score_tot_solved,
        "score_tot_attempts": score_tot_attempts,
        "score_tot_accuracy": score_tot_accuracy,
        "score_avg_metric": score_avg_metric,
        "score_percent": score_percent,
        "score_percentage": score_percent,  # alias B
        "score_consistency_solved_vs_conceptsolve": score_consistency_solved_vs_conceptsolve,
        "score_consistency_attempts_vs_match": score_consistency_attempts_vs_match,
    })

    return base, concept_counts, match_target_counts, resolution_time_per_concept

def build_features_and_sessions(input_dir):
    # Supporta sia .txt che .csv
    files_txt = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    files_csv = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    all_files = files_txt + files_csv

    if not all_files:
        raise FileNotFoundError(f"No .txt or .csv found in {input_dir}")

    per_session = []
    sessions_raw = []

    all_touch_concepts = set()
    all_match_concepts = set()
    all_res_concepts = set()

    for path in all_files:
        sid = os.path.splitext(os.path.basename(path))[0]
        df = read_session_txt(path)
        if df.empty:
            continue
        sessions_raw.append((sid, df))
        base, ccounts, mcounts, rtimes = extract_features_from_df(df)
        per_session.append((sid, base, ccounts, mcounts, rtimes))

        all_touch_concepts.update(ccounts.keys())
        all_match_concepts.update(mcounts.keys())
        all_res_concepts.update(rtimes.keys())

    if not per_session:
        raise RuntimeError("No feature rows generated.")

    rows = []
    for sid, base, ccounts, mcounts, rtimes in per_session:
        row = {"sessionID": sid, **base}

        # Touch counts by concept 
        for c in all_touch_concepts:
            row[f"concept_{c}"] = int(ccounts.get(c, 0))

        # number of MatchAttempt events by concept 
        for c in all_match_concepts:
            row[f"match_attempt_{c}"] = int(mcounts.get(c, 0))

        # resolution time by concept 
        for c in all_res_concepts:
            val = rtimes.get(c, np.nan)
            row[f"resolution_time_{c}"] = float(val) if val is not None and not pd.isna(val) else np.nan

        rows.append(row)

    feats = pd.DataFrame(rows)

    # --- Riduzione feature ---
    final_cols = ["sessionID"] + list(DESIRED_FEATURES)

    # Create any missing columns with sensible defaults
    for c in final_cols:
        if c not in feats.columns:
            if c.startswith("match_attempt_") or c in {
                # contatori / interi
                "count_Teleport",
                "count_Grab_on_table",
                "count_Grab_on_map",
                "count_PlayAnnotation",
                "count_Touch_on_table",
                "count_Touch_on_map",
                "unique_concepts",
                "number_of_match_attempts",
                "total_match_errors",
                "number_of_concepts_solved",

                # contatori performance
                "match_attempts_total",
                "match_correct",
                "match_wrong",
                "conceptsolve_count",
                "conceptsolve_first_try",
                "conceptsolve_multi_try",
                "score_tot_solved",
                "score_tot_attempts",
            }:
                feats[c] = 0
            else:
                feats[c] = np.nan


    feats = feats[final_cols].copy()
    return feats, sessions_raw

# -----------------------
# EDA
# -----------------------
def eda_section(X):
    desc_html = X.describe().T.to_html()

    cols = X.columns.tolist()
    n = len(cols)
    c = min(4, max(1, n))
    r = int(np.ceil(n / c))
    fig, axes = plt.subplots(r, c, figsize=(4 * c, 3 * max(1, r)))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        vals = X[col].dropna().values
        if vals.size == 0:
            ax.set_title(f"{col} (empty)", fontsize=9)
            ax.axis("off")
            continue
        ax.hist(vals, bins=20, label=col)
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("Frequency (number of sessions)", fontsize=8)

    for ax in axes[len(cols):]:
        ax.axis("off")

    fig.suptitle("Feature histograms", fontsize=12)
    fig.tight_layout()
    hist_b64 = fig_to_base64(fig)

    corr = X.corr(numeric_only=True)
    fig = plt.figure(figsize=(min(1.2 * len(cols), 16), min(1.2 * len(cols), 12)))
    ax = fig.add_subplot(111)
    if corr.shape[0] > 0:
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90, fontsize=8)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=8)
        ax.set_title("Correlation matrix")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_title("Correlation matrix (not available)")
        ax.axis("off")
    fig.tight_layout()
    corr_b64 = fig_to_base64(fig)

    return {
        "desc_html": desc_html,
        "hist_b64": hist_b64,
        "corr_b64": corr_b64,
    }

# -----------------------
# Feature Selection 
# -----------------------
def variance_corr_filter(X, var_thresh=1e-6, corr_thresh=0.9):
    """
    1) Impute NaN values with the median
    2) remove features with variance < var_thresh (reason='variance')
    3) rimuove feature fortemente correlate (|corr| > corr_thresh) (reason='correlation')
    """
    if X.shape[1] == 0:
        return X.copy(), list(X.columns), pd.DataFrame(
            columns=["reason", "feature", "correlated_with", "corr_value"]
        )

    dropped_rows = []

    # Imputazione
    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # 1) Varianza
    vt = VarianceThreshold(threshold=var_thresh)
    vt.fit(X_imp)
    keep_mask = vt.get_support()
    X1 = X_imp.loc[:, keep_mask]

    for c in X_imp.columns[~keep_mask]:
        dropped_rows.append({
            "reason": "variance",
            "feature": c,
            "correlated_with": "",
            "corr_value": np.nan
        })

    # 2) Correlazione
    if X1.shape[1] > 1:
        corr = X1.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        corr_pairs = []

        for col in upper.columns:
            for row, v in upper[col].dropna().items():
                if v > corr_thresh:
                    to_drop.add(col)
                    corr_pairs.append((col, row, float(v)))

        kept = [c for c in X1.columns if c not in to_drop]
        X2 = X1[kept].copy()

        for feat, other, v in corr_pairs:
            if feat in to_drop:
                dropped_rows.append({
                    "reason": "correlation",
                    "feature": feat,
                    "correlated_with": other,
                    "corr_value": v
                })
    else:
        kept = list(X1.columns)
        X2 = X1.copy()

    dropped_tbl = pd.DataFrame(
        dropped_rows,
        columns=["reason", "feature", "correlated_with", "corr_value"]
    )
    if "corr_value" in dropped_tbl.columns:
        dropped_tbl["corr_value"] = dropped_tbl["corr_value"].round(3)

    return X2, kept, dropped_tbl

# -----------------------
# Clustering 
# -----------------------
def _k_grid():
    return [3, 4, 5, 6, 7]

def _dbscan_auto_eps(Xs, kth=5, perc=90):
    n = Xs.shape[0]
    if n <= kth:
        return 0.5, max(3, n // 10 or 3)
    nn = NearestNeighbors(n_neighbors=kth).fit(Xs)
    dists, _ = nn.kneighbors(Xs)
    kth_d = dists[:, -1]
    eps = float(np.percentile(kth_d, perc))
    min_samples = max(5, int(0.02 * n))
    return max(eps, 1e-6), min_samples

def best_of_five_clustering(X):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_imp)

    n_samples = Xs.shape[0]
    if n_samples < 2:
        return {
            "res_df": pd.DataFrame([{
                "model": "N/A",
                "k": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_clusters": 0
            }]),
            "best_name": "N/A",
            "best_labels": np.full(n_samples, -1),
            "best_k": np.nan,
            "labels_by_model": {},
            "X_imp": X_imp,
            "Xs": Xs
        }

    results = []
    labels_store = {}
    labels_by_model = {}
    k_list = _k_grid()

    for k in k_list:
        # KMeans
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            lab = km.fit_predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"KMeans_k={k}"
            results.append({
                "model": name,
                "k": k,
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(dbi),
                "n_clusters": int(len(uniq))
            })
            labels_store[("KMeans", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

        # Agglomerative
        try:
            agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
            lab = agg.fit_predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"Agglomerative_ward_k={k}"
            results.append({
                "model": name,
                "k": k,
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(dbi),
                "n_clusters": int(len(uniq))
            })
            labels_store[("Agglomerative", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

        # Spectral
        try:
            nn = min(10, max(5, n_samples - 1))
            sp = SpectralClustering(
                n_clusters=k, affinity="nearest_neighbors", n_neighbors=nn,
                assign_labels="kmeans", random_state=RANDOM_STATE
            )
            lab = sp.fit_predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"Spectral_nn={nn}_k={k}"
            results.append({
                "model": name,
                "k": k,
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(dbi),
                "n_clusters": int(len(uniq))
            })
            labels_store[("Spectral", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

        # GMM
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE)
            gmm.fit(Xs)
            lab = gmm.predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"GMM_full_k={k}"
            results.append({
                "model": name,
                "k": k,
                "silhouette": float(sil),
                "calinski_harabasz": float(ch),
                "davies_bouldin": float(dbi),
                "n_clusters": int(len(uniq))
            })
            labels_store[("GMM", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

    # DBSCAN
    try:
        eps, ms = _dbscan_auto_eps(Xs, kth=5, perc=90)
        db = DBSCAN(eps=eps, min_samples=ms)
        lab = db.fit_predict(Xs)
        lbl = np.asarray(lab)
        mask = (lbl != -1)
        uniq = np.unique(lbl[mask]) if np.any(mask) else np.array([])

        if np.sum(mask) >= 2 and len(uniq) >= 2:
            sil = silhouette_score(Xs[mask], lbl[mask])
            ch = calinski_harabasz_score(Xs[mask], lbl[mask])
            dbi = davies_bouldin_score(Xs[mask], lbl[mask])
        else:
            sil = ch = dbi = np.nan

        name = f"DBSCAN_auto(eps~{eps:.3f},min={ms})"
        results.append({
            "model": name,
            "k": np.nan,
            "silhouette": float(sil),
            "calinski_harabasz": float(ch),
            "davies_bouldin": float(dbi),
            "n_clusters": int(len(uniq))
        })
        labels_store[("DBSCAN", np.nan)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

    if not results:
        return {
            "res_df": pd.DataFrame([{
                "model": "N/A",
                "k": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_clusters": 0
            }]),
            "best_name": "N/A",
            "best_labels": np.full(Xs.shape[0], -1),
            "best_k": np.nan,
            "labels_by_model": {},
            "X_imp": X_imp,
            "Xs": Xs
        }

    res_df = pd.DataFrame(results).sort_values(
        by=["silhouette", "n_clusters", "davies_bouldin"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    best_name = res_df.iloc[0]["model"]
    best_k = res_df.iloc[0]["k"]

    if best_name.startswith("KMeans"):
        key = ("KMeans", int(best_k))
    elif best_name.startswith("Agglomerative"):
        key = ("Agglomerative", int(best_k))
    elif best_name.startswith("Spectral"):
        key = ("Spectral", int(best_k))
    elif best_name.startswith("GMM"):
        key = ("GMM", int(best_k))
    else:
        key = ("DBSCAN", np.nan)

    best_labels = labels_store.get(key, np.full(Xs.shape[0], -1))

    return {
        "res_df": res_df,
        "best_name": best_name,
        "best_labels": best_labels,
        "best_k": best_k,
        "labels_by_model": labels_by_model,
        "X_imp": X_imp,
        "Xs": Xs
    }

def best_of_five_concept_clustering(concept_df, k_fixed=3):
    """
    Clustering sui concept (righe = concept).
    5 modelli: KMeans, Agglomerative, Spectral, GMM, DBSCAN.
    k=3 for the first 4, DBSCAN auto-eps.
    Best: silhouette desc, n_clusters desc, davies_bouldin asc (same as sessions).
    """
    if concept_df is None or concept_df.empty:
        return {
            "res_df": pd.DataFrame([{
                "model": "N/A",
                "k": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_clusters": 0
            }]),
            "best_name": "N/A",
            "best_labels": np.array([], dtype=int),
            "best_k": np.nan,
            "labels_by_model": {},
            "X_imp": np.empty((0, 0)),
            "Xs": np.empty((0, 0)),
            "feature_cols": []
        }

    feature_cols = [
        "n_sessions",
        "solve_mean_s",
        "match_accuracy",
        "play_max_streak",
        "match_max_streak",
        "problem_score",
    ]
    feature_cols = [c for c in feature_cols if c in concept_df.columns]

    X = concept_df[feature_cols].copy()
    if X.shape[0] < 2 or X.shape[1] == 0:
        n = X.shape[0]
        return {
            "res_df": pd.DataFrame([{
                "model": "N/A",
                "k": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_clusters": 0
            }]),
            "best_name": "N/A",
            "best_labels": np.full(n, -1),
            "best_k": np.nan,
            "labels_by_model": {},
            "X_imp": X.values,
            "Xs": X.values,
            "feature_cols": feature_cols
        }

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    sc = StandardScaler()
    Xs = sc.fit_transform(X_imp)

    n_samples = Xs.shape[0]
    results = []
    labels_store = {}
    labels_by_model = {}

    k = int(k_fixed)

    # KMeans
    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        lab = km.fit_predict(Xs)
        uniq = np.unique(lab)
        sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
        ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
        dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
        name = f"KMeans_k={k}"
        results.append({"model": name, "k": k, "silhouette": float(sil),
                        "calinski_harabasz": float(ch), "davies_bouldin": float(dbi),
                        "n_clusters": int(len(uniq))})
        labels_store[("KMeans", k)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

    # Agglomerative
    try:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = agg.fit_predict(Xs)
        uniq = np.unique(lab)
        sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
        ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
        dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
        name = f"Agglomerative_ward_k={k}"
        results.append({"model": name, "k": k, "silhouette": float(sil),
                        "calinski_harabasz": float(ch), "davies_bouldin": float(dbi),
                        "n_clusters": int(len(uniq))})
        labels_store[("Agglomerative", k)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

    # Spectral
    try:
        nn = min(10, max(5, n_samples - 1))
        sp = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=nn,
            assign_labels="kmeans", random_state=RANDOM_STATE
        )
        lab = sp.fit_predict(Xs)
        uniq = np.unique(lab)
        sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
        ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
        dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
        name = f"Spectral_nn={nn}_k={k}"
        results.append({"model": name, "k": k, "silhouette": float(sil),
                        "calinski_harabasz": float(ch), "davies_bouldin": float(dbi),
                        "n_clusters": int(len(uniq))})
        labels_store[("Spectral", k)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

    # GMM
    try:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE)
        gmm.fit(Xs)
        lab = gmm.predict(Xs)
        uniq = np.unique(lab)
        sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
        ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
        dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
        name = f"GMM_full_k={k}"
        results.append({"model": name, "k": k, "silhouette": float(sil),
                        "calinski_harabasz": float(ch), "davies_bouldin": float(dbi),
                        "n_clusters": int(len(uniq))})
        labels_store[("GMM", k)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

    # DBSCAN 
    try:
        eps, ms = _dbscan_auto_eps(Xs, kth=5, perc=90)
        db = DBSCAN(eps=eps, min_samples=ms)
        lab = db.fit_predict(Xs)
        lbl = np.asarray(lab)
        mask = (lbl != -1)
        uniq = np.unique(lbl[mask]) if np.any(mask) else np.array([])

        if np.sum(mask) >= 2 and len(uniq) >= 2:
            sil = silhouette_score(Xs[mask], lbl[mask])
            ch = calinski_harabasz_score(Xs[mask], lbl[mask])
            dbi = davies_bouldin_score(Xs[mask], lbl[mask])
        else:
            sil = ch = dbi = np.nan

        name = f"DBSCAN_auto(eps~{eps:.3f},min={ms})"
        results.append({"model": name, "k": np.nan, "silhouette": float(sil),
                        "calinski_harabasz": float(ch), "davies_bouldin": float(dbi),
                        "n_clusters": int(len(uniq))})
        labels_store[("DBSCAN", np.nan)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

    if not results:
        return {
            "res_df": pd.DataFrame([{
                "model": "N/A",
                "k": np.nan,
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "n_clusters": 0
            }]),
            "best_name": "N/A",
            "best_labels": np.full(n_samples, -1),
            "best_k": np.nan,
            "labels_by_model": {},
            "X_imp": X_imp,
            "Xs": Xs,
            "feature_cols": feature_cols
        }

    res_df = pd.DataFrame(results).sort_values(
        by=["silhouette", "n_clusters", "davies_bouldin"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    best_name = res_df.iloc[0]["model"]
    best_k = res_df.iloc[0]["k"]

    if best_name.startswith("KMeans"):
        key = ("KMeans", int(best_k))
    elif best_name.startswith("Agglomerative"):
        key = ("Agglomerative", int(best_k))
    elif best_name.startswith("Spectral"):
        key = ("Spectral", int(best_k))
    elif best_name.startswith("GMM"):
        key = ("GMM", int(best_k))
    else:
        key = ("DBSCAN", np.nan)

    best_labels = labels_store.get(key, np.full(n_samples, -1))

    return {
        "res_df": res_df,
        "best_name": best_name,
        "best_labels": best_labels,
        "best_k": best_k,
        "labels_by_model": labels_by_model,
        "X_imp": X_imp,
        "Xs": Xs,
        "feature_cols": feature_cols
    }


def pca_scatter_for_labels(Xs, labels, title="PCA scatter"):
    if Xs.shape[1] < 2 or Xs.shape[0] < 2:
        return ""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    PC = pca.fit_transform(Xs)

    uniq = sorted([l for l in np.unique(labels) if l != -1]) + \
           ([-1] if np.any(labels == -1) else [])
    colors = make_colors(len(uniq))
    cmap = {l: colors[i % len(colors)] for i, l in enumerate(uniq)}

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(
        PC[:, 0], PC[:, 1],
        c=[cmap.get(l, (0.6, 0.6, 0.6, 0.6)) for l in labels],
        s=25, alpha=0.8, edgecolor="none"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    handles = []
    for l in uniq:
        label = "noise (-1)" if l == -1 else f"cluster {l}"
        handles.append(Patch(facecolor=cmap[l], edgecolor="none", label=label))
    if handles:
        ax.legend(handles=handles, title="Cluster", fontsize=8)

    fig.tight_layout()
    return fig_to_base64(fig)

# -----------------------
# Sequence Analysis 
# -----------------------
def sessions_to_sequences(sessions_raw):
    return {sid: df["Action"].astype(str).tolist() for sid, df in sessions_raw}

def ngram_counts(seqs, n=2, top=30):
    cnt = Counter()
    for s in seqs.values():
        for i in range(len(s) - n + 1):
            cnt[tuple(s[i:i + n])] += 1
    items = cnt.most_common(top)
    return pd.DataFrame([{"ngram": " → ".join(k), "count": v} for k, v in items])

def transition_matrix(sessions_raw, actions_order=None):
    all_actions = set()
    for _, df in sessions_raw:
        all_actions.update(df["Action"].astype(str).unique().tolist())
    acts = actions_order if actions_order else sorted(all_actions)
    idx = {a: i for i, a in enumerate(acts)}

    M = np.zeros((len(acts), len(acts)), dtype=float)
    for _, df in sessions_raw:
        s = df["Action"].astype(str).tolist()
        for i in range(len(s) - 1):
            M[idx[s[i]], idx[s[i + 1]]] += 1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = M.sum(axis=1, keepdims=True)
        P = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)
    return acts, P

def transition_heatmap_b64(acts, P):
    fig = plt.figure(figsize=(min(1.2 * len(acts), 16), min(1.2 * len(acts), 12)))
    ax = fig.add_subplot(111)
    if len(acts) > 0:
        im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(acts)))
        ax.set_xticklabels(acts, rotation=90, fontsize=8)
        ax.set_yticks(range(len(acts)))
        ax.set_yticklabels(acts, fontsize=8)
        ax.set_title("Transition matrix (probabilities)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_title("Transition matrix (not available)")
        ax.axis("off")
    fig.tight_layout()
    return fig_to_base64(fig)

def dwell_time_stats(sessions_raw):
    times = {}
    for _, df in sessions_raw:
        for a, g in df.groupby("Action"):
            ts = g["Timestamp"].sort_values().values.astype("datetime64[ns]").astype(np.int64) / 1e9
            if len(ts) >= 2:
                times.setdefault(a, []).extend(np.diff(ts).tolist())
    if not times:
        return pd.DataFrame(columns=["action", "mean_dwell_s", "median_dwell_s", "count_intervals"])

    rows = []
    for a, arr in times.items():
        arr = np.array(arr, dtype=float)
        rows.append({
            "action": str(a),
            "mean_dwell_s": float(np.mean(arr)),
            "median_dwell_s": float(np.median(arr)),
            "count_intervals": int(len(arr))
        })
    return pd.DataFrame(rows).sort_values("mean_dwell_s")

def error_cooccurrence_matrix(sessions_raw):
    """
    n×n matrix of conceptual errors from incorrect MatchAttempt events.
    Rows = target concept (Field1)
    Columns = chosen concept (Field2)
    Value = number of errors.
    """
    concepts = set()
    conf_counts = {}

    for _, df in sessions_raw:
        if "Action" not in df.columns:
            continue

        ma = df[df["Action"] == "MatchAttempt"].copy()
        if ma.empty:
            continue

        ma["Outcome"] = ma["Field3"].astype(str).str.strip().str.lower()
        wrong = ma[ma["Outcome"].isin(["false", "0", "no", "f"])]
        if wrong.empty:
            continue

        for _, row in wrong.iterrows():
            target = str(row["Field1"]).strip()
            chosen = str(row["Field2"]).strip()
            if not target or not chosen:
                continue

            concepts.update([target, chosen])
            key = (target, chosen)
            conf_counts[key] = conf_counts.get(key, 0) + 1

    concepts = sorted(concepts)
    n = len(concepts)

    if n == 0:
        return concepts, np.zeros((0, 0), dtype=int), pd.DataFrame(), ""

    idx = {c: i for i, c in enumerate(concepts)}
    M = np.zeros((n, n), dtype=int)

    for (target, chosen), v in conf_counts.items():
        i = idx[target]
        j = idx[chosen]
        M[i, j] += v

    df_mat = pd.DataFrame(M, index=concepts, columns=concepts)

    fig = plt.figure(figsize=(min(1.2 * n, 16), min(1.2 * n, 12)))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, cmap="Reds")

    ax.set_xticks(range(n))
    ax.set_xticklabels(concepts, rotation=90, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(concepts, fontsize=8)
    ax.set_title("Error matrix: target concept vs chosen concept (incorrect matches only)")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    heatmap_b64 = fig_to_base64(fig)

    return concepts, M, df_mat, heatmap_b64

def session_timeline_plot(df):
    t0 = df["Timestamp"].iloc[0]
    df = df.copy()
    df["t_sec"] = (df["Timestamp"] - t0).dt.total_seconds()
    actions = list(dict.fromkeys(list(MAIN_ACTIONS) + df["Action"].astype(str).tolist()))
    colors = make_colors(len(actions))
    cmap = {a: colors[i % len(colors)] for i, a in enumerate(actions)}

    fig = plt.figure(figsize=(9, 3.6))
    ax = fig.add_subplot(111)
    ax.scatter(
        df["t_sec"],
        [actions.index(a) for a in df["Action"].astype(str)],
        c=[cmap[a] for a in df["Action"].astype(str)],
        s=18, alpha=0.8, edgecolor="none"
    )
    ax.set_yticks(range(len(actions)))
    ax.set_yticklabels(actions, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Event timeline")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)

def session_confusion_errors(df):
    """
    Returns, for ONE single session (df),
    the table of confusion errors between concepts:
      target_concept | chosen_concept | count
    considerando solo i MatchAttempt errati.
    """
    ma = df[df["Action"] == "MatchAttempt"].copy()
    if ma.empty:
        return pd.DataFrame(columns=["target_concept", "chosen_concept", "count"])

    outcome = ma["Field3"].astype(str).str.strip().str.lower()
    incorrect = ma[outcome.isin(["false", "0", "no", "f"])]

    if incorrect.empty:
        return pd.DataFrame(columns=["target_concept", "chosen_concept", "count"])

    tmp = pd.DataFrame({
        "target_concept": incorrect["Field1"].astype(str).str.strip(),
        "chosen_concept": incorrect["Field2"].astype(str).str.strip(),
    })

    tmp = tmp[(tmp["target_concept"] != "") & (tmp["chosen_concept"] != "")]
    if tmp.empty:
        return pd.DataFrame(columns=["target_concept", "chosen_concept", "count"])

    agg = tmp.groupby(["target_concept", "chosen_concept"]).size().reset_index(name="count")
    agg = agg.sort_values("count", ascending=False).reset_index(drop=True)
    return agg

def confusion_matrix_concepts(sessions_raw):
    """
    Long table of conceptual confusions:
    target_concept (Field1) vs chosen_concept (Field2) for incorrect MatchAttempt events.
    """
    rows = []
    for sid, df in sessions_raw:
        ma = df[df["Action"] == "MatchAttempt"].copy()
        if ma.empty:
            continue

        outcome = ma["Field3"].astype(str).str.strip().str.lower()
        incorrect = ma[outcome.isin(["false", "0", "no", "f"])]

        if incorrect.empty:
            continue

        for _, r in incorrect.iterrows():
            rows.append({
                "sessionID": sid,
                "target_concept": str(r["Field1"]),
                "chosen_concept": str(r["Field2"])
            })

    if not rows:
        return pd.DataFrame(columns=["sessionID", "target_concept", "chosen_concept", "count"])

    df_all = pd.DataFrame(rows)
    agg = df_all.groupby(["target_concept", "chosen_concept"]).size().reset_index(name="count")
    agg = agg.sort_values("count", ascending=False)
    return agg

# -----------------------
# Archetypes — combina pattern sequenziali + performance
# -----------------------
def _q(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.quantile(q)) if not s.empty else np.nan

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _add_arch(archetypes, name):
    if name and name not in archetypes:
        archetypes.append(name)

# Mappa traduzione archetipi EN -> IT
ARCHETYPE_TRANSLATION = {
    "Efficient Expert": "Efficient Expert",
    "Linear Achiever": "Linear Achiever",
    "Explorer": "Explorer",
    "Interrupted / Idle": "Interrupted / Idle",
    "Guessing Matcher": "Guessing Matcher",
    "Persistent Improver": "Persistent Improver",
    "Methodical Solver": "Methodical Solver",
    "Fast but Error-prone": "Fast but Error-prone",
    "Struggler": "Struggler",
    "Fragmented Cautious": "Fragmented Cautious",
    "Mixed Profile": "Mixed Profile"
}


def assign_archetypes_row(row, thr):
    """
    Ritorna:
      - archetype_primary (string)
      - archetypes (lista)
    Regole data-driven: soglie basate su quantili del dataset.
    """
    arch = []

    # --- metriche sequenza ---
    t2p = _safe_float(row.get("touch_to_play_rate", np.nan))
    mt2p = _safe_float(row.get("median_touch_to_play_sec", np.nan))
    pause = _safe_float(row.get("pause_per_min", np.nan))
    tele = _safe_float(row.get("teleport_per_min", np.nan))
    idle = _safe_float(row.get("idle_gaps_gt60", np.nan))
    ent = _safe_float(row.get("entropy_next_after_touch", np.nan))
    dur = _safe_float(row.get("duration_sec", np.nan))
    pscore = _safe_float(row.get("pattern_score", np.nan))

    # --- performance ---
    macc = _safe_float(row.get("match_accuracy", np.nan))
    mw = _safe_float(row.get("match_wrong", np.nan))
    mat = _safe_float(row.get("match_attempts_total", np.nan))
    csc = _safe_float(row.get("conceptsolve_count", np.nan))
    cs_t = _safe_float(row.get("conceptsolve_mean_time", np.nan))
    cs_a = _safe_float(row.get("conceptsolve_mean_attempts", np.nan))
    cs_ft = _safe_float(row.get("conceptsolve_first_try", np.nan))
    cs_mt = _safe_float(row.get("conceptsolve_multi_try", np.nan))
    scorep = _safe_float(row.get("score_percent", np.nan))
    st_solved = _safe_float(row.get("score_tot_solved", np.nan))

    hi = thr

    def is_hi(x, key):
        return (not np.isnan(x)) and (key in hi) and (not np.isnan(hi[key])) and (x >= hi[key])

    def is_lo(x, key):
        return (not np.isnan(x)) and (key in hi) and (not np.isnan(hi[key])) and (x <= hi[key])

    # 1) Lineare & performante
    if is_hi(t2p, "t2p_hi") and is_lo(ent, "ent_lo") and is_hi(scorep, "scorep_hi"):
        _add_arch(arch, "Linear & high-performing")

    # 2) Rapido & preciso
    if is_lo(mt2p, "mt2p_lo") and is_lo(cs_t, "cs_time_lo") and is_hi(macc, "macc_hi") and is_hi(scorep, "scorep_hi"):
        _add_arch(arch, "Fast & accurate")

    # 3) Slow but accurate
    if is_hi(mt2p, "mt2p_hi") and is_hi(cs_t, "cs_time_hi") and is_hi(macc, "macc_hi"):
        _add_arch(arch, "Slow but accurate")

    # 4) Fast but superficial
    if is_lo(mt2p, "mt2p_lo") and is_lo(cs_t, "cs_time_lo") and is_lo(macc, "macc_lo"):
        _add_arch(arch, "Fast but superficial")

    # 5) Esploratore erratico
    if is_hi(tele, "tele_hi") and is_hi(ent, "ent_hi"):
        _add_arch(arch, "Erratic explorer")

    # 6) Navigazione compulsiva
    if is_hi(tele, "tele_vhi") and not is_lo(scorep, "scorep_lo"):
        _add_arch(arch, "Compulsive navigator")

    # 7) Fragmented session
    if is_hi(pause, "pause_hi") and (idle > 0 or is_hi(idle, "idle_hi")):
        _add_arch(arch, "Fragmented session")

    # 8) Abbandono precoce
    if is_lo(dur, "dur_lo") and is_lo(st_solved, "solved_lo") and (is_hi(idle, "idle_hi") or idle > 0):
        _add_arch(arch, "Early dropout")

    # 9) Repeated attempts
    if is_hi(mat, "mat_hi") and is_hi(cs_a, "cs_att_hi") and is_hi(cs_mt, "cs_mt_hi"):
        _add_arch(arch, "Repeated attempts")

    # 10) Confusione concettuale
    if is_hi(mw, "mw_hi") and is_lo(macc, "macc_lo"):
        _add_arch(arch, "Conceptual confusion")

    # 11) Match-heavy
    if is_hi(mat, "mat_hi") and (not np.isnan(csc)) and csc > 0 and (mat / max(csc, 1.0)) >= 3.0:
        _add_arch(arch, "Match-heavy")

    # 12) Solve-focused
    if is_hi(csc, "csc_hi") and not is_hi(mw, "mw_hi") and is_hi(scorep, "scorep_hi"):
        _add_arch(arch, "Solve-focused")

    # 13) Apprendimento progressivo
    if is_hi(mat, "mat_hi") and is_hi(cs_mt, "cs_mt_hi") and not is_lo(scorep, "scorep_lo"):
        _add_arch(arch, "Progressive learner")

    # 14) Strategia prudente
    if is_hi(t2p, "t2p_hi") and is_hi(macc, "macc_hi") and (not np.isnan(pause)) and ("pause_med" in hi) and (not np.isnan(hi["pause_med"])) and pause >= hi["pause_med"]:
        _add_arch(arch, "Cautious strategy")

    # 15) Workflow disallineato
    if not np.isnan(pscore) and pscore < 0 and is_hi(scorep, "scorep_hi"):
        _add_arch(arch, "Misaligned workflow")

    # 16) Risolutore immediato
    if is_hi(cs_ft, "cs_ft_hi") and is_hi(macc, "macc_hi"):
        _add_arch(arch, "Immediate solver")

    # 17) Struggling with concepts
    if is_hi(cs_t, "cs_time_hi") and is_hi(cs_a, "cs_att_hi"):
        _add_arch(arch, "Struggling with concepts")

    # 18) Performance bassa
    if is_lo(scorep, "scorep_lo") and is_lo(macc, "macc_lo"):
        _add_arch(arch, "Low performance")

    if not arch:
        arch = ["Mixed Profile"]

    priority = [
        "Linear & high-performing",
        "Fast & accurate",
        "Slow but accurate",
        "Fast but superficial",
        "Conceptual confusion",
        "Repeated attempts",
        "Erratic explorer",
        "Fragmented session",
        "Early dropout",
        "Struggling with concepts",
        "Low performance",
        "Mixed Profile",
    ]
    primary = None
    for p in priority:
        if p in arch:
            primary = p
            break
    if primary is None:
        primary = arch[0]
    return primary, arch

# -----------------------
# Sequence Pattern Detection + Archetypes
# -----------------------
def _entropy(prob_vec):
    p = np.asarray([x for x in prob_vec if x > 0], dtype=float)
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))

def touch_to_play_deltas(df):
    touch = df.loc[df["Action"] == "Touch", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    play = df.loc[df["Action"] == "PlayAnnotation", "Timestamp"].to_numpy(dtype="datetime64[ns]")
    if touch.size == 0 or play.size == 0:
        return np.array([], dtype=float)
    idx = np.searchsorted(play, touch, side="left")
    valid = idx < play.size
    deltas = (play[idx[valid]] - touch[valid]).astype("timedelta64[ns]").astype(np.int64) / 1e9
    return deltas[deltas >= 0]

def next_action_entropy_after(df, action_name="Touch"):
    nxt = []
    actions = df["Action"].astype(str).tolist()
    for i, a in enumerate(actions[:-1]):
        if a == action_name:
            nxt.append(actions[i + 1])
    if not nxt:
        return 0.0
    cnt = Counter(nxt)
    p = np.array(list(cnt.values()), dtype=float)
    return _entropy(p)

def max_consecutive(df, action_name):
    runs, curr = 0, 0
    for a in df["Action"].astype(str):
        if a == action_name:
            curr += 1
            runs = max(runs, curr)
        else:
            curr = 0
    return runs

def count_idle_gaps(df, thr_sec=60.0):
    ts = df["Timestamp"].sort_values().values.astype("datetime64[ns]").astype(np.int64) / 1e9
    if ts.size < 2:
        return 0
    gaps = np.diff(ts)
    return int(np.sum(gaps > thr_sec))

def per_minute_rate(df, action_name):
    dur = (df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds()
    if dur <= 0:
        return 0.0
    c = int((df["Action"] == action_name).sum())
    return 60.0 * c / dur

def session_sequence_metrics(df):
    deltas = touch_to_play_deltas(df)
    n_touch = int((df["Action"] == "Touch").sum())
    t2p_rate = float(len(deltas)) / float(n_touch) if n_touch > 0 else 0.0

    metrics = {
        "n_events": int(len(df)),
        "duration_sec": float((df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds()),
        "touch_count": n_touch,
        "playann_count": int((df["Action"] == "PlayAnnotation").sum()),
        "touch_to_play_rate": t2p_rate,
        "median_touch_to_play_sec": (float(np.median(deltas)) if deltas.size else np.nan),
        "pause_per_min": per_minute_rate(df, "Pause"),
        "teleport_per_min": per_minute_rate(df, "Teleport"),
        "max_pause_streak": max_consecutive(df, "Pause"),
        "max_teleport_streak": max_consecutive(df, "Teleport"),
        "idle_gaps_gt60": count_idle_gaps(df, thr_sec=60.0),
        "entropy_next_after_touch": next_action_entropy_after(df, "Touch"),
    }
    return metrics

def pattern_score_hist_b64(pat_sess_df):
    if pat_sess_df.empty or "pattern_score" not in pat_sess_df.columns:
        return ""
    vals = pat_sess_df["pattern_score"].dropna().values
    if vals.size == 0:
        return ""
    fig = plt.figure(figsize=(6.5, 3.2))
    ax = fig.add_subplot(111)
    ax.hist(vals, bins=min(12, max(3, len(np.unique(vals)))))
    ax.set_title("Pattern score distribution")
    ax.set_xlabel("Pattern score (higher = more linear / effective)")
    ax.set_ylabel("Number of sessions")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig_to_base64(fig)

def _safe_quantile(series, q):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(q))

def _in_top(series, value, q=0.8):
    thr = _safe_quantile(series, q)
    return (not np.isnan(thr)) and (value is not None) and (not (isinstance(value, float) and np.isnan(value))) and (float(value) >= thr)

def _in_bottom(series, value, q=0.2):
    thr = _safe_quantile(series, q)
    return (not np.isnan(thr)) and (value is not None) and (not (isinstance(value, float) and np.isnan(value))) and (float(value) <= thr)

def assign_archetypes(pat_df, min_hits=2, max_secondary=4):
    """
    Soft archetype inference:
    - each archetype has conditions (hits). Each true condition = +1.
    - the archetype is activated if hits >= min_hits (default 2).
    - primary = archetype with the highest hits (priority-based tie-break).
    - archetypes = list ('; ' string) of active archetypes (or top-k).
    """
    if pat_df is None or pat_df.empty:
        return pat_df

    df = pat_df.copy()

    # --- required columns ---
    needed = [
        "pattern_score",
        "median_touch_to_play_sec",
        "pause_per_min",
        "teleport_per_min",
        "idle_gaps_gt60",
        "entropy_next_after_touch",
        "match_attempts_total",
        "match_wrong",
        "match_accuracy",
        "conceptsolve_count",
        "conceptsolve_mean_time",
        "conceptsolve_mean_attempts",
        "conceptsolve_first_try",
        "score_percent",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    # --- soglie dataset-relative: P80 / P20 / mediana ---
    def q(col, quant):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.quantile(quant)) if not s.empty else np.nan

    def med(col):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.median()) if not s.empty else np.nan

    thr = {
        # P80 (alto)
        "ps_p80": q("pattern_score", 0.80),
        "score_p80": q("score_percent", 0.80),
        "macc_p80": q("match_accuracy", 0.80),
        "tele_p80": q("teleport_per_min", 0.80),
        "pause_p80": q("pause_per_min", 0.80),
        "idle_p80": q("idle_gaps_gt60", 0.80),
        "ent_p80": q("entropy_next_after_touch", 0.80),
        "mat_p80": q("match_attempts_total", 0.80),
        "mw_p80": q("match_wrong", 0.80),
        "cs_time_p80": q("conceptsolve_mean_time", 0.80),
        "cs_att_p80": q("conceptsolve_mean_attempts", 0.80),

        # P20 (basso)
        "score_p20": q("score_percent", 0.20),
        "macc_p20": q("match_accuracy", 0.20),
        "mt2p_p20": q("median_touch_to_play_sec", 0.20),
        "cs_time_p20": q("conceptsolve_mean_time", 0.20),

        # mediane
        "score_med": med("score_percent"),
        "macc_med": med("match_accuracy"),
        "pause_med": med("pause_per_min"),
        "tele_med": med("teleport_per_min"),
        "cs_time_med": med("conceptsolve_mean_time"),
        "cs_att_med": med("conceptsolve_mean_attempts"),
        "ps_med": med("pattern_score"),
    }

    def num(x):
        return pd.to_numeric(x, errors="coerce")

    def ge(x, t):
        return (pd.notna(x) and pd.notna(t) and float(x) >= float(t))

    def le(x, t):
        return (pd.notna(x) and pd.notna(t) and float(x) <= float(t))

    priority = [
        "Efficient Expert",
        "Linear Achiever",
        "Methodical Solver",
        "Explorer",
        "Persistent Improver",
        "Fast but Error-prone",
        "Guessing Matcher",
        "Fragmented Cautious",
        "Interrupted / Idle",
        "Struggler",
        "Mixed Profile",
    ]
    pr_rank = {a: i for i, a in enumerate(priority)}

    prim_list = []
    arch_list = []

    for _, r in df.iterrows():
        ps = num(r.get("pattern_score"))
        scorep = num(r.get("score_percent"))
        macc = num(r.get("match_accuracy"))
        mw = num(r.get("match_wrong"))
        mat = num(r.get("match_attempts_total"))

        mt2p = num(r.get("median_touch_to_play_sec"))
        pause = num(r.get("pause_per_min"))
        tele = num(r.get("teleport_per_min"))
        idle = num(r.get("idle_gaps_gt60"))
        ent = num(r.get("entropy_next_after_touch"))

        cs_cnt = num(r.get("conceptsolve_count"))
        cs_time = num(r.get("conceptsolve_mean_time"))
        cs_att = num(r.get("conceptsolve_mean_attempts"))
        cs_ft = num(r.get("conceptsolve_first_try"))

        first_try_rate = np.nan
        if pd.notna(cs_cnt) and float(cs_cnt) > 0 and pd.notna(cs_ft):
            first_try_rate = float(cs_ft) / float(cs_cnt)

        hits = {}

        def count_hits(name, conds):
            hits[name] = int(sum(bool(c) for c in conds))

        count_hits("Efficient Expert", [
            ge(ps, thr["ps_p80"]),
            ge(scorep, thr["score_p80"]),
            ge(macc, thr["macc_p80"]),
            (pd.notna(first_try_rate) and first_try_rate >= 0.70),
            le(cs_time, thr["cs_time_med"]),
        ])

        count_hits("Linear Achiever", [
            (pd.notna(ps) and float(ps) >= 0),
            ge(scorep, thr["score_med"]),
            ge(macc, thr["macc_med"]),
            le(pause, thr["pause_p80"]),
            le(tele, thr["tele_p80"]),
        ])

        count_hits("Explorer", [
            ge(tele, thr["tele_p80"]),
            ge(ent, thr["ent_p80"]),
            ge(mat, thr["mat_p80"]),
        ])

        count_hits("Interrupted / Idle", [
            ge(idle, thr["idle_p80"]),
            ge(pause, thr["pause_p80"]),
            (pd.notna(ps) and pd.notna(thr["ps_med"]) and float(ps) <= float(thr["ps_med"])),
        ])

        count_hits("Guessing Matcher", [
            ge(mat, thr["mat_p80"]),
            le(macc, thr["macc_p20"]),
            ge(mw, thr["mw_p80"]),
            ge(cs_att, thr["cs_att_med"]),
        ])

        count_hits("Persistent Improver", [
            ge(mat, thr["mat_p80"]),
            ge(cs_att, thr["cs_att_p80"]),
            (pd.notna(ps) and float(ps) >= 0),
            ge(macc, thr["macc_p20"]),
            (not le(scorep, thr["score_p20"])),
        ])

        count_hits("Methodical Solver", [
            ge(cs_time, thr["cs_time_p80"]),
            ge(macc, thr["macc_p80"]),
            le(tele, thr["tele_med"]),
            (pd.notna(ps) and float(ps) >= 0),
        ])

        count_hits("Fast but Error-prone", [
            le(mt2p, thr["mt2p_p20"]),
            le(cs_time, thr["cs_time_p20"]),
            le(macc, thr["macc_p20"]),
            ge(mw, thr["mw_p80"]),
        ])

        count_hits("Struggler", [
            le(scorep, thr["score_p20"]),
            le(macc, thr["macc_p20"]),
            ge(cs_time, thr["cs_time_p80"]),
            ge(cs_att, thr["cs_att_p80"]),
            ge(pause, thr["pause_med"]),
        ])

        count_hits("Fragmented Cautious", [
            ge(pause, thr["pause_p80"]),
            ge(ent, thr["ent_p80"]),
            ge(scorep, thr["score_p20"]),
            ge(macc, thr["macc_p20"]),
            (pd.notna(ps) and pd.notna(thr["ps_med"]) and float(ps) >= float(thr["ps_med"])),
        ])

        active = [k for k, h in hits.items() if h >= int(min_hits)]

        if not active:
            primary = "Mixed Profile"
            multi = "Mixed Profile"
        else:
            active_sorted = sorted(active, key=lambda k: (-hits[k], pr_rank.get(k, 999)))
            primary = active_sorted[0]
            active_top = active_sorted[:max_secondary] if max_secondary else active_sorted
            multi = "; ".join(active_top)

        primary_it = ARCHETYPE_TRANSLATION.get(primary, primary)
        multi_it = "; ".join(
            ARCHETYPE_TRANSLATION.get(a.strip(), a.strip())
            for a in str(multi).split(";")
            if a.strip()
        )

        prim_list.append(primary_it)
        arch_list.append(multi_it)

    df["archetype_primary"] = prim_list
    df["archetypes"] = arch_list
    return df


def sequence_pattern_detection(sessions_raw, feats_df, clust_payload):
    """
    v2.1:
    - computes sequence metrics for each session
    - soglie data-driven basate sui quantili
    - genera flag 'effective' e 'problematic'
    - computes pattern_score (effective - problematic)
    - links each session to the cluster
    - merges with feats_df (includes MatchAttempt/ConceptSolve/ScoreSummary metrics, etc.)
    - assegna archetipi (archetype_primary + archetypes)
    - computes bigrams/trigrams and the transition matrix
    """
    rows = []
    for sid, df in sessions_raw:
        m = session_sequence_metrics(df)
        rows.append({"sessionID": sid, **m})

    if not rows:
        empty_df = pd.DataFrame()
        seqs = sessions_to_sequences(sessions_raw)
        bi = ngram_counts(seqs, n=2, top=30)
        tri = ngram_counts(seqs, n=3, top=30)
        acts, P = transition_matrix(sessions_raw)
        trans_b64 = transition_heatmap_b64(acts, P)
        return {
            "per_session": empty_df,
            "bigrams": bi,
            "trigrams": tri,
            "trans_b64": trans_b64,
            "cluster_summary": pd.DataFrame()
        }

    sess_df = pd.DataFrame(rows)

    # ---- soglie data-driven (quantili) ----
    def q_pair(col, low=0.2, high=0.8):
        s = sess_df[col].dropna()
        if s.empty:
            return np.nan, np.nan
        return float(s.quantile(low)), float(s.quantile(high))

    t2p_rate_low, t2p_rate_high = q_pair("touch_to_play_rate", 0.2, 0.8)
    mt2p_low, mt2p_high = q_pair("median_touch_to_play_sec", 0.2, 0.8)

    _, pause_high = q_pair("pause_per_min", 0.2, 0.8)
    _, tele_high = q_pair("teleport_per_min", 0.2, 0.8)
    _, idle_high = q_pair("idle_gaps_gt60", 0.2, 0.8)
    _, ent_high = q_pair("entropy_next_after_touch", 0.2, 0.8)

    pause_med = sess_df["pause_per_min"].median(skipna=True)
    tele_med = sess_df["teleport_per_min"].median(skipna=True)
    ent_med = sess_df["entropy_next_after_touch"].median(skipna=True)

    eff_list = []
    prob_list = []
    score_list = []

    for _, row in sess_df.iterrows():
        ef = []
        pb = []
        score = 0

        r = row["touch_to_play_rate"]
        if not np.isnan(r):
            if not np.isnan(t2p_rate_high) and r >= t2p_rate_high:
                ef.append("High Touch→Play consistency (among the best)")
                score += 1
            if not np.isnan(t2p_rate_low) and r <= t2p_rate_low:
                pb.append("Low Touch→Play consistency (among the worst)")
                score -= 1

        mt = row["median_touch_to_play_sec"]
        if not np.isnan(mt):
            if not np.isnan(mt2p_low) and mt <= mt2p_low:
                ef.append("Fast Touch→Play time (among the shortest)")
                score += 1
            if not np.isnan(mt2p_high) and mt >= mt2p_high:
                pb.append("Slow Touch→Play time (among the longest)")
                score -= 1

        pp = row["pause_per_min"]
        if not np.isnan(pp):
            if not np.isnan(pause_med) and pp <= pause_med:
                ef.append("Limited Pause usage (below the median)")
                score += 1
            if not np.isnan(pause_high) and pp >= pause_high:
                pb.append("High Pause usage (among the highest)")
                score -= 1

        tp = row["teleport_per_min"]
        if not np.isnan(tp):
            if not np.isnan(tele_med) and tp <= tele_med:
                ef.append("Moderate Teleport usage (below the median)")
                score += 1
            if not np.isnan(tele_high) and tp >= tele_high:
                pb.append("Frequent Teleport usage (among the highest)")
                score -= 1

        ig = row["idle_gaps_gt60"]
        if not np.isnan(ig):
            if ig == 0:
                ef.append("No long inactive period (>60s)")
                score += 1
            if not np.isnan(idle_high) and ig >= idle_high and ig > 0:
                pb.append("Many long inactive periods (>60s)")
                score -= 1

        ent = row["entropy_next_after_touch"]
        if not np.isnan(ent):
            if not np.isnan(ent_med) and ent <= ent_med:
                ef.append("Sequenze dopo Touch stabili (bassa entropia)")
                score += 1
            if not np.isnan(ent_high) and ent >= ent_high:
                pb.append("Sequenze dopo Touch instabili (entropia alta)")
                score -= 1

        eff_list.append("; ".join(ef))
        prob_list.append("; ".join(pb))
        score_list.append(score)

    sess_df["effective_flags"] = eff_list
    sess_df["problem_flags"] = prob_list
    sess_df["pattern_score"] = score_list

    # merge with static features
    sess_df = sess_df.merge(feats_df, on="sessionID", how="left")

    # link cluster
    labels_df = pd.DataFrame({
        "sessionID": feats_df["sessionID"].values,
        "cluster": clust_payload["best_labels"]
    })
    sess_df = sess_df.merge(labels_df, on="sessionID", how="left")


    # assegna archetipi
    sess_df = assign_archetypes(sess_df)

    # summary by cluster (main sequence metrics)
    metrics_cols = [
        "touch_to_play_rate",
        "median_touch_to_play_sec",
        "pause_per_min",
        "teleport_per_min",
        "idle_gaps_gt60",
        "entropy_next_after_touch",
        "pattern_score",
    ]
    clust_summary = pd.DataFrame()
    if "cluster" in sess_df.columns:
        valid = sess_df[(sess_df["cluster"].notna()) & (sess_df["cluster"] != -1)]
        if not valid.empty:
            cols_exist = [c for c in metrics_cols if c in valid.columns]
            if cols_exist:
                clust_summary = valid[["cluster"] + cols_exist] \
                    .groupby("cluster")[cols_exist] \
                    .agg(["mean", "median"])

    # bigrams / trigrams / transition matrix
    seqs = sessions_to_sequences(sessions_raw)
    bi = ngram_counts(seqs, n=2, top=30)
    tri = ngram_counts(seqs, n=3, top=30)

    def tag_bigram(s):
        if "Touch → PlayAnnotation" in s:
            return "useful"
        if "Pause → Pause" in s or "Teleport → Teleport" in s:
            return "critical"
        return ""

    if not bi.empty:
        bi["tag"] = bi["ngram"].apply(tag_bigram)

    acts, P = transition_matrix(sessions_raw)
    trans_b64 = transition_heatmap_b64(acts, P)

    return {
        "per_session": sess_df,
        "bigrams": bi,
        "trigrams": tri,
        "trans_b64": trans_b64,
        "cluster_summary": clust_summary
    }

# -----------------------
# Per-session summary
# -----------------------
def session_summary(df, pat_row=None):
    t_start = df["Timestamp"].iloc[0]
    t_end = df["Timestamp"].iloc[-1]
    dur = (t_end - t_start).total_seconds()

    counts_all = df["Action"].astype(str).value_counts()

    touch_df = df[df["Action"] == "Touch"]
    unique_concepts = int(touch_df["ActionID"].nunique()) if not touch_df.empty else 0
    top_concepts = touch_df["ActionID"].value_counts().head(5)

    avg_tp = avg_touch_to_next_play_seconds(df)
    seq_m = session_sequence_metrics(df)

    pat_info = {}
    if pat_row is not None and not isinstance(pat_row, dict):
        pat_row = pat_row.to_dict()
    if pat_row is not None:
        keys_from_pat = [
            "touch_to_play_rate",
            "median_touch_to_play_sec",
            "pause_per_min",
            "teleport_per_min",
            "idle_gaps_gt60",
            "entropy_next_after_touch",
            "max_pause_streak",
            "max_teleport_streak",

            "match_attempts_total",
            "match_correct",
            "match_wrong",
            "match_accuracy",

            "conceptsolve_count",
            "conceptsolve_mean_time",
            "conceptsolve_mean_attempts",
            "conceptsolve_first_try",
            "conceptsolve_multi_try",

            "score_tot_solved",
            "score_tot_attempts",
            "score_tot_accuracy",
            "score_percent",
        ]
        pat_info = {k: pat_row.get(k, np.nan) for k in keys_from_pat}
        pat_info.update({
            "pattern_score": pat_row.get("pattern_score", np.nan),
            "effective_flags": pat_row.get("effective_flags", ""),
            "problem_flags": pat_row.get("problem_flags", ""),
            "cluster": pat_row.get("cluster", np.nan),
            "archetype_primary": pat_row.get("archetype_primary", "Unclassified"),
            "archetypes": pat_row.get("archetypes", ""),
        })

    return {
        "t_start": t_start,
        "t_end": t_end,
        "duration_sec": float(dur),
        "duration_min": float(dur / 60.0),
        "n_events": int(len(df)),
        "counts_main": {a: int(counts_all.get(a, 0)) for a in MAIN_ACTIONS},
        "counts_all": {a: int(v) for a, v in counts_all.items()},
        "unique_concepts_touched": unique_concepts,
        "top_concepts": "; ".join([f"{k}({v})" for k, v in top_concepts.items()])
                        if not top_concepts.empty else "",
        "avg_touch_to_play_sec": float(avg_tp) if pd.notna(avg_tp) else np.nan,
        **seq_m,
        **pat_info
    }

def render_session_block(
    sid, df, sm,
    seq_dists, action_dists,
    include_details_wrapper=True
):
    def _percentile_rank(arr, x):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0 or x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(np.mean(arr <= float(x)) * 100.0)

    def _fmt_val_and_pct(val, pct, fmt="{:.3f}", unit=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        vtxt = fmt.format(val) + unit
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            return vtxt
        return f"{vtxt} (percentile {pct:.1f}°)"

    scat_b64 = session_timeline_plot(df)

    def pct_for(metric_key, val):
        return _percentile_rank(seq_dists.get(metric_key, []), val)

    dur_val = sm["duration_sec"]
    t2p_val = sm.get("touch_to_play_rate", np.nan)
    mt2p_val = sm.get("median_touch_to_play_sec", np.nan)
    pp_val = sm.get("pause_per_min", np.nan)
    tp_val = sm.get("teleport_per_min", np.nan)
    ig_val = sm.get("idle_gaps_gt60", np.nan)
    ent_val = sm.get("entropy_next_after_touch", np.nan)
    mps_val = sm.get("max_pause_streak", np.nan)
    mts_val = sm.get("max_teleport_streak", np.nan)

    dur_pct = pct_for("duration_sec", dur_val)
    t2p_pct = pct_for("touch_to_play_rate", t2p_val)
    mt2p_pct = pct_for("median_touch_to_play_sec", mt2p_val)
    pp_pct = pct_for("pause_per_min", pp_val)
    tp_pct = pct_for("teleport_per_min", tp_val)
    ig_pct = pct_for("idle_gaps_gt60", ig_val)
    ent_pct = pct_for("entropy_next_after_touch", ent_val)
    mps_pct = pct_for("max_pause_streak", mps_val)
    mts_pct = pct_for("max_teleport_streak", mts_val)

    counts_all_rows = []
    for a, v in sm["counts_all"].items():
        pct = _percentile_rank(action_dists.get(a, []), v)
        counts_all_rows.append({
            "Action": a,
            "Count": int(v),
            "Percentile": (np.nan if np.isnan(pct) else round(pct, 1))
        })
    counts_all_df = pd.DataFrame(counts_all_rows).sort_values("Count", ascending=False)
    counts_all_tbl = counts_all_df.to_html(index=False)

    counts_main_tbl = "<table><tr>" + \
        "".join([f"<th>{k}</th>" for k in sm["counts_main"].keys()]) + \
        "</tr><tr>" + \
        "".join([f"<td>{v}</td>" for v in sm["counts_main"].values()]) + \
        "</tr></table>"

    conf_sess_df = session_confusion_errors(df)
    if conf_sess_df.empty:
        confusion_tbl = "<p>(no incorrect MatchAttempt in this session)</p>"
    else:
        confusion_tbl = conf_sess_df.to_html(index=False)

    eff_flags = sm.get("effective_flags", "") or "-"
    prob_flags = sm.get("problem_flags", "") or "-"

    score_raw = sm.get("score_percent", np.nan)
    score_pct = pct_for("score_percent", score_raw)
    if pd.isna(score_raw):
        score_text = "N/A"
    elif pd.isna(score_pct):
        score_text = f"{score_raw:.1f}%"
    else:
        score_text = f"{score_raw:.1f}% (p{score_pct:.1f})"

    dur_text = _fmt_val_and_pct(dur_val, dur_pct, fmt="{:.1f}", unit=" s")
    arche_p = sm.get("archetype_primary", "Unclassified")
    arche_s = sm.get("archetypes", "") or "-"

    inner = f"""
  <p>
    <b>Start:</b> {sm["t_start"]}<br/>
    <b>End:</b> {sm["t_end"]}<br/>
    <b>Duration:</b> {dur_text}<br/>
    <b>Events:</b> {sm["n_events"]}<br/>
    <b>Cluster:</b> {sm.get("cluster", "N/A")}<br/>
    <b>Score percent:</b> {score_text}<br/>
    <b>Pattern score:</b> {sm.get("pattern_score", "N/A")}<br/>
    <b>Primary archetype:</b> {arche_p}<br/>
    <b>Archetypes:</b> {arche_s}<br/>
    <b>Unique concepts touched:</b> {sm["unique_concepts_touched"]}<br/>
    <b>Top concepts (Touch):</b> {sm["top_concepts"] or "(none)"}<br/>
    <b>Avg Touch→Play (s):</b> {("" if pd.isna(sm["avg_touch_to_play_sec"]) else f"{sm['avg_touch_to_play_sec']:.3f}")}<br/>
  </p>

  <h4>Pattern Detection Flags</h4>
  <p><b>Effective flags:</b><br/>{eff_flags}</p>
  <p><b>Problem flags:</b><br/>{prob_flags}</p>

  <h4>Sequence metrics (with percentile)</h4>
  <ul>
    <li><b>Touch→Play rate</b>: {_fmt_val_and_pct(t2p_val, t2p_pct)}</li>
    <li><b>Median Touch→Play time (s)</b>: {_fmt_val_and_pct(mt2p_val, mt2p_pct)}</li>
    <li><b>Pauses per minute</b>: {_fmt_val_and_pct(pp_val, pp_pct)}</li>
    <li><b>Teleports per minute</b>: {_fmt_val_and_pct(tp_val, tp_pct)}</li>
    <li><b>Idle gap &gt; 60s</b>: {_fmt_val_and_pct(ig_val, ig_pct, fmt="{:.0f}")}</li>
    <li><b>Entropy after Touch</b>: {_fmt_val_and_pct(ent_val, ent_pct)}</li>
    <li><b>Max Pause streak</b>: {_fmt_val_and_pct(mps_val, mps_pct, fmt="{:.0f}")}</li>
    <li><b>Max Teleport streak</b>: {_fmt_val_and_pct(mts_val, mts_pct, fmt="{:.0f}")}</li>
  </ul>

  <h4>Performance metrics (MatchAttempt / ConceptSolve / ScoreSummary)</h4>
  <ul>
    <li><b>Total MatchAttempts</b>: {sm.get("match_attempts_total", "N/A")}</li>
    <li><b>Correct / wrong matches</b>: {sm.get("match_correct", "N/A")} / {sm.get("match_wrong", "N/A")}</li>
    <li><b>MatchAttempt accuracy</b>: {("N/A" if pd.isna(sm.get("match_accuracy", np.nan)) else f"{sm.get('match_accuracy'):.3f}")}</li>
    <li><b>Recorded ConceptSolve events</b>: {sm.get("conceptsolve_count", "N/A")}</li>
    <li><b>Mean ConceptSolve time (s)</b>: {("N/A" if pd.isna(sm.get("conceptsolve_mean_time", np.nan)) else f"{sm.get('conceptsolve_mean_time'):.3f}")}</li>
    <li><b>Mean ConceptSolve attempts</b>: {("N/A" if pd.isna(sm.get("conceptsolve_mean_attempts", np.nan)) else f"{sm.get('conceptsolve_mean_attempts'):.3f}")}</li>
    <li><b>First-try / multi-try</b>: {sm.get("conceptsolve_first_try", "N/A")} / {sm.get("conceptsolve_multi_try", "N/A")}</li>
    <li><b>Total score (solved/attempts)</b>: {sm.get("score_tot_solved", "N/A")} / {sm.get("score_tot_attempts", "N/A")}</li>
    <li><b>Total score accuracy</b>: {("N/A" if pd.isna(sm.get("score_tot_accuracy", np.nan)) else f"{sm.get('score_tot_accuracy'):.3f}")}</li>
    <li><b>Final percentage</b>: {("N/A" if pd.isna(sm.get("score_percent", np.nan)) else f"{sm.get('score_percent'):.1f}%")}</li>
  </ul>

  <h4>Concept confusion errors (incorrect MatchAttempts only)</h4>
  <div class="table-scroll">{confusion_tbl}</div>

  <h4>Main counts</h4>
  {counts_main_tbl}

  <h4>All action counts (with percentile)</h4>
  <div class="table-scroll">{counts_all_tbl}</div>

  <div class="figure">
    <h4>Event timeline</h4>
    <img src="{scat_b64}" alt="timeline scatter"/>
  </div>
"""

    if include_details_wrapper:
        return f"""
<details>
<summary>
  <b>Session:</b> <code>{sid}</code>
  — duration: {dur_val:.1f} s
  — events: {sm["n_events"]}
  — cluster: {sm.get("cluster", "N/A")}
  — score_percent: {score_text}
  — pattern_score: {sm.get("pattern_score", "N/A")}
  — archetype: {arche_p}
</summary>
{inner}
</details>
"""
    else:
        return inner

def build_per_session_section(sessions_raw, pat_df=None):
    blocks = []
    sessions_raw_sorted = sorted(sessions_raw, key=lambda x: x[0])

    pat_map = {}
    if pat_df is not None and not pat_df.empty and "sessionID" in pat_df.columns:
        pat_map = {r["sessionID"]: r for _, r in pat_df.iterrows()}

    seq_metric_keys = [
        "duration_sec",
        "touch_to_play_rate",
        "median_touch_to_play_sec",
        "pause_per_min",
        "teleport_per_min",
        "idle_gaps_gt60",
        "entropy_next_after_touch",
        "max_pause_streak",
        "max_teleport_streak",
        "score_percent",
    ]
    seq_dists = {k: [] for k in seq_metric_keys}
    if pat_df is not None and not pat_df.empty:
        for k in seq_metric_keys:
            if k in pat_df.columns:
                seq_dists[k] = pat_df[k].astype(float).tolist()

    all_actions = set()
    counts_per_session = []
    for sid, df in sessions_raw_sorted:
        c = df["Action"].astype(str).value_counts().to_dict()
        counts_per_session.append((sid, c))
        all_actions.update(c.keys())

    all_actions = sorted(all_actions)
    action_dists = {a: [] for a in all_actions}
    for _, cdict in counts_per_session:
        for a in all_actions:
            action_dists[a].append(float(cdict.get(a, 0)))

    for sid, df in sessions_raw_sorted:
        pat_row = pat_map.get(sid, None)
        sm = session_summary(df, pat_row=pat_row)

        blocks.append(
            render_session_block(
                sid=sid,
                df=df,
                sm=sm,
                seq_dists=seq_dists,
                action_dists=action_dists,
                include_details_wrapper=True
            )
        )

    return "\n".join(blocks)

def save_single_session_reports(sessions_raw, pat_df, out_dir):
    session_reports_dir = Path(out_dir) / "SessionReports"
    session_reports_dir.mkdir(parents=True, exist_ok=True)

    sessions_raw_sorted = sorted(sessions_raw, key=lambda x: x[0])

    pat_map = {}
    if pat_df is not None and not pat_df.empty and "sessionID" in pat_df.columns:
        pat_map = {r["sessionID"]: r for _, r in pat_df.iterrows()}

    seq_metric_keys = [
        "duration_sec",
        "touch_to_play_rate",
        "median_touch_to_play_sec",
        "pause_per_min",
        "teleport_per_min",
        "idle_gaps_gt60",
        "entropy_next_after_touch",
        "max_pause_streak",
        "max_teleport_streak",
        "score_percent",
    ]
    seq_dists = {k: [] for k in seq_metric_keys}
    if pat_df is not None and not pat_df.empty:
        for k in seq_metric_keys:
            if k in pat_df.columns:
                seq_dists[k] = pat_df[k].astype(float).tolist()

    all_actions = set()
    counts_per_session = []
    for sid, df in sessions_raw_sorted:
        c = df["Action"].astype(str).value_counts().to_dict()
        counts_per_session.append((sid, c))
        all_actions.update(c.keys())

    all_actions = sorted(all_actions)
    action_dists = {a: [] for a in all_actions}
    for _, cdict in counts_per_session:
        for a in all_actions:
            action_dists[a].append(float(cdict.get(a, 0)))

    base_head = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Session Report</title>
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #222; }
h1,h2,h3,h4 { margin: .4em 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }
th { background: #fafafa; }
.figure { margin: 14px 0; }
img { max-width: 100%; height: auto; border: 1px solid #eee; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#f6f8fa; padding: 2px 6px; border-radius: 6px; }
.small { font-size: 12px; color:#666; }
.table-scroll {
  max-height: 400px;
  overflow: auto;
  border: 1px solid #ddd;
  padding: 4px;
  background: #fff;
  margin-top: 6px;
  margin-bottom: 10px;
}
.table-scroll table { font-size: 12px; }
</style>
</head>
<body>
"""

    base_footer = """
<footer><p class="small">Automatically generated.</p></footer>
</body>
</html>
"""

    for sid, df in sessions_raw_sorted:
        pat_row = pat_map.get(sid, None)
        sm = session_summary(df, pat_row=pat_row)

        content_html = render_session_block(
            sid=sid,
            df=df,
            sm=sm,
            seq_dists=seq_dists,
            action_dists=action_dists,
            include_details_wrapper=False
        )

        page = base_head + f"""
<h1>Session Report: <code>{sid}</code></h1>
{content_html}
""" + base_footer

        out_path = session_reports_dir / f"{sid}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page)

    print("[OK] Per-session reports saved in:", session_reports_dir.resolve())



# -----------------------
# Concept Timeline Diagnostics (cross-session)
# -----------------------
def _play_concepts_from_last_touch(df):
    """
    Assigns each PlayAnnotation to the most recently touched concept (ActionID of the latest Touch).
    Returns a list (same length as PlayAnnotation rows) with concept_id (string) or "".
    """
    last = ""
    out = []
    for _, r in df.iterrows():
        a = str(r.get("Action", ""))
        if a == "Touch":
            cid = str(r.get("ActionID", "")).strip()
            if cid:
                last = cid
        elif a == "PlayAnnotation":
            out.append(last)
    return out


def concept_timeline_diagnostics(sessions_raw, top_n=15):
    """
    Concept diagnostics (cross-session) — versione allineata a:
      - play_max_streak: max PlayAnnotation consecutivi sullo stesso concept (stimato via last Touch)
      - match_max_streak: max consecutive MatchAttempt events with the same target concept
      - solve_mean_s: tempo medio ConceptSolve del concept
      - match_accuracy: accuracy aggregata sui MatchAttempt del concept (target)

    Ranking (problem_score) with equal weights on:
      - solve_mean_s (alto = peggio)
      - 1 - match_accuracy (alto = peggio)
      - play_max_streak (alto = peggio)
      - match_max_streak (alto = peggio)

    Output:
      dict con:
        concept_df (ranking),
        worst_sessions_by_concept (dict)
    """

    def _max_run_len(seq):
        """Maximum run length of equal consecutive values (ignores '' / None)."""
        best = 0
        curr = 0
        prev = None
        for x in seq:
            if x is None:
                x = ""
            x = str(x).strip()
            if x == "":
                curr = 0
                prev = None
                continue
            if x == prev:
                curr += 1
            else:
                prev = x
                curr = 1
            best = max(best, curr)
        return int(best)

    per_session_rows = []

    for sid, df in sessions_raw:
        if df is None or df.empty:
            continue

        # -------------------------
        # PlayAnnotation
        # -------------------------
        play_df = df[df["Action"] == "PlayAnnotation"].copy()
        play_streak_by_concept = {}
        if not play_df.empty:
            play_concepts = _play_concepts_from_last_touch(df) 
            best = {}
            prev = None
            curr = 0
            for c in play_concepts:
                c = str(c).strip() if c is not None else ""
                if c == "":
                    prev = None
                    curr = 0
                    continue
                if c == prev:
                    curr += 1
                else:
                    prev = c
                    curr = 1
                best[c] = max(best.get(c, 0), curr)
            play_streak_by_concept = {k: int(v) for k, v in best.items()}

        # -------------------------
        # MatchAttempt target streak
        # -------------------------
        ma = df[df["Action"] == "MatchAttempt"].copy()
        match_streak_by_concept = {}
        ma_by_target = {}
        if not ma.empty:
            ma["target"] = ma["Field1"].astype(str).str.strip()
            ma = ma[ma["target"] != ""].copy()

            # streak by concept (target)
            targets_seq = ma["target"].astype(str).tolist()
            best = {}
            prev = None
            curr = 0
            for t in targets_seq:
                t = str(t).strip()
                if t == "":
                    prev = None
                    curr = 0
                    continue
                if t == prev:
                    curr += 1
                else:
                    prev = t
                    curr = 1
                best[t] = max(best.get(t, 0), curr)
            match_streak_by_concept = {k: int(v) for k, v in best.items()}

            # aggregated accuracy by target
            outcome = ma["Field3"].astype(str).str.strip().str.lower()
            correct_mask = outcome.isin(["true", "1", "t", "yes"])
            ma["is_correct"] = correct_mask.astype(int)

            grp = ma.groupby("target").agg(
                match_total=("target", "size"),
                match_correct=("is_correct", "sum"),
            ).reset_index()

            for _, r in grp.iterrows():
                tgt = str(r["target"]).strip()
                mt = int(r["match_total"])
                mc = int(r["match_correct"])
                mw = int(mt - mc)
                ma_by_target[tgt] = (mt, mw)

        # -------------------------
        # ConceptSolve times by concept
        # -------------------------
        cs = df[df["Action"] == "ConceptSolve"].copy()
        cs_by_concept = {}
        if not cs.empty:
            cs["concept"] = cs["Field1"].astype(str).str.strip()
            cs["solve_time"] = pd.to_numeric(cs["Field4"], errors="coerce")
            cs = cs.dropna(subset=["solve_time"])
            cs = cs[cs["concept"] != ""]
            if not cs.empty:
                g = cs.groupby("concept")["solve_time"].agg(["count", "mean"]).reset_index()
                cs_by_concept = {str(r["concept"]).strip(): (int(r["count"]), float(r["mean"])) for _, r in g.iterrows()}

        # -------------------------
        # Union of concepts present in the session
        # -------------------------
        all_concepts = set(play_streak_by_concept) | set(match_streak_by_concept) | set(ma_by_target) | set(cs_by_concept)

        for c in all_concepts:
            mt, mw = ma_by_target.get(c, (0, 0))
            macc = (1.0 - (mw / mt)) if mt > 0 else np.nan

            cs_cnt, cs_mean = cs_by_concept.get(c, (0, np.nan))

            per_session_rows.append({
                "sessionID": sid,
                "concept": c,
                "solve_count": int(cs_cnt),
                "solve_mean_s": float(cs_mean) if not pd.isna(cs_mean) else np.nan,

                "match_total": int(mt),
                "match_wrong": int(mw),
                "match_accuracy": float(macc) if not pd.isna(macc) else np.nan,

                "play_max_streak": int(play_streak_by_concept.get(c, 0)),
                "match_max_streak": int(match_streak_by_concept.get(c, 0)),
            })

    if not per_session_rows:
        empty = pd.DataFrame(columns=[
            "concept", "n_sessions",
            "solve_mean_s", "match_accuracy",
            "play_max_streak", "match_max_streak",
            "problem_score"
        ])
        return {
            "concept_df": empty,
            "worst_sessions_by_concept": {}
        }

    ps = pd.DataFrame(per_session_rows)

    # -------------------------
    # Aggregation by concept
    # -------------------------
    agg = ps.groupby("concept").agg(
        n_sessions=("sessionID", "nunique"),

        # solve: mean of session means 
        solve_mean_s=("solve_mean_s", "mean"),

        # match accuracy: aggregated across all attempts
        match_total=("match_total", "sum"),
        match_wrong=("match_wrong", "sum"),

        # streak: maximum observed in any session
        play_max_streak=("play_max_streak", "max"),
        match_max_streak=("match_max_streak", "max"),
    ).reset_index()

    agg["match_accuracy"] = np.where(
        agg["match_total"] > 0,
        1.0 - (agg["match_wrong"] / agg["match_total"]),
        np.nan
    )

    # -------------------------
    # Problem score: pesi uguali
    # -------------------------
    def qnorm(s):
        s = pd.to_numeric(s, errors="coerce")
        r = s.rank(pct=True)
        return r.fillna(0.0)

    q_solve = qnorm(agg["solve_mean_s"])  # alto = peggio
    q_inacc = qnorm(1.0 - agg["match_accuracy"])  # alto = peggio
    q_play_streak = qnorm(agg["play_max_streak"])  # alto = peggio
    q_match_streak = qnorm(agg["match_max_streak"])  # alto = peggio

    agg["problem_score"] = (q_solve + q_inacc + q_play_streak + q_match_streak) / 4.0

    # ordinamento
    concept_df = agg.sort_values("problem_score", ascending=False).reset_index(drop=True)

    # rounding
    for c in ["solve_mean_s", "match_accuracy", "problem_score"]:
        if c in concept_df.columns:
            concept_df[c] = pd.to_numeric(concept_df[c], errors="coerce").round(3)

    # -------------------------
    # Worst sessions by concept
    # -------------------------
    worst_sessions_by_concept = {}
    top_concepts = concept_df.head(min(top_n, len(concept_df)))["concept"].tolist()

    for c in top_concepts:
        tmp = ps[ps["concept"] == c].copy()

        # 0..1 normalization within each concept to make scales comparable
        def _minmax(col, invert=False):
            v = pd.to_numeric(tmp[col], errors="coerce").fillna(np.nan).values.astype(float)
            if invert:
                v = 1.0 - v
            v_nanmask = np.isnan(v)
            if np.all(v_nanmask):
                return np.zeros(len(v), dtype=float)
            vv = v.copy()
            vv[v_nanmask] = np.nanmin(vv[~v_nanmask])
            mn = np.min(vv)
            mx = np.max(vv)
            if mx <= mn:
                return np.zeros(len(vv), dtype=float)
            return (vv - mn) / (mx - mn)

        # indicatori allineati (tutti "alto = peggio")
        s_solve = _minmax("solve_mean_s", invert=False)
        s_inacc = _minmax("match_accuracy", invert=True) 
        s_play = _minmax("play_max_streak", invert=False)
        s_match = _minmax("match_max_streak", invert=False)

        tmp["impact"] = (s_solve + s_inacc + s_play + s_match) / 4.0

        tmp = tmp.sort_values("impact", ascending=False).head(10).reset_index(drop=True)

        worst_sessions_by_concept[c] = tmp[[
            "sessionID",
            "solve_mean_s",
            "match_accuracy",
            "play_max_streak",
            "match_max_streak",
            "impact"
        ]]

        for col in ["solve_mean_s", "match_accuracy", "impact"]:
            worst_sessions_by_concept[c][col] = pd.to_numeric(worst_sessions_by_concept[c][col], errors="coerce").round(3)

    return {
        "concept_df": concept_df[[
            "concept", "n_sessions",
            "solve_mean_s", "match_accuracy",
            "play_max_streak", "match_max_streak",
            "problem_score"
        ]].copy(),
        "worst_sessions_by_concept": worst_sessions_by_concept,
        "concept_session_long": ps[["sessionID", "concept"]].drop_duplicates().reset_index(drop=True)
    }

def concept_event_long_from_logs(sessions_raw):
    """
    Builds a long table for (sessionID, concept) using:
      - PlayAnnotation attribuiti al concept dell'ultimo Touch
      - MatchAttempt attribuiti al target concept (Field1)

    Output columns:
      sessionID, concept, play_count, match_count, event_count
    """
    rows = []

    for sid, df in sessions_raw:
        if df is None or df.empty:
            continue

        play_concepts = _play_concepts_from_last_touch(df) 
        play_counts = Counter([str(c).strip() for c in play_concepts if str(c).strip() != ""])

        # --- MatchAttempt -> target concept ---
        ma = df[df["Action"] == "MatchAttempt"].copy()
        if not ma.empty:
            targets = ma["Field1"].astype(str).str.strip()
            targets = targets[targets != ""]
            match_counts = Counter(targets.tolist())
        else:
            match_counts = Counter()

        # Union of concepts seen in that session
        concepts = set(play_counts.keys()) | set(match_counts.keys())
        for c in concepts:
            pc = int(play_counts.get(c, 0))
            mc = int(match_counts.get(c, 0))
            rows.append({
                "sessionID": sid,
                "concept": c,
                "play_count": pc,
                "match_count": mc,
                "event_count": pc + mc
            })

    if not rows:
        return pd.DataFrame(columns=["sessionID", "concept", "play_count", "match_count", "event_count"])

    return pd.DataFrame(rows)


# -----------------------
# REPORT HTML
# -----------------------
def build_html_report(
    feats_df, eda_payload, fs_X, fs_kept, fs_drop_tbl,
    clust_payload, seq_payload, sessions_raw, pat
):
    head_html = feats_df.head(10).to_html(index=False)

    kept_list = ", ".join(fs_kept) if fs_kept else "(tutte)"
    dropped_html = fs_drop_tbl.to_html(index=False, classes="drop-log") \
        if not fs_drop_tbl.empty else "<p>No dropped features (variance/correlation).</p>"

    res_html = clust_payload["res_df"].to_html(index=False)
    pca_b64 = pca_scatter_for_labels(
        clust_payload["Xs"],
        clust_payload["best_labels"],
        title=f"PCA scatter — {clust_payload['best_name']}"
    )

    top_next_html = "<p>No additional model available.</p>"
    try:
        res_df = clust_payload["res_df"]
        next4 = res_df.iloc[1:5].copy()
        imgs = []
        labels_by_model = clust_payload.get("labels_by_model", {})
        for _, r in next4.iterrows():
            mname = r["model"]
            lab = labels_by_model.get(mname, None)
            if lab is None:
                continue
            b64 = pca_scatter_for_labels(
                clust_payload["Xs"],
                lab,
                title=f"PCA scatter — {mname}"
            )
            if b64:
                imgs.append((mname, b64))
        if imgs:
            cards = []
            for title, b64 in imgs:
                cards.append(f"""
                <div style="flex:1; min-width:260px; border:1px solid #eee; padding:10px; border-radius:8px; background:#fafafa;">
                  <h4 style="margin-top:0;">{title}</h4>
                  <img src="{b64}" alt="PCA {title}"/>
                </div>
                """)
            top_next_html = f"""
            <div style="display:flex; flex-wrap:wrap; gap:12px;">
              {''.join(cards)}
            </div>
            """
    except Exception:
        pass

    bigrams_html = seq_payload["bigrams"].to_html(index=False) \
        if not seq_payload["bigrams"].empty else "<p>No bigrams.</p>"
    trigrams_html = seq_payload["trigrams"].to_html(index=False) \
        if not seq_payload["trigrams"].empty else "<p>No trigrams.</p>"
    trans_b64 = seq_payload["trans_b64"]
    dwell_html = seq_payload["dwell_df"].to_html(index=False) \
        if not seq_payload["dwell_df"].empty else "<p>No computable dwell time.</p>"

    error_mat_html = "<p>No MatchAttempt errors detected.</p>"
    if "error_mat_df" in seq_payload and isinstance(seq_payload["error_mat_df"], pd.DataFrame):
        if not seq_payload["error_mat_df"].empty:
            error_mat_html = seq_payload["error_mat_df"].to_html()
    error_mat_b64 = seq_payload.get("error_mat_b64", "")

    if "confusion_df" in seq_payload and isinstance(seq_payload["confusion_df"], pd.DataFrame):
        conf_df = seq_payload["confusion_df"]
        if not conf_df.empty:
            confusion_html = conf_df.head(50).to_html(index=False)
        else:
            confusion_html = "<p>No MatchAttempt errors detected.</p>"
    else:
        confusion_html = "<p>No MatchAttempt errors detected.</p>"

    profiles_html = "<p>No profile available.</p>"
    try:
        Ximp = pd.DataFrame(clust_payload["X_imp"], columns=fs_X.columns)
        prof = Ximp.copy()
        prof["cluster"] = clust_payload["best_labels"]
        if np.any(prof["cluster"] != -1) and fs_X.shape[1] > 0:
            profiles = prof[prof["cluster"] != -1].groupby("cluster").agg(["mean", "median"]).T
            if not profiles.empty:
                profiles.to_csv(PROFILES_CSV)
                profiles_html = profiles.head(40).to_html()
    except Exception:
        profiles_html = "<p>Profili non disponibili (dimensioni non allineate).</p>"

    feats_df.to_csv(FEATURES_CSV, index=False)
    clust_payload["res_df"].to_csv(RES_CSV, index=False)
    pd.DataFrame({
        "sessionID": feats_df["sessionID"].values,
        "cluster": clust_payload["best_labels"]
    }).to_csv(ASSIGN_CSV, index=False)

    per_session_html = build_per_session_section(
        sessions_raw,
        pat_df=pat["per_session"]
    )

    pat_sess_df = pat["per_session"].copy()

    # -----------------------
    # NEW: Concept Timeline Diagnostics (cross-session)
    # -----------------------
    concept_diag = concept_timeline_diagnostics(sessions_raw, top_n=15)

    concept_df = concept_diag["concept_df"]
    worst_by_concept = concept_diag["worst_sessions_by_concept"]

    # NEW: long mapping concept <-> session
    concept_session_long = concept_diag.get(
        "concept_session_long",
        pd.DataFrame(columns=["sessionID", "concept"])
    )



    concept_cluster_html = "<p>Clustering concept non disponibile.</p>"

    try:
        if concept_df is not None and not concept_df.empty:
            ccl = best_of_five_concept_clustering(concept_df, k_fixed=3)

            # add cluster label to concepts
            concept_df = concept_df.copy()
            concept_df["concept_cluster"] = ccl["best_labels"]

            concept_cluster_metrics_html = ccl["res_df"].to_html(index=False)

            concept_cluster_pca_b64 = pca_scatter_for_labels(
                ccl["Xs"], ccl["best_labels"],
                title=f"Concept PCA scatter — {ccl['best_name']}"
            )

            concept_cluster_html = f"""
            <p><b>Best concept model:</b> <code>{ccl['best_name']}</code></p>

            <h4>PCA scatter (concept)</h4>
            <div class="figure">
              <img src="{concept_cluster_pca_b64}" alt="Concept PCA scatter"/>
            </div>

            <h4>Metriche modelli (k=3)</h4>
            {concept_cluster_metrics_html}
            """
    except Exception:
        pass

    concept_tbl_html = concept_df.head(30).to_html(index=False) if not concept_df.empty else "<p>No diagnosable concept.</p>"

    # -----------------------
    # Concept-Cluster × SessionCluster matrix (PlayAnnotation + MatchAttempt)
    # -----------------------
    conc_clust_vs_sess_clust_html = "<p>Matrix not available.</p>"

    try:
        # session -> session_cluster
        sess_clusters = pat_sess_df[["sessionID", "cluster"]].copy() \
            if ("sessionID" in pat_sess_df.columns and "cluster" in pat_sess_df.columns) \
            else pd.DataFrame()

        # concept -> concept_cluster
        conc_clusters = concept_df[["concept", "concept_cluster"]].copy() \
            if (concept_df is not None and not concept_df.empty and "concept_cluster" in concept_df.columns) \
            else pd.DataFrame()

        if not sess_clusters.empty and not conc_clusters.empty:

            # (sessionID, concept) with Play and Match counts
            event_long = concept_event_long_from_logs(sessions_raw)

            if not event_long.empty:
                tmp = (event_long
                    .merge(conc_clusters, on="concept", how="left")
                    .merge(sess_clusters.rename(columns={"cluster": "session_cluster"}), on="sessionID", how="left")
                )

                tmp = tmp.dropna(subset=["concept_cluster", "session_cluster"]).copy()
                tmp["concept_cluster"] = tmp["concept_cluster"].astype(int)
                tmp["session_cluster"] = tmp["session_cluster"].astype(int)

                # row label: concept-name + cluster
                tmp["concept_label"] = tmp["concept"] + "-" + tmp["concept_cluster"].astype(str)

                pres = tmp[tmp["event_count"] > 0].copy()

                presence_matrix = pd.pivot_table(
                    pres,
                    index="concept_label",
                    columns="session_cluster",
                    values="sessionID",
                    aggfunc=pd.Series.nunique,   # number of distinct sessions
                    fill_value=0
                )

                presence_matrix = presence_matrix.reindex(sorted(presence_matrix.columns), axis=1)
                presence_matrix = presence_matrix.sort_index()
                presence_matrix.columns.name = None

                presence_html = presence_matrix.to_html()

                # header "Session Cluster"
                presence_html = presence_html.replace(
                    "<thead>",
                    "<thead><tr><th></th><th colspan=\"{}\" style=\"text-align:center;\">Session Cluster</th></tr>".format(len(presence_matrix.columns))
                )

                tmp["event_count"] = pd.to_numeric(tmp["event_count"], errors="coerce").fillna(0)

                intensity_matrix = pd.pivot_table(
                    tmp,
                    index="concept_label",
                    columns="session_cluster",
                    values="event_count",
                    aggfunc="mean",
                    fill_value=0
                )

                intensity_matrix = intensity_matrix.reindex(sorted(intensity_matrix.columns), axis=1)
                intensity_matrix = intensity_matrix.sort_index()
                intensity_matrix = intensity_matrix.round(3)
                intensity_matrix.columns.name = None

                intensity_html = intensity_matrix.to_html()

                # header "Session Cluster"
                intensity_html = intensity_html.replace(
                    "<thead>",
                    "<thead><tr><th></th><th colspan=\"{}\" style=\"text-align:center;\">Session Cluster</th></tr>".format(len(intensity_matrix.columns))
                )

                conc_clust_vs_sess_clust_html = f"""
                <h3>PRESENCE matrix (PlayAnnotation + MatchAttempt): Concept-Cluster × SessionCluster</h3>
                <p class="small">
                  Rows = concept with its cluster (Name-Cluster).<br>
                  Columns = session cluster.<br>
                  Value = number of sessions in the cluster where that concept has at least 1 event
                  (attributed PlayAnnotation or MatchAttempt as target).
                </p>
                <div class="table-scroll">
                  {presence_html}
                </div>

                <h3>Mean INTENSITY matrix (Play+Match per session): Concept-Cluster × SessionCluster</h3>
                <p class="small">
                  Value = mean of (PlayAnnotation + MatchAttempt) per session, within the session cluster.<br><br>
                  For a given concept and a given cluster:<br>
                  – for each session in the cluster count: (#PlayAnnotation attributed to the concept) + (#MatchAttempt with the concept as target)<br>
                  – compute the mean across all sessions in the cluster.
                </p>
                <div class="table-scroll">
                  {intensity_html}
                </div>
                """
    except Exception:
        pass


    concept_cols_explain_html = """
      <h3>Concept ranking</h3>
      <p class="small">
        Each row aggregates a <b>concept</b> across all sessions. The columns represent:
        <ul>
          <li><b>n_sessions</b>: number of sessions in which the concept appears.</li>
          <li><b>solve_mean_s</b>: mean concept resolution time (from <code>ConceptSolve</code>), in seconds.</li>
          <li><b>match_accuracy</b>: accuracy on <code>MatchAttempt</code> where the concept is the target (1 − wrong/total).</li>
          <li><b>play_max_streak</b>: maximum number of consecutive <code>PlayAnnotation</code> attributed to the concept.</li>
          <li><b>match_max_streak</b>: maximum number of consecutive <code>MatchAttempt</code> on the same target concept.</li>
          <li><b>problem_score</b>: ranking index based on 4 indicators:
            <code>solve_mean_s</code>, <code>1−match_accuracy</code>, <code>play_max_streak</code>, <code>match_max_streak</code>.
          </li>
        </ul>
      </p>
    """

    worst_blocks = []
    for c, dfx in worst_by_concept.items():
        # show the concept cluster in the summary
        c_cluster = "N/A"
        try:
            if concept_df is not None and not concept_df.empty and "concept" in concept_df.columns and "concept_cluster" in concept_df.columns:
                tmpc = concept_df.loc[concept_df["concept"] == c, "concept_cluster"]
                if not tmpc.empty:
                    c_cluster = int(tmpc.iloc[0]) if pd.notna(tmpc.iloc[0]) else "N/A"
        except Exception:
            pass

        worst_blocks.append(f"""
<details>
  <summary><b>Sessions contributing the most:</b> <code>{c}</code> — <b>concept_cluster:</b> {c_cluster} (top 10 by impact)</summary>
  <div class="table-scroll">
    {dfx.to_html(index=False)}
  </div>
</details>
""")

    worst_sessions_html = "\n".join(worst_blocks) if worst_blocks else "<p>No per-session detail available.</p>"

    # -----------------------
    # Archetype tables (summary + per-session)
    # -----------------------
    arch_summary_html = "<p>No archetype data available.</p>"
    arch_sess_html = "<p>No archetype data available.</p>"

    if not pat_sess_df.empty and "archetype_primary" in pat_sess_df.columns:
        # Summary: count by archetype_primary (+ percentages)
        arch_sum = (
            pat_sess_df["archetype_primary"]
            .fillna("Unclassified")
            .value_counts(dropna=False)
            .rename_axis("archetype_primary")
            .reset_index(name="n_sessions")
        )
        arch_sum["pct"] = (100.0 * arch_sum["n_sessions"] / max(len(pat_sess_df), 1)).round(1)
        arch_summary_html = arch_sum.to_html(index=False)

        # Per-session table
        arch_cols = [
            c for c in [
                "sessionID",
                "cluster",
                "score_percent",
                "pattern_score",
                "archetype_primary",
                "archetypes",
            ]
            if c in pat_sess_df.columns
        ]
        arch_sess_html = pat_sess_df[arch_cols].to_html(index=False, escape=False)


    pat_summary_html = "<p>No Pattern Detection data available.</p>"
    pat_hist_b64 = ""
    top_worst_html = "<p>N/A</p>"
    top_best_html = "<p>N/A</p>"

    def _score_to_class(score):
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return "badge-gray"
        try:
            s = float(score)
        except Exception:
            return "badge-gray"
        if s >= 2:
            return "badge-green"
        elif s >= 0:
            return "badge-yellow"
        else:
            return "badge-red"

    def _color_session_id(sid, score):
        cls = _score_to_class(score)
        return f'<span class="{cls}" style="background:transparent;border:none;padding:0;font-weight:700;">{sid}</span>'

    if not pat_sess_df.empty and "pattern_score" in pat_sess_df.columns:
        pat_hist_b64 = pattern_score_hist_b64(pat_sess_df)

        n_tot = len(pat_sess_df)
        n_good = int((pat_sess_df["pattern_score"] >= 2).sum())
        n_mid = int(((pat_sess_df["pattern_score"] >= 0) &
                     (pat_sess_df["pattern_score"] < 2)).sum())
        n_bad = int((pat_sess_df["pattern_score"] < 0).sum())

        show_cols = [
            "sessionID",
            "archetype_primary",
            "archetypes",
            "score_percent",
            "pattern_score",
            "effective_flags",
            "problem_flags",
            "touch_to_play_rate",
            "median_touch_to_play_sec",
            "pause_per_min",
            "teleport_per_min",
            "idle_gaps_gt60",
            "entropy_next_after_touch",
            "match_attempts_total",
            "match_correct",
            "match_wrong",
            "match_accuracy",
            "conceptsolve_count",
            "conceptsolve_mean_time",
            "conceptsolve_mean_attempts",
            "conceptsolve_first_try",
            "conceptsolve_multi_try",
            "score_tot_solved",
            "score_tot_attempts",
            "score_tot_accuracy",
            "cluster",
        ]
        show_cols = [c for c in show_cols if c in pat_sess_df.columns]

        worst5 = pat_sess_df.sort_values("pattern_score", ascending=True).head(5)[show_cols].copy()
        best5 = pat_sess_df.sort_values("pattern_score", ascending=False).head(5)[show_cols].copy()

        for df_ in (worst5, best5):
            if "effective_flags" in df_.columns:
                df_["effective_flags"] = df_["effective_flags"].fillna("").replace("", "-")
            if "problem_flags" in df_.columns:
                df_["problem_flags"] = df_["problem_flags"].fillna("").replace("", "-")
            if "archetypes" in df_.columns:
                df_["archetypes"] = df_["archetypes"].fillna("").replace("", "-")
            if "sessionID" in df_.columns:
                df_["sessionID"] = [
                    _color_session_id(sid, sc)
                    for sid, sc in zip(df_["sessionID"], df_["pattern_score"])
                ]

        top_worst_html = worst5.to_html(index=False, escape=False)
        top_best_html = best5.to_html(index=False, escape=False)

        pat_summary_html = f"""
        <ul>
          <li><b>Total analyzed sessions:</b> {n_tot}</li>
          <li><b>Linear sessions (score ≥ 2):</b> {n_good}</li>
          <li><b>Intermediate sessions (0 ≤ score &lt; 2):</b> {n_mid}</li>
          <li><b>Critical sessions (score &lt; 0):</b> {n_bad}</li>
        </ul>
        """

    if not pat_sess_df.empty:
        base_order = [
            "sessionID",
            "archetype_primary",
            "archetypes",
            "score_percent",
            "pattern_score",
            "effective_flags",
            "problem_flags",
            "touch_to_play_rate",
            "median_touch_to_play_sec",
            "pause_per_min",
            "teleport_per_min",
            "idle_gaps_gt60",
            "entropy_next_after_touch",
            "match_attempts_total",
            "match_correct",
            "match_wrong",
            "match_accuracy",
            "conceptsolve_count",
            "conceptsolve_mean_time",
            "conceptsolve_mean_attempts",
            "conceptsolve_first_try",
            "conceptsolve_multi_try",
            "score_tot_solved",
            "score_tot_attempts",
            "score_tot_accuracy",
            "cluster",
        ]
        base_order = [c for c in base_order if c in pat_sess_df.columns]
        other_cols = [c for c in pat_sess_df.columns if c not in base_order]
        pat_sess_df = pat_sess_df[base_order + other_cols]

        if "sessionID" in pat_sess_df.columns and "pattern_score" in pat_sess_df.columns:
            pat_sess_df["sessionID"] = [
                _color_session_id(sid, sc)
                for sid, sc in zip(pat_sess_df["sessionID"], pat_sess_df["pattern_score"])
            ]

    pat_sess_html = pat_sess_df.to_html(index=False, escape=False)

    pat_cluster_html = "<p>No cluster summary available.</p>"
    if "cluster_summary" in pat and isinstance(pat["cluster_summary"], pd.DataFrame):
        if not pat["cluster_summary"].empty:
            pat_cluster_html = pat["cluster_summary"].to_html()

    flags_guide_html = """
    <h3>Pattern interpretation (flag guide)</h3>
    <p>
      The table below helps interpret the flags assigned to sessions.
      Each flag describes a behavior relative to the rest of the dataset.
    </p>

    <div class="table-scroll">
    <table>
      <tr>
        <th>Flag</th>
        <th>Significato quantitativo</th>
        <th>Interpretazione comportamentale</th>
      </tr>
      <tr><td>High Touch→Play consistency</td><td>touch_to_play_rate in the highest percentiles (≈ ≥80th)</td><td>Ordered workflow: after Touch, the user often starts PlayAnnotation.</td></tr>
      <tr><td>Low Touch→Play consistency</td><td>touch_to_play_rate in the lowest percentiles (≈ ≤20th)</td><td>Scattered exploration or difficulty progressing after Touch.</td></tr>
      <tr><td>Fast Touch→Play time</td><td>median_touch_to_play_sec in the lowest percentiles</td><td>High responsiveness: the user moves from Touch to Play immediately.</td></tr>
      <tr><td>Slow Touch→Play time</td><td>median_touch_to_play_sec in the highest percentiles</td><td>Hesitation or slowdown before starting Play.</td></tr>
      <tr><td>Limited Pause usage</td><td>pause_per_min below the median / low percentiles</td><td>Smooth session, few interruptions.</td></tr>
      <tr><td>High Pause usage</td><td>pause_per_min in the highest percentiles</td><td>Fragmented or difficult session.</td></tr>
      <tr><td>Moderate Teleport usage</td><td>teleport_per_min below the median / low percentiles</td><td>Controlled, non-chaotic navigation.</td></tr>
      <tr><td>Frequent Teleport usage</td><td>teleport_per_min in the highest percentiles</td><td>Disordered exploration or continuous searching.</td></tr>
      <tr><td>No long inactive period</td><td>idle_gaps_gt60 = 0</td><td>Continuously active session.</td></tr>
      <tr><td>Many long inactive periods</td><td>idle_gaps_gt60 in the highest percentiles</td><td>Long pauses / temporary abandonment.</td></tr>
      <tr><td>Stable post-Touch sequences</td><td>entropy_next_after_touch low / low percentiles</td><td>Predictable and coherent actions after Touch.</td></tr>
      <tr><td>Unstable post-Touch sequences</td><td>entropy_next_after_touch high / high percentiles</td><td>Irregular workflow, trial and error.</td></tr>
    </table>
    </div>

<h4>Archetypes (pattern + performance) — practical guide</h4>

<p class="small">
  Archetypes are interpretive labels computed by combining multiple metrics.
  Thresholds are <b>relative to the dataset</b> and are expressed as:
  <ul>
    <li><b>P80</b> = 80° percentile (valori “alti” rispetto al dataset)</li>
    <li><b>P20</b> = 20° percentile (valori “bassi” rispetto al dataset)</li>
    <li><b>mediana</b> = 50° percentile</li>
  </ul>
<p class="small">
  <b>Activation rule (soft):</b> each archetype has a set of conditions; each true condition counts as 1 hit.
  An archetype is <b>activated</b> if it reaches at least <b>2 hits</b>.
  The <b>primary archetype</b> is the one with the most hits.
</p>
</p>

<div class="table-scroll">
<table>
  <tr>
    <th>Archetype</th>
    <th>Significato quantitativo </th>
    <th>Interpretazione comportamentale</th>
  </tr>

  <tr>
    <td><b>Esperto Efficiente</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • pattern_score ≥ P80<br/>
      • score_percent ≥ P80<br/>
      • match_accuracy ≥ P80<br/>
      • (conceptsolve_first_try / conceptsolve_count) ≥ 0.70<br/>
      • conceptsolve_mean_time ≤ mediana
    </td>
    <td>Workflow lineare, molto preciso e rapido. Padronanza alta e risoluzione “pulita” (molti first-try).</td>
  </tr>

  <tr>
    <td><b>Performer Lineare</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • pattern_score ≥ 0<br/>
      • score_percent ≥ mediana<br/>
      • match_accuracy ≥ mediana<br/>
      • pause_per_min ≤ P80<br/>
      • teleport_per_min ≤ P80
    </td>
    <td>Ordered and stable behavior with good outcome.</td>
  </tr>

  <tr>
    <td><b>Esploratore</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • teleport_per_min ≥ P80<br/>
      • entropy_next_after_touch ≥ P80<br/>
      <b>Note:</b> performance is not constrained (it may be average or even high).
    </td>
    <td>Intense navigation and frequent context changes. Exploratory/search-oriented strategy, not necessarily struggling.</td>
  </tr>

  <tr>
    <td><b>Interrotto / Inattivo</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • idle_gaps_gt60 ≥ P80<br/>
      • pause_per_min ≥ P80<br/>
      <b>Spesso:</b> pattern_score ≤ mediana
    </td>
    <td>Fragmented session: long interruptions and/or many pauses. Possible distraction, temporary drop-offs, or stop-and-go interaction.</td>
  </tr>

  <tr>
    <td><b>Attempt Iterator</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • match_attempts_total ≥ P80<br/>
      • match_accuracy ≤ P20<br/>
      • match_wrong ≥ P80<br/>
      <b>Spesso:</b> conceptsolve_mean_attempts ≥ mediana
    </td>
    <td>Trial and error: many matches with low precision.</td>
  </tr>

  <tr>
    <td><b>Performer Persistente</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • match_attempts_total ≥ P80<br/>
      • conceptsolve_mean_attempts ≥ P80<br/>
      • match_accuracy ≥ P20<br/>
      • pattern_score ≥ 0
    </td>
    <td>Persistence: many attempts and effort, but maintains minimum coherence and is not purely random.</td>
  </tr>

  <tr>
    <td><b>Risolutore Metodico</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • conceptsolve_mean_time ≥ P80<br/>
      • match_accuracy ≥ P80<br/>
      • teleport_per_min ≤ mediana<br/>
      • pattern_score ≥ 0
    </td>
    <td>Slow but accurate: reflective, controlled approach with good precision and non-chaotic navigation.</td>
  </tr>

  <tr>
    <td><b>Fast but Error-prone</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • median_touch_to_play_sec ≤ P20<br/>
      • conceptsolve_mean_time ≤ P20<br/>
      • match_accuracy ≤ P20<br/>
      • match_wrong ≥ P80
    </td>
    <td>Very fast but often wrong: possible impulsiveness/overconfidence or superficial content processing.</td>
  </tr>

  <tr>
    <td><b>In Difficolta'</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • score_percent ≤ P20<br/>
      • match_accuracy ≤ P20<br/>
      • conceptsolve_mean_time ≥ P80<br/>
      • conceptsolve_mean_attempts ≥ P80<br/>
      <b>Spesso:</b> pause_per_min ≥ mediana
    </td>
    <td>Consistent difficulty: slowness + many attempts + low accuracy and low final result.</td>
  </tr>

  <tr>
    <td><b>Cauto Frammentato</b></td>
    <td>
      <b>Criteri principali:</b><br/>
      • pause_per_min ≥ P80<br/>
      • entropy_next_after_touch ≥ P80<br/>
      • score_percent ≥ P20<br/>
      • match_accuracy ≥ P20
    </td>
    <td>Unstable and cautious workflow: proceeds in small steps, pauses often, and reorganizes strategy without necessarily failing.</td>
  </tr>

</table>
</div>
    """

    best_k = clust_payload.get("best_k", np.nan)
    if isinstance(best_k, (int, np.integer)):
        best_k_txt = f"(k = {best_k})"
    elif isinstance(best_k, float) and not np.isnan(best_k):
        best_k_txt = f"(k = {int(best_k)})"
    else:
        best_k_txt = ""

    pca_explain_html = """
  <p>
    The PCA plot represents sessions in a two-dimensional space (PC1, PC2) 
    derived from the decomposition of the selected features. Each point is a session
    and color indicates the cluster assigned by the best clustering model.
    Visually well-separated clusters suggest that the selected features distinguish
    different interaction strategies well.
  </p>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Report — Preprocess + EDA + Feature Selection + Clustering + Sequence</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; color: #222; }}
h1,h2,h3,h4 {{ margin: .4em 0; }}
section {{ margin-bottom: 28px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; }}
th {{ background: #fafafa; }}
.figure {{ margin: 14px 0; }}
img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#f6f8fa; padding: 2px 6px; border-radius: 6px; }}
.small {{ font-size: 12px; color:#666; }}
details {{ border: 1px solid #eee; padding: 10px 12px; border-radius: 8px; margin-bottom: 10px; background:#fafafa; }}
summary {{ cursor: pointer; }}
hr {{ border: none; border-top: 1px solid #eee; margin: 16px 0; }}
table.drop-log {{ width: auto; margin: 0 auto 10px auto; font-size: 13px; }}
table.drop-log th, table.drop-log td {{ text-align: center; }}

.table-scroll {{
  max-height: 400px;
  overflow: auto;
  border: 1px solid #ddd;
  padding: 4px;
  background: #fff;
  margin-top: 6px;
  margin-bottom: 10px;
}}
.table-scroll table {{
  font-size: 12px;
}}

.badge {{
  display:inline-block;
  padding:2px 8px;
  border-radius:999px;
  font-size:12px;
  font-weight:700;
  line-height:1.4;
  vertical-align:middle;
  margin-left:6px;
  border:1px solid transparent;
}}
.badge-green {{ background:#e8f7ee; color:#156f3b; border-color:#bfe7cf; }}
.badge-yellow {{ background:#fff6da; color:#7a5b00; border-color:#f0dea6; }}
.badge-red {{ background:#fde8e8; color:#8a1c1c; border-color:#f5bcbc; }}
.badge-gray {{ background:#f2f2f2; color:#444; border-color:#ddd; }}
</style>
</head>
<body>

<h1>Report — Preprocess + EDA + Feature Selection + Clustering + Sequence + Scoring + Archetypes</h1>

<section>
  <h2>Overview</h2>
  <p>Input: <code>{INPUT_DIR}/*.txt, *.csv</code></p>
  <p>Output CSV: <code>{FEATURES_CSV.name}</code>, <code>{RES_CSV.name}</code>, <code>{ASSIGN_CSV.name}</code>, <code>{PROFILES_CSV.name}</code></p>
</section>

<section>
  <h2>Session features (preview)</h2>
  <div class="table-scroll">
    {head_html}
  </div>
</section>

<hr/>

<section>
  <h2>EDA</h2>
  <h3>Statistiche descrittive</h3>

  <p>
    The table below shows a statistical summary of the numerical features extracted from all sessions.
    Each row represents a feature, while each column describes one aspect of its distribution:<br/><br/>
    <b>count</b>: number of sessions with a valid value.<br/>
    <b>mean</b>: average value of the feature.<br/>
    <b>std</b>: standard deviation, indicating variability across sessions.<br/>
    <b>min</b>: valore minimo osservato.<br/>
    <b>25%</b>: first quartile (25% of sessions have values ≤ this).<br/>
    <b>50%</b>: median (50% of sessions have values ≤ this).<br/>
    <b>75%</b>: third quartile (75% of sessions have values ≤ this).<br/>
    <b>max</b>: valore massimo osservato.<br/><br/>
  </p>

  {eda_payload["desc_html"]}

  <h3>Histograms</h3>

  <p>
    Histograms show the distribution of each numerical feature analyzed. Each chart represents a single feature and
    shows how its values are distributed across sessions:<br/><br/>
    <b>X-axis</b>: numerical values of the feature (e.g. duration, number of events, mean time, etc.)<br/>
    <b>Y-axis</b>: frequency, i.e. how many sessions fall into that interval<br/>
    <b>Legend</b>: name of the feature shown in the plot.<br/><br/>
  </p>

  <div class="figure">
    <img src="{eda_payload['hist_b64']}" alt="Histograms"/>
  </div>

  <h3>Correlation matrix</h3>

  <p>
    The correlation matrix shows the linear relationship among all numerical features analyzed.
    Each cell in the matrix contains a value between -1 and +1, where:<br/><br/>
    <b>+1</b>: correlazione positiva perfetta (le due feature aumentano insieme);<br/>
    <b>0</b>: no linear correlation;<br/>
    <b>-1</b>: correlazione negativa perfetta (una aumenta mentre l’altra diminuisce).<br/><br/>
    In the heatmap, more intense colors indicate stronger relationships:
    warm tones represent positive correlations, while cool tones indicate negative correlations.
    This analysis helps identify redundant features, recurring patterns, and possible relationships between variables useful
    alla comprensione del dataset e ai passaggi successivi (feature selection e clustering).
  </p>

  <div class="figure">
    <img src="{eda_payload['corr_b64']}" alt="Correlation matrix"/>
  </div>
</section>

<hr/>

<section>
  <h2>Feature Selection</h2>

  <p>
    In this section, only the features truly useful for analysis and clustering are selected.
    The procedure is based on two main steps:<br/><br/>

    <b>1) Variance Threshold</b>: removes features with near-zero variance, i.e. variables that take
    sempre lo stesso valore e che quindi non aggiungono informazione.<br/><br/>

    <b>2) Correlation Threshold</b>: removes one of two features that are strongly correlated
    (correlation &gt; 0.9), avoiding redundancy while keeping the more representative variable.<br/><br/>

    This selection reduces noise, removes redundancy, and improves the quality
    for subsequent analyses, especially clustering.
  </p>

  <p><b>Feature mantenute (Keep):</b> {kept_list}</p>

  <h3>Drop log (variance/correlation)</h3>
  {dropped_html}
</section>

<hr/>

<section>
  <h2>Clustering</h2>

  <p>
    Clustering makes it possible to automatically group sessions based on their
    behavioral characteristics. Each cluster represents a set of similar sessions, identifying patterns
    distinctive patterns such as faster, more exploratory, more cautious, or more systematic users.<br/><br/>

    In the report, the selected features are used and scaled with <code>StandardScaler</code>.
    Five algorithms are compared (KMeans, Agglomerative, Spectral, GMM, DBSCAN), and for the first 4,
    a grid of values <b>k = 3..7</b> is explored.<br/><br/>

    <b>Best model selection criterion</b>:
    results are ranked by:
    <ol>
      <li><b>Silhouette score</b> (higher is better)</li>
      <li><b>Number of clusters obtained</b> (higher is better, with equal silhouette)</li>
      <li><b>Davies–Bouldin index</b> (lower is better)</li>
    </ol>

    The model at the top of this ranking is labeled as the <b>"Best model"</b> and is
    used to assign clusters and compute average profiles.
  </p>

  <p><b>Best model:</b> <code>{clust_payload['best_name']}</code> {best_k_txt}</p>

  {pca_explain_html}

  <h3>PCA scatter — best</h3>
  <div class="figure">
    <img src="{pca_b64}" alt="PCA scatter"/>
  </div>

  <h3>PCA scatter — next best 4</h3>
  <p class="small">
    The following charts show the next 4 models in the ranking, to visually compare
    cluster separation.
  </p>
  {top_next_html}

  <h3>Metrics (5 models × k=3..7)</h3>

  <p>
    This section compares the different algorithms using three standard metrics:
    <ul>
      <li><b>Silhouette Score</b>: higher = more separated and compact clusters</li>
      <li><b>Calinski-Harabasz Index</b>: higher = more distant and compact clusters</li>
      <li><b>Davies-Bouldin Index</b>: lower = more distinct clusters</li>
    </ul>
    The best model is selected according to the criterion described above.
  </p>

  {res_html}

  <h3>Cluster profiles</h3>

  <p>
    This table shows the <b>average profile</b> of each cluster, obtained by calculating the mean and median of the
    selected features for all sessions belonging to the same group.<br/><br/>

    <b>Note:</b> the profiles are based on the assignments of the <b>best model</b>
    (<code>{clust_payload['best_name']}</code>).<br/><br/>

    These values form the <b>centroid</b> of the cluster and make it possible to understand which features distinguish
    the groups.
  </p>

  {profiles_html}
</section>

<hr/>

<section>
  <h2>Sequence Analysis</h2>

  <p>
    Sequence Analysis analyzes the temporal order of events generated during sessions,
    helping understand how interaction evolves step by step. Unlike purely
    aggregate statistics, the goal here is to observe the <b>flow</b> of actions
    (Touch, PlayAnnotation, Grab, Release, Teleport, etc.) and the patterns with which they unfold over time.<br/><br/>

    The following are shown:
    <ul>
      <li>the <b>transition matrix</b>, which describes the probabilities of moving from one action to the next;</li>
      <li>the most frequent <b>bigrams</b> and <b>trigrams</b> (sequences of 2 or 3 consecutive actions);</li>
      <li><b>dwell time</b> statistics, i.e. time intervals between consecutive events of the same action;</li>
      <li>the <b>error co-occurrence matrix</b> and the conceptual confusion table.</li>
    </ul>
  </p>

  <div class="figure">
    <h3>Transition matrix</h3>
    <img src="{trans_b64}" alt="Transition matrix"/>
  </div>

  <h3>Most frequent bigrams</h3>
  {bigrams_html}

  <h3>Most frequent trigrams</h3>
  {trigrams_html}

  <h3>Dwell time by action</h3>
  {dwell_html}

  <h3>Error co-occurrence matrix (target vs chosen)</h3>
  <p><em>Aggregated n×n table: rows = correct concept (target), columns = incorrectly chosen concept, values = number of errors.</em></p>

  <div class="figure">
    <img src="{error_mat_b64}" alt="Error co-occurrence matrix"/>
  </div>

  <h4>Full table (n×n matrix)</h4>
  <p><em>Tabular version of the matrix above.</em></p>
  <div class="table-scroll">
    {error_mat_html}
  </div>

  <h3>Concept confusion errors</h3>
  <p class="small">
    Each row represents a pair (target concept, chosen concept) in an incorrect MatchAttempt,
    with the number of times the confusion occurred.
  </p>
  {confusion_html}
</section>

<hr/>



<section>
  <h2>Concept Timeline Diagnostics (cross-session)</h2>

  <p class="small">
    Purpose: cross-session comparison to identify which <b>concepts</b> are more difficult by combining performance and repetition signals.
    <br/>
  </p>

  {concept_cols_explain_html}

  <div class="table-scroll">
    {concept_tbl_html}
  </div>

  <h3>Concept clustering (k = 3)</h3>
  <p class="small">
    Concepts are clustered using aggregated features (n_sessions, solve_mean_s, match_accuracy,
    play_max_streak, match_max_streak, problem_score). Five models are compared and the best is selected
    using the same criterion used for sessions.
  </p>

  {concept_cluster_html}

  {conc_clust_vs_sess_clust_html}

  <h3>Sessions contributing the most (per concept)</h3>
  <p class="small">
    Impact is calculated using the <b>same indicators</b> as the ranking (solve_mean_s, 1−match_accuracy, play_max_streak, match_max_streak),
    normalized to 0..1 within each concept to make them comparable.
  </p>

  {worst_sessions_html}

</section>



<hr/>

<section>
  <h2>Sequence Pattern Detection + Archetypes</h2>

  <p>
    This section analyzes interaction patterns at the single-session level, using
    metrics derived from event sequences (e.g. Touch, PlayAnnotation, Pause, Teleport) and
    additional information available in the more detailed logs (MatchAttempt, ConceptSolve, ScoreSummary).<br/><br/>

    Several metrics are computed for each session, including:
    <ul>
      <li><b>touch_to_play_rate</b>: share of Touch events followed by PlayAnnotation;</li>
      <li><b>median_touch_to_play_sec</b>: median time between a Touch and the corresponding Play;</li>
      <li><b>pause_per_min</b> and <b>teleport_per_min</b>: frequency of Pause and Teleport per minute;</li>
      <li><b>idle_gaps_gt60</b>: number of long inactive intervals (&gt; 60 seconds);</li>
      <li><b>entropy_next_after_touch</b>: variability of the actions following a Touch;</li>
      <li><b>metrics from MatchAttempt</b> (e.g. <code>match_attempts_total</code>, <code>match_accuracy</code>,
          <code>match_correct</code>, <code>match_wrong</code>);</li>
      <li><b>metrics from ConceptSolve</b> (e.g. <code>conceptsolve_count</code>, mean/median times,
          mean attempts, share of first-try vs multi-try);</li>
      <li><b>metrics from ScoreSummary</b> (e.g. <code>score_tot_solved</code>, <code>score_tot_accuracy</code>,
          <code>score_percent</code>).</li>
    </ul>

    Based on the quantiles of these distributions, the following are assigned:
    <ul>
      <li><b>effective_flags</b>: positive patterns relative to the other sessions;</li>
      <li><b>problem_flags</b>: potentially problematic patterns.</li>
    </ul>

    A <b>pattern_score</b> is derived from these flags, summarizing the behavioral quality
    sequential quality of the session (effective − problematic).
  </p>

  <h3>Quick overview</h3>
  {pat_summary_html}

  <div class="figure">
    <img src="{pat_hist_b64}" alt="Pattern score distribution"/>
  </div>

  {flags_guide_html}

  <h3>Top 5 most critical sessions</h3>
  <p class="small">
    Selected by sorting sessions by <b>pattern_score</b> in ascending order
    (most negative first). The table combines interaction patterns (flags + sequence metrics)
    with performance metrics (MatchAttempt, ConceptSolve, ScoreSummary) to understand <b>why</b>
    a session is identified as critical.
  </p>
  <div class="table-scroll">
    {top_worst_html}
  </div>

  <h3>Top 5 most linear sessions</h3>
  <p class="small">
    The 5 sessions with the highest <b>pattern_score</b> (many effective_flags and few problem_flags).
    The column structure is identical to the critical sessions table, so you can compare directly
    workflow and outcome.
  </p>
  <div class="table-scroll">
    {top_best_html}
  </div>

  <h3>All sessions (complete metrics &amp; flags)</h3>
  <p class="small">
    Complete per-session view: flags and sequence metrics, performance (accuracy/times/final percentage)
    and assigned cluster.
  </p>
  <div class="table-scroll">
    {pat_sess_html}
  </div>

  <h3>Cluster summary</h3>
  <p>
    Sequence metrics are also aggregated by cluster of the best model, reporting means and medians.
    This helps characterize groups in terms of interaction strategies.
  </p>
  {pat_cluster_html}

  <hr/>

  <h2>Archetypes</h2>
  <p class="small">
    Archetypes are interpretive (rule-based) labels derived from <b>pattern_score</b>, performance, and interaction signals.
    They provide an immediate qualitative reading of clusters/sessions, without replacing the numerical results.
  </p>

  <h3>Archetype Summary</h3>
  <div class="table-scroll">
    {arch_summary_html}
  </div>

  <h3>Archetypes by session</h3>
  <div class="table-scroll">
    {arch_sess_html}
  </div>
</section>

<hr/>

<section>
  <h2>Per-session summary (flags, performance &amp; archetypes)</h2>
  <p class="small">
    Each session panel is a “reading card” that coherently brings together what was introduced in
    <b>Sequence Pattern Detection</b> and <b>Archetypes</b>:
    <ul>
      <li><b>Session identity</b>: ID, duration, number of events, cluster assigned by the best model.</li>
      <li><b>Pattern Score</b> with colored badge (green = more linear, yellow = intermediate, red = critical), based on effective/problem flags.</li>
      <li><b>Flags</b>:
        <ul>
          <li><code>effective_flags</code>: “positive” behaviors relative to the rest of the dataset (high Touch→Play consistency, fast times, few pauses, etc.).</li>
          <li><code>problem_flags</code>: potentially critical patterns (many pauses, frequent teleports, long idle periods, high entropy after Touch, etc.).</li>
        </ul>
      </li>
      <li><b>Sequence metrics with percentile</b>: the same metrics used to build the flags (Touch→Play rate, Touch→Play times, pauses/teleports per minute, idle, entropy, streaks).</li>
      <li><b>Performance metrics</b> (MatchAttempt / ConceptSolve / ScoreSummary): accuracy, attempts, resolution times, and final percentage.</li>
      <li><b>Archetype</b>: rule-based interpretive label summarizing interaction/performance style (e.g. “exploratory”, “linear”, “struggling”).</li>
      <li><b>Action counts</b>: compact summary of main actions and full table with percentiles to understand how “atypical” a behavior is.</li>
      <li><b>Event timeline</b>: time scatter plot with actions on the vertical axis, useful to see phases of intense activity, pauses, and strategy changes.</li>
      <li><b>Concept confusion errors</b>: per-session table of incorrect MatchAttempt events, to understand which concepts are most often confused.</li>
    </ul>
  </p>

  {per_session_html}
</section>

<footer><p class="small">Automatically generated.</p></footer>
</body>
</html>
"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

# -----------------------
# MAIN
# -----------------------
def main(log_dir=None, out_root=None, skip_eda=False, export_processed_csv=True):
    global INPUT_DIR, OUT_DIR
    global FEATURES_CSV, RES_CSV, ASSIGN_CSV, PROFILES_CSV, REPORT_HTML


    """Run the full Mode 4 pipeline.

    Parameters
    ----------
    log_dir : str or None
        Path to the folder containing Mode 4 logs (.txt/.csv). If None, the user is prompted.
    out_root : str or None
        Path to the output root folder where the 'Report' folder will be created. If None, the user is prompted.
    skip_eda : bool
        If True, skips the heavy EDA plots section (useful on low-memory machines).
    export_processed_csv : bool
        If True, exports cleaned per-session logs as CSV to Report/ProcessedLogs (useful for downstream notebooks/models).
    """

    if log_dir is None:
        log_dir = input("Enter the path of the log folder (.txt/.csv): ").strip().strip('"')
    if out_root is None:
        out_root = input("Enter the path of the folder where the 'Report' folder will be created: ").strip().strip('"')


    if not os.path.isdir(log_dir):
        print(f"[ERROR] Log folder does not exist: {log_dir}")
        return

    if not os.path.isdir(out_root):
        print(f"[INFO] Output folder does not exist, creating it: {out_root}")
        os.makedirs(out_root, exist_ok=True)

    INPUT_DIR = log_dir
    OUT_DIR = Path(out_root) / "Report"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FEATURES_CSV = OUT_DIR / "Mode4Features.csv"
    RES_CSV = OUT_DIR / "clustering_results.csv"
    ASSIGN_CSV = OUT_DIR / "best_model_assignments.csv"
    PROFILES_CSV = OUT_DIR / "best_cluster_profiles.csv"
    REPORT_HTML = OUT_DIR / "report.html"

    print(f"[OK] Log folder: {INPUT_DIR}")
    print(f"[OK] Report folder: {OUT_DIR}")

    feats, sessions_raw = build_features_and_sessions(INPUT_DIR)
    feats.to_csv(FEATURES_CSV, index=False)

    if export_processed_csv:
        proc_dir = OUT_DIR / "ProcessedLogs"
        proc_dir.mkdir(parents=True, exist_ok=True)
        for sid, df in sessions_raw:
            df.to_csv(proc_dir / f"{sid}.csv", index=False)

    X_full = feats[DESIRED_FEATURES].copy()
    if skip_eda:
        eda_payload = {"html": "<p><em>EDA skipped (skip_eda=True).</em></p>"}
    else:
        eda_payload = eda_section(X_full)

    X_sel, kept_cols, dropped_tbl = variance_corr_filter(X_full)

    clust = best_of_five_clustering(X_sel)

    seqs = sessions_to_sequences(sessions_raw)
    bi_df = ngram_counts(seqs, n=2, top=30)
    tri_df = ngram_counts(seqs, n=3, top=30)
    acts, P = transition_matrix(sessions_raw)
    trans_b64 = transition_heatmap_b64(acts, P)
    dwell_df = dwell_time_stats(sessions_raw)

    err_concepts, err_M, err_df, err_b64 = error_cooccurrence_matrix(sessions_raw)
    conf_df = confusion_matrix_concepts(sessions_raw)

    seq_payload = {
        "bigrams": bi_df,
        "trigrams": tri_df,
        "trans_b64": trans_b64,
        "dwell_df": dwell_df,
        "error_mat_df": err_df,
        "error_mat_b64": err_b64,
        "confusion_df": conf_df,
    }

    pat = sequence_pattern_detection(sessions_raw, feats, clust)

    # also save CSV with archetypes
    try:
        pat_out = OUT_DIR / "pattern_detection_with_archetypes.csv"
        if pat.get("per_session", pd.DataFrame()).empty is False:
            pat["per_session"].to_csv(pat_out, index=False)
            print("[OK] CSV Pattern+Archetypes:", pat_out.resolve())
    except Exception:
        pass

    save_single_session_reports(sessions_raw, pat["per_session"], OUT_DIR)

    build_html_report(
        feats_df=feats,
        eda_payload=eda_payload,
        fs_X=X_sel, fs_kept=kept_cols, fs_drop_tbl=dropped_tbl,
        clust_payload=clust,
        seq_payload=seq_payload,
        sessions_raw=sessions_raw,
        pat=pat
    )

    print("[OK] Report created:", REPORT_HTML.resolve())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WP4 Learning Analytics — Mode 4 pipeline (D4.1).")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to Mode 4 logs folder (txt/csv). If omitted, you will be prompted.")
    parser.add_argument("--out_root", type=str, default=None, help="Output root folder (a 'Report' folder will be created here). If omitted, you will be prompted.")
    parser.add_argument("--skip_eda", action="store_true", help="Skip heavy EDA plots to avoid memory issues.")
    parser.add_argument("--no_export_processed_csv", action="store_true", help="Do not export per-session processed CSV logs.")
    args = parser.parse_args()

    main(
        log_dir=args.log_dir,
        out_root=args.out_root,
        skip_eda=args.skip_eda,
        export_processed_csv=not args.no_export_processed_csv
    )
