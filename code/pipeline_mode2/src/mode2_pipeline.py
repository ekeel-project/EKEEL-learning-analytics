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
# Config (runtime values)
# -----------------------
INPUT_DIR = None
OUT_DIR = None

FEATURES_CSV = None
RES_CSV = None
ASSIGN_CSV = None
PROFILES_CSV = None
REPORT_HTML = None

TIMESTAMP_FMT = "%Y-%m-%d-%H:%M:%S:%f"
COLS = ["Timestamp", "Action", "ActionID", "Direction"]

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PER_SESSION_LIMIT = None
BIN_SECONDS_TIMELINE = 10
MAIN_ACTIONS = ["Touch", "Grab", "Release", "Teleport",
                "PlayAnnotation", "PlayVideo", "Pause", "Resume"]

# -----------------------
# Plot utils
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
    df = pd.read_csv(
        path, header=None, names=COLS, sep=",",
        dtype={"Action": "string", "ActionID": "string", "Direction": "string"},
        engine="python", na_values=["", "null", "NULL", "NaN", "nan", "None"]
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=TIMESTAMP_FMT, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    for c in ["Action", "ActionID", "Direction"]:
        df[c] = df[c].astype("string").fillna("").str.strip()
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
    dur = (df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]).total_seconds()
    action_counts = df["Action"].value_counts().to_dict()
    touch_df = df[df["Action"] == "Touch"]
    unique_concepts = int(touch_df["ActionID"].nunique()) if not touch_df.empty else 0
    concept_counts = touch_df["ActionID"].value_counts().to_dict() if not touch_df.empty else {}

    base = {
        "session_duration": float(dur),
        "count_Teleport": count_action(action_counts, "Teleport"),
        "count_Grab": count_action(action_counts, "Grab"),
        "count_Release": count_action(action_counts, "Release"),
        "count_PlayVideo": count_action(action_counts, "PlayVideo"),
        "count_Pause": count_action(action_counts, "Pause"),
        "count_Resume": count_action(action_counts, "Resume"),
        "count_Touch": count_action(action_counts, "Touch"),
        "unique_concepts_touched": unique_concepts,
        "touch_play_count": count_touch_followed_by_play(df),
        "grab_release_count": int(min(count_action(action_counts, "Grab"),
                                      count_action(action_counts, "Release"))),
        "avg_touch_play_duration": avg_touch_to_next_play_seconds(df),
    }
    return base, concept_counts

def build_features_and_sessions(input_dir):
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    per_session = []
    sessions_raw = []
    all_concepts = set()

    for path in txt_files:
        sid = os.path.splitext(os.path.basename(path))[0]
        df = read_session_txt(path)
        if df.empty:
            continue
        sessions_raw.append((sid, df))
        base, ccounts = extract_features_from_df(df)
        per_session.append((sid, base, ccounts))
        all_concepts.update(ccounts.keys())

    if not per_session:
        raise RuntimeError("No feature rows generated.")

    all_concepts = sorted(all_concepts)

    rows = []
    for sid, base, ccounts in per_session:
        row = {"sessionID": sid, **base}
        for c in all_concepts:
            row[f"concept_{c}"] = int(ccounts.get(c, 0))
        rows.append(row)

    feats = pd.DataFrame(rows)

    base_cols = [
        "sessionID", "session_duration", "count_Teleport", "count_Grab", "count_Release",
        "count_PlayVideo", "count_Pause", "count_Resume", "count_Touch",
        "unique_concepts_touched", "touch_play_count", "grab_release_count", "avg_touch_play_duration",
    ]
    concept_cols = [c for c in feats.columns if c.startswith("concept_")]
    return feats[base_cols + sorted(concept_cols)], sessions_raw

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

    return {"desc_html": desc_html, "hist_b64": hist_b64, "corr_b64": corr_b64}

# -----------------------
# Feature Selection
# -----------------------
def variance_corr_filter(X, var_thresh=1e-6, corr_thresh=0.9):
    """
    1) Median imputation
    2) Drop low-variance features
    3) Drop highly correlated features: between two correlated features, drop the "worse" one
       (lower variance; if tied, more missing values; if tied, deterministic tie-break).
    """
    if X.shape[1] == 0:
        return X.copy(), list(X.columns), pd.DataFrame(
            columns=["reason", "feature", "correlated_with", "corr_value"]
        )

    dropped_rows = []

    # --- pre-imputation missingness (used to decide what to keep) ---
    miss_rate = X.isna().mean().to_dict()

    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # --- (1) Variance threshold ---
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

    if X1.shape[1] <= 1:
        kept = list(X1.columns)
        dropped_tbl = pd.DataFrame(dropped_rows,
                                  columns=["reason", "feature", "correlated_with", "corr_value"])
        if "corr_value" in dropped_tbl.columns:
            dropped_tbl["corr_value"] = dropped_tbl["corr_value"].round(3)
        return X1.copy(), kept, dropped_tbl

    # --- Correlation threshold ---
    corr = X1.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # post-imputation variance (higher = better)
    var = X1.var(axis=0).to_dict()

    def _choose_drop(a, b):
        """
        Returns which feature to drop between (a,b) according to:
        - drop the one with lower variance
        - if tied: drop the one with more original missing values
        - if tied: drop the one with "greater" name (determinism)
        """
        va, vb = var.get(a, 0.0), var.get(b, 0.0)
        if va < vb:
            return a
        if vb < va:
            return b

        ma, mb = miss_rate.get(a, 0.0), miss_rate.get(b, 0.0)
        if ma > mb:
            return a
        if mb > ma:
            return b

        return max(a, b)

    # list of correlated pairs, sorted by descending correlation
    pairs = []
    for col in upper.columns:
        for row, v in upper[col].dropna().items():
            if v > corr_thresh:
                pairs.append((row, col, float(v)))  # (feat1, feat2, corr)
    pairs.sort(key=lambda x: x[2], reverse=True)

    to_drop = set()

    for f1, f2, v in pairs:
        if f1 in to_drop or f2 in to_drop:
            continue
        drop_feat = _choose_drop(f1, f2)
        keep_feat = f2 if drop_feat == f1 else f1
        to_drop.add(drop_feat)
        dropped_rows.append({
            "reason": "correlation",
            "feature": drop_feat,
            "correlated_with": keep_feat,
            "corr_value": v
        })

    kept = [c for c in X1.columns if c not in to_drop]
    X2 = X1[kept].copy()

    dropped_tbl = pd.DataFrame(dropped_rows,
                              columns=["reason", "feature", "correlated_with", "corr_value"])
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
            "res_df": pd.DataFrame([{"model": "N/A", "k": np.nan,
                                     "silhouette": np.nan, "calinski_harabasz": np.nan,
                                     "davies_bouldin": np.nan, "n_clusters": 0}]),
            "best_name": "N/A",
            "best_labels": np.full(n_samples, -1),
            "best_k": np.nan,
            "labels_by_model": {},
            "X_imp": X_imp, "Xs": Xs
        }

    results = []
    labels_store = {}
    labels_by_model = {}
    k_list = _k_grid()

    for k in k_list:
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
            lab = km.fit_predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"KMeans_k={k}"
            results.append({"model": name, "k": k, "silhouette": float(sil),
                            "calinski_harabasz": float(ch),
                            "davies_bouldin": float(dbi),
                            "n_clusters": int(len(uniq))})
            labels_store[("KMeans", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

        try:
            agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
            lab = agg.fit_predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"Agglomerative_ward_k={k}"
            results.append({"model": name, "k": k, "silhouette": float(sil),
                            "calinski_harabasz": float(ch),
                            "davies_bouldin": float(dbi),
                            "n_clusters": int(len(uniq))})
            labels_store[("Agglomerative", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

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
                            "calinski_harabasz": float(ch),
                            "davies_bouldin": float(dbi),
                            "n_clusters": int(len(uniq))})
            labels_store[("Spectral", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=RANDOM_STATE)
            gmm.fit(Xs)
            lab = gmm.predict(Xs)
            uniq = np.unique(lab)
            sil = silhouette_score(Xs, lab) if len(uniq) >= 2 else np.nan
            ch = calinski_harabasz_score(Xs, lab) if len(uniq) >= 2 else np.nan
            dbi = davies_bouldin_score(Xs, lab) if len(uniq) >= 2 else np.nan
            name = f"GMM_full_k={k}"
            results.append({"model": name, "k": k, "silhouette": float(sil),
                            "calinski_harabasz": float(ch),
                            "davies_bouldin": float(dbi),
                            "n_clusters": int(len(uniq))})
            labels_store[("GMM", k)] = lab
            labels_by_model[name] = lab
        except Exception:
            pass

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
                        "calinski_harabasz": float(ch),
                        "davies_bouldin": float(dbi),
                        "n_clusters": int(len(uniq))})
        labels_store[("DBSCAN", np.nan)] = lab
        labels_by_model[name] = lab
    except Exception:
        pass

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
# Sequence Analysis (base)
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
    fig = plt.figure(figsize=(min(1.2 * len(acts), 16),
                              min(1.2 * len(acts), 12)))
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
            ts = g["Timestamp"].sort_values().values.astype("datetime64[ns]") \
                   .astype(np.int64) / 1e9
            if len(ts) >= 2:
                times.setdefault(a, []).extend(np.diff(ts).tolist())
    if not times:
        return pd.DataFrame(columns=["action", "mean_dwell_s",
                                     "median_dwell_s", "count_intervals"])

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

def session_timeline_plot(df):
    t0 = df["Timestamp"].iloc[0]
    df = df.copy()
    df["t_sec"] = (df["Timestamp"] - t0).dt.total_seconds()
    actions = list(dict.fromkeys(list(MAIN_ACTIONS)
                                 + df["Action"].astype(str).tolist()))
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

# -----------------------
# Sequence Pattern Detection v2.0
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
    deltas = (play[idx[valid]] - touch[valid]).astype("timedelta64[ns]") \
                .astype(np.int64) / 1e9
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
    ts = df["Timestamp"].sort_values().values.astype("datetime64[ns]") \
            .astype(np.int64) / 1e9
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

    return {
        "n_events": int(len(df)),
        "duration_sec": float((df["Timestamp"].iloc[-1]
                               - df["Timestamp"].iloc[0]).total_seconds()),
        "touch_count": n_touch,
        "playann_count": int((df["Action"] == "PlayAnnotation").sum()),
        "touch_to_play_rate": t2p_rate,
        "median_touch_to_play_sec": (float(np.median(deltas))
                                     if deltas.size else np.nan),
        "pause_per_min": per_minute_rate(df, "Pause"),
        "teleport_per_min": per_minute_rate(df, "Teleport"),
        "max_pause_streak": max_consecutive(df, "Pause"),
        "max_teleport_streak": max_consecutive(df, "Teleport"),
        "idle_gaps_gt60": count_idle_gaps(df, thr_sec=60.0),
        "entropy_next_after_touch": next_action_entropy_after(df, "Touch"),
    }

def assign_vr_archetypes(pat_sess_df):
    """
    Reduced version (8 archetypes) and more orthogonal.
    Output:
      - archetypes: labels separated by '; ' (max 2)
      - archetype_primary: primary label
      - archetype_score: primary score (0..3)
    """
    if pat_sess_df is None or pat_sess_df.empty:
        return pat_sess_df

    df = pat_sess_df.copy()

    # ---- data-driven thresholds ----
    def q(series, p, default=np.nan):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.quantile(p)) if not s.empty else default

    # Content engagement (Touch→Play)
    t2p_hi = q(df["touch_to_play_rate"], 0.80)
    t2p_lo = q(df["touch_to_play_rate"], 0.20)

    # Decision speed (Touch→Play time): low=fast, high=slow
    mt2p_lo = q(df["median_touch_to_play_sec"], 0.20)
    mt2p_hi = q(df["median_touch_to_play_sec"], 0.80)

    # Pause / Teleport
    pause_med = q(df["pause_per_min"], 0.50)
    pause_hi  = q(df["pause_per_min"], 0.80)

    tele_med  = q(df["teleport_per_min"], 0.50)
    tele_hi   = q(df["teleport_per_min"], 0.80)

    # Elastic idle
    idle_med  = q(df["idle_gaps_gt60"], 0.50, default=0.0)
    idle_hi   = q(df["idle_gaps_gt60"], 0.80, default=0.0)
    idle_low_thr = max(1.0, float(idle_med if not np.isnan(idle_med) else 1.0))

    # Stability (entropy)
    ent_med = q(df["entropy_next_after_touch"], 0.50)
    ent_hi  = q(df["entropy_next_after_touch"], 0.80)

    def ok(x):
        return x is not None and not (isinstance(x, float) and np.isnan(x))

    def flags(row):
        t2p = row.get("touch_to_play_rate", np.nan)
        mt  = row.get("median_touch_to_play_sec", np.nan)
        pp  = row.get("pause_per_min", np.nan)
        tp  = row.get("teleport_per_min", np.nan)
        ig  = row.get("idle_gaps_gt60", np.nan)
        en  = row.get("entropy_next_after_touch", np.nan)

        return {
            "t2p_high": ok(t2p) and (not np.isnan(t2p_hi)) and t2p >= t2p_hi,
            "t2p_low":  ok(t2p) and (not np.isnan(t2p_lo)) and t2p <= t2p_lo,

            "t2p_fast": ok(mt) and (not np.isnan(mt2p_lo)) and mt <= mt2p_lo,
            "t2p_slow": ok(mt) and (not np.isnan(mt2p_hi)) and mt >= mt2p_hi,

            "pause_low":  ok(pp) and (not np.isnan(pause_med)) and pp <= pause_med,
            "pause_high": ok(pp) and (not np.isnan(pause_hi)) and pp >= pause_hi,

            "tele_low":  ok(tp) and (not np.isnan(tele_med)) and tp <= tele_med,
            "tele_high": ok(tp) and (not np.isnan(tele_hi)) and tp >= tele_hi,

            "idle_low":  ok(ig) and ig <= idle_low_thr,
            "idle_high": ok(ig) and (not np.isnan(idle_hi)) and ig >= idle_hi and ig > 0,

            "ent_stable":   ok(en) and (not np.isnan(ent_med)) and en <= ent_med,
            "ent_unstable": ok(en) and (not np.isnan(ent_hi)) and en >= ent_hi,
        }

    # ---- 8 archetypes (more distinct) ----
    archetypes = [
        ("Fast content consumer",
         ["t2p_high", "t2p_fast", "pause_low"], 2),

        ("Reflective learner",
         ["t2p_high", "t2p_slow", "pause_high"], 2),

        ("Focused explorer",
         ["t2p_high", "ent_stable", "tele_low"], 2),

        ("Spatial explorer",
         ["tele_high", "t2p_low", "pause_low"], 2),

        ("Zapping / scattered exploration",
         ["tele_high", "ent_unstable", "t2p_fast"], 2),

        ("Disoriented / struggling",
         ["ent_unstable", "pause_high", "idle_high"], 2),

        ("Intermittent",
         ["idle_high", "pause_high"], 2),

        ("Passive observer",
         ["t2p_low", "tele_low", "ent_stable"], 2),
    ]

    out_labels, out_primary, out_primary_score = [], [], []

    for _, row in df.iterrows():
        f = flags(row)

        scored = []
        for name, conds, thr in archetypes:
            s = sum(1 for c in conds if f.get(c, False))
            if s >= thr:
                scored.append((name, s))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)

        labels = [x[0] for x in scored[:2]]
        out_labels.append("; ".join(labels) if labels else "")
        out_primary.append(labels[0] if labels else "")
        out_primary_score.append(float(scored[0][1]) if scored else np.nan)

    df["archetypes"] = out_labels
    df["archetype_primary"] = out_primary
    df["archetype_score"] = out_primary_score

    return df

def sequence_pattern_detection(sessions_raw, feats_df, clust_payload):
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
        return {"per_session": empty_df, "bigrams": bi,
                "trigrams": tri, "trans_b64": trans_b64,
                "cluster_summary": pd.DataFrame()}

    sess_df = pd.DataFrame(rows)

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
    _, ent_high  = q_pair("entropy_next_after_touch", 0.2, 0.8)

    pause_med = sess_df["pause_per_min"].median(skipna=True)
    tele_med  = sess_df["teleport_per_min"].median(skipna=True)
    ent_med   = sess_df["entropy_next_after_touch"].median(skipna=True)

    metrics_cols = [
        "touch_to_play_rate",
        "median_touch_to_play_sec",
        "pause_per_min",
        "teleport_per_min",
        "idle_gaps_gt60",
        "entropy_next_after_touch",
    ]

    percentiles = {col: {} for col in metrics_cols}
    for col in metrics_cols:
        s = sess_df[col]
        ranks = s.rank(pct=True) * 100.0
        for sid, pct in zip(sess_df["sessionID"], ranks):
            percentiles[col][sid] = float(pct) if pd.notna(pct) else np.nan

    eff_list, prob_list, score_list = [], [], []

    def _join_flags(lst):
        return "<br/>".join([f"• {x}" for x in lst]) if lst else ""

    for _, row in sess_df.iterrows():
        ef, pb, score = [], [], 0
        sid = row["sessionID"]

        r = row["touch_to_play_rate"]
        pct = percentiles["touch_to_play_rate"][sid]
        if not np.isnan(r):
            if not np.isnan(t2p_rate_high) and r >= t2p_rate_high:
                ef.append(f"High Touch→Play consistency (percentile {pct:.1f}°, among the best)")
                score += 1
            if not np.isnan(t2p_rate_low) and r <= t2p_rate_low:
                pb.append(f"Low Touch→Play consistency (percentile {pct:.1f}°, among the worst)")
                score -= 1

        mt = row["median_touch_to_play_sec"]
        pct = percentiles["median_touch_to_play_sec"][sid]
        if not np.isnan(mt):
            if not np.isnan(mt2p_low) and mt <= mt2p_low:
                ef.append(f"Fast Touch→Play time (percentile {pct:.1f}°, among the shortest)")
                score += 1
            if not np.isnan(mt2p_high) and mt >= mt2p_high:
                pb.append(f"Slow Touch→Play time (percentile {pct:.1f}°, among the longest)")
                score -= 1

        pp = row["pause_per_min"]
        pct = percentiles["pause_per_min"][sid]
        if not np.isnan(pp):
            if not np.isnan(pause_med) and pp <= pause_med:
                ef.append(f"Limited Pause usage (percentile {pct:.1f}°, below the median)")
                score += 1
            if not np.isnan(pause_high) and pp >= pause_high:
                pb.append(f"High Pause usage (percentile {pct:.1f}°, among the highest)")
                score -= 1

        tp = row["teleport_per_min"]
        pct = percentiles["teleport_per_min"][sid]
        if not np.isnan(tp):
            if not np.isnan(tele_med) and tp <= tele_med:
                ef.append(f"Moderate Teleport usage (percentile {pct:.1f}°, below the median)")
                score += 1
            if not np.isnan(tele_high) and tp >= tele_high:
                pb.append(f"Frequent Teleport usage (percentile {pct:.1f}°, among the highest)")
                score -= 1

        ig = row["idle_gaps_gt60"]
        pct = percentiles["idle_gaps_gt60"][sid]
        if not np.isnan(ig):
            if ig == 0:
                ef.append("No long inactive periods (>60s)")
                score += 1
            if not np.isnan(idle_high) and ig >= idle_high and ig > 0:
                pb.append(f"Many long inactive periods (>60s) (percentile {pct:.1f}°)")
                score -= 1

        ent = row["entropy_next_after_touch"]
        pct = percentiles["entropy_next_after_touch"][sid]
        if not np.isnan(ent):
            if not np.isnan(ent_med) and ent <= ent_med:
                ef.append(f"Stable post-Touch sequences (percentile {pct:.1f}°, low entropy)")
                score += 1
            if not np.isnan(ent_high) and ent >= ent_high:
                pb.append(f"Unstable post-Touch sequences (percentile {pct:.1f}°, high entropy)")
                score -= 1

        eff_list.append(_join_flags(ef))
        prob_list.append(_join_flags(pb))
        score_list.append(score)

    sess_df["effective_flags"] = eff_list
    sess_df["problem_flags"] = prob_list
    sess_df["pattern_score"] = score_list

    labels_df = pd.DataFrame({
        "sessionID": feats_df["sessionID"].values,
        "cluster": clust_payload["best_labels"]
    })
    sess_df = sess_df.merge(labels_df, on="sessionID", how="left")

    sess_df = assign_vr_archetypes(sess_df)

    clust_summary = pd.DataFrame()
    if "cluster" in sess_df.columns:
        valid = sess_df[(sess_df["cluster"].notna()) & (sess_df["cluster"] != -1)]
        if not valid.empty:
            clust_summary = valid.groupby("cluster")[metrics_cols + ["pattern_score"]] \
                                 .agg(["mean", "median"])

    seqs = sessions_to_sequences(sessions_raw)
    bi = ngram_counts(seqs, n=2, top=30)
    tri = ngram_counts(seqs, n=3, top=30)

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
# Session summary
# -----------------------
def session_summary(df, pat_row=None):
    t_start = df["Timestamp"].iloc[0]
    t_end   = df["Timestamp"].iloc[-1]
    dur = (t_end - t_start).total_seconds()

    counts_all = df["Action"].astype(str).value_counts()

    touch_df = df[df["Action"] == "Touch"]
    unique_concepts = int(touch_df["ActionID"].nunique()) if not touch_df.empty else 0
    top_concepts = touch_df["ActionID"].value_counts().head(5)

    avg_tp = avg_touch_to_next_play_seconds(df)
    seq_m = session_sequence_metrics(df)

    pat_info = {}
    if pat_row is not None and not pat_row.empty:
        pat_info = {
            "pattern_score": pat_row.get("pattern_score", np.nan),
            "effective_flags": pat_row.get("effective_flags", ""),
            "problem_flags": pat_row.get("problem_flags", ""),
            "cluster": pat_row.get("cluster", np.nan),
            "archetypes": pat_row.get("archetypes", ""),
            "archetype_primary": pat_row.get("archetype_primary", ""),
        }

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

def build_per_session_section(sessions_raw, pat_df=None):
    blocks = []

    # --- sort by sessionID ---
    sessions_raw_sorted = sorted(sessions_raw, key=lambda x: x[0])

    # ---- pattern detection map per session ----
    pat_map = {}
    if pat_df is not None and not pat_df.empty and "sessionID" in pat_df.columns:
        pat_map = {r["sessionID"]: r for _, r in pat_df.iterrows()}

    # -----------------------------
    # Percentile rank helper
    # -----------------------------
    def _percentile_rank(arr, x):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0 or x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(np.mean(arr <= float(x)) * 100.0)

    def _fmt_val_and_pct(val, pct, fmt="{:.3f}", unit=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return f"N/A"
        vtxt = fmt.format(val) + unit
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            return vtxt
        return f"{vtxt} (percentile {pct:.1f}°)"

    # -----------------------------
    # Global distributions
    # -----------------------------
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
    ]
    seq_dists = {k: [] for k in seq_metric_keys}

    if pat_df is not None and not pat_df.empty:
        for k in seq_metric_keys:
            if k in pat_df.columns:
                seq_dists[k] = pat_df[k].astype(float).tolist()

    # Global action counts
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

    # -----------------------------
    # Score badge
    # -----------------------------
    def _score_badge_html(score):
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return '<span class="badge badge-gray" title="score not available">N/A</span>'
        try:
            s = float(score)
        except:
            return '<span class="badge badge-gray">N/A</span>'

        if s >= 2:
            cls = "badge-green"
            title = "linear session (score ≥ 2)"
        elif s >= 0:
            cls = "badge-yellow"
            title = "intermediate session (0 ≤ score < 2)"
        else:
            cls = "badge-red"
            title = "critical session (score < 0)"
        return f'<span class="badge {cls}" title="{title}">{s:.0f}</span>'

    # -----------------------------
    # Session loop
    # -----------------------------
    for sid, df in sessions_raw_sorted:
        pat_row = pat_map.get(sid, None)
        sm = session_summary(df, pat_row=pat_row)
        scat_b64 = session_timeline_plot(df)

        dur_val = sm["duration_sec"]
        dur_pct = _percentile_rank(seq_dists.get("duration_sec", []), dur_val)

        def pct_for(metric_key, val):
            return _percentile_rank(seq_dists.get(metric_key, []), val)

        t2p_val = sm.get("touch_to_play_rate", np.nan)
        mt2p_val = sm.get("median_touch_to_play_sec", np.nan)
        pp_val = sm.get("pause_per_min", np.nan)
        tp_val = sm.get("teleport_per_min", np.nan)
        ig_val = sm.get("idle_gaps_gt60", np.nan)
        ent_val = sm.get("entropy_next_after_touch", np.nan)
        mps_val = sm.get("max_pause_streak", np.nan)
        mts_val = sm.get("max_teleport_streak", np.nan)

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

        eff_flags = sm.get("effective_flags", "") or "-"
        prob_flags = sm.get("problem_flags", "") or "-"
        score_val = sm.get("pattern_score", np.nan)
        badge_html = _score_badge_html(score_val)

        block = f"""
<details>
  <summary>
    <b>Session:</b> <code>{sid}</code>
    — duration: {_fmt_val_and_pct(dur_val, dur_pct, fmt="{:.1f}", unit="s")}
    — events: {sm["n_events"]}
    — cluster: {sm.get("cluster", "N/A")}
    — archetype: {sm.get("archetype_primary","") or "N/A"}
    — pattern score: {badge_html}
  </summary>

  <p>
    <b>Start:</b> {sm["t_start"]}<br/>
    <b>End:</b> {sm["t_end"]}<br/>
    <b>Unique concepts touched:</b> {sm["unique_concepts_touched"]}<br/>
    <b>Top concepts (Touch):</b> {sm["top_concepts"] or "(none)"}<br/>
    <b>Archetype:</b> {sm.get("archetype_primary","") or "N/A"}<br/>
    <b>Top archetypes:</b> {sm.get("archetypes","") or "(none)"}<br/>
    <b>Avg Touch→Play (s):</b> {("" if pd.isna(sm["avg_touch_to_play_sec"]) else f"{sm['avg_touch_to_play_sec']:.3f}")}<br/>
  </p>

  <h4>Pattern Detection Flags</h4>
  <p><b>Effective flags:</b><br/>{eff_flags}</p>
  <p><b>Problem flags:</b><br/>{prob_flags}</p>

  <h4>Sequence metrics (with percentile)</h4>
  <ul>
    <li><b>Touch→Play rate</b> (touch_to_play_rate):
        {_fmt_val_and_pct(t2p_val, t2p_pct)}</li>

    <li><b>Median Touch→Play time (s)</b> (median_touch_to_play_sec):
        {_fmt_val_and_pct(mt2p_val, mt2p_pct)}</li>

    <li><b>Pauses per minute</b> (pause_per_min):
        {_fmt_val_and_pct(pp_val, pp_pct)}</li>

    <li><b>Teleports per minute</b> (teleport_per_min):
        {_fmt_val_and_pct(tp_val, tp_pct)}</li>

    <li><b>Idle gaps &gt; 60s (count)</b> (idle_gaps_gt60):
        {_fmt_val_and_pct(ig_val, ig_pct, fmt="{:.0f}")}</li>

    <li><b>Entropy after Touch</b> (entropy_next_after_touch):
        {_fmt_val_and_pct(ent_val, ent_pct)}</li>

    <li><b>Max Pause streak</b> (max_pause_streak):
        {_fmt_val_and_pct(mps_val, mps_pct, fmt="{:.0f}")}</li>

    <li><b>Max Teleport streak</b> (max_teleport_streak):
        {_fmt_val_and_pct(mts_val, mts_pct, fmt="{:.0f}")}</li>
  </ul>

  <h4>Main counts</h4>
  {counts_main_tbl}

  <h4>All action counts (with percentile)</h4>
  <div class="table-scroll">{counts_all_tbl}</div>

  <div class="figure">
    <h4>Event timeline</h4>
    <img src="{scat_b64}" alt="timeline scatter"/>
  </div>
</details>
"""
        blocks.append(block)

    return "\n".join(blocks)

def save_single_session_reports(sessions_raw, pat_df, out_dir):
    """
    Creates a 'SessionReports' folder inside out_dir and saves one HTML file per session.
    The content is the same as the per-session summary panel, but as a standalone page.
    """
    session_reports_dir = Path(out_dir) / "SessionReports"
    session_reports_dir.mkdir(parents=True, exist_ok=True)

    sessions_raw_sorted = sorted(sessions_raw, key=lambda x: x[0])

    pat_map = {}
    if pat_df is not None and not pat_df.empty and "sessionID" in pat_df.columns:
        pat_map = {r["sessionID"]: r for _, r in pat_df.iterrows()}

    # -----------------------------
    # Percentile rank helper
    # -----------------------------
    def _percentile_rank(arr, x):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0 or x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(np.mean(arr <= float(x)) * 100.0)

    def _fmt_val_and_pct(val, pct, fmt="{:.3f}", unit=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return f"N/A"
        vtxt = fmt.format(val) + unit
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            return vtxt
        return f"{vtxt} (percentile {pct:.1f}°)"

    # -----------------------------
    # Global distributions
    # -----------------------------
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

    # -----------------------------
    # Score badge
    # -----------------------------
    def _score_badge_html(score):
        if score is None or (isinstance(score, float) and np.isnan(score)):
            return '<span class="badge badge-gray" title="score not available">N/A</span>'
        try:
            s = float(score)
        except:
            return '<span class="badge badge-gray">N/A</span>'
        if s >= 2:
            cls = "badge-green"; title="linear session (score ≥ 2)"
        elif s >= 0:
            cls = "badge-yellow"; title="intermediate session (0 ≤ score < 2)"
        else:
            cls = "badge-red"; title="critical session (score < 0)"
        return f'<span class="badge {cls}" title="{title}">{s:.0f}</span>'

    # -----------------------------
    # Base HTML template
    # -----------------------------
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

.badge {
  display:inline-block;
  padding:2px 8px;
  border-radius:999px;
  font-size:12px;
  font-weight:700;
  line-height:1.4;
  vertical-align:middle;
  margin-left:6px;
  border:1px solid transparent;
}
.badge-green { background:#e8f7ee; color:#156f3b; border-color:#bfe7cf; }
.badge-yellow { background:#fff6da; color:#7a5b00; border-color:#f0dea6; }
.badge-red { background:#fde8e8; color:#8a1c1c; border-color:#f5bcbc; }
.badge-gray { background:#f2f2f2; color:#444; border-color:#ddd; }
</style>
</head>
<body>
"""

    base_footer = """
<footer><p class="small">Automatically generated.</p></footer>
</body>
</html>
"""

    # -----------------------------
    # Save per session
    # -----------------------------
    for sid, df in sessions_raw_sorted:
        pat_row = pat_map.get(sid, None)
        sm = session_summary(df, pat_row=pat_row)
        scat_b64 = session_timeline_plot(df)

        dur_val = sm["duration_sec"]
        dur_pct = _percentile_rank(seq_dists.get("duration_sec", []), dur_val)

        def pct_for(metric_key, val):
            return _percentile_rank(seq_dists.get(metric_key, []), val)

        t2p_val = sm.get("touch_to_play_rate", np.nan)
        mt2p_val = sm.get("median_touch_to_play_sec", np.nan)
        pp_val = sm.get("pause_per_min", np.nan)
        tp_val = sm.get("teleport_per_min", np.nan)
        ig_val = sm.get("idle_gaps_gt60", np.nan)
        ent_val = sm.get("entropy_next_after_touch", np.nan)
        mps_val = sm.get("max_pause_streak", np.nan)
        mts_val = sm.get("max_teleport_streak", np.nan)

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

        eff_flags = sm.get("effective_flags", "") or "-"
        prob_flags = sm.get("problem_flags", "") or "-"
        score_val = sm.get("pattern_score", np.nan)
        badge_html = _score_badge_html(score_val)

        page = base_head + f"""
<h1>Session Report: <code>{sid}</code></h1>

<section>
  <h2>Summary</h2>
  <p>
    <b>Duration:</b> {_fmt_val_and_pct(dur_val, dur_pct, fmt="{:.1f}", unit="s")}<br/>
    <b>Number of events:</b> {sm["n_events"]}<br/>
    <b>Cluster:</b> {sm.get("cluster","N/A")}<br/>
    <b>Archetype:</b> {sm.get("archetype_primary","") or "N/A"}<br/>
    <b>Top archetypes:</b> {sm.get("archetypes","") or "(none)"}<br/>
    <b>Pattern score:</b> {badge_html}<br/>
  </p>
  <p>
    <b>Start:</b> {sm["t_start"]}<br/>
    <b>End:</b> {sm["t_end"]}<br/>
    <b>Unique concepts touched:</b> {sm["unique_concepts_touched"]}<br/>
    <b>Top concepts (Touch):</b> {sm["top_concepts"] or "(none)"}<br/>
    <b>Avg Touch→Play (s):</b> {("" if pd.isna(sm["avg_touch_to_play_sec"]) else f"{sm['avg_touch_to_play_sec']:.3f}")}<br/>
  </p>
</section>

<section>
  <h2>Pattern Detection Flags</h2>
  <p><b>Effective flags:</b><br/>{eff_flags}</p>
  <p><b>Problem flags:</b><br/>{prob_flags}</p>
</section>

<section>
  <h2>Sequence metrics (with percentile)</h2>
  <ul>
    <li><b>Touch→Play rate</b> (touch_to_play_rate):
        {_fmt_val_and_pct(t2p_val, t2p_pct)}</li>

    <li><b>Median Touch→Play time (s)</b> (median_touch_to_play_sec):
        {_fmt_val_and_pct(mt2p_val, mt2p_pct)}</li>

    <li><b>Pauses per minute</b> (pause_per_min):
        {_fmt_val_and_pct(pp_val, pp_pct)}</li>

    <li><b>Teleports per minute</b> (teleport_per_min):
        {_fmt_val_and_pct(tp_val, tp_pct)}</li>

    <li><b>Idle gap &gt; 60s (count)</b> (idle_gaps_gt60):
        {_fmt_val_and_pct(ig_val, ig_pct, fmt="{:.0f}")}</li>

    <li><b>Entropy after Touch</b> (entropy_next_after_touch):
        {_fmt_val_and_pct(ent_val, ent_pct)}</li>

    <li><b>Max Pause streak</b> (max_pause_streak):
        {_fmt_val_and_pct(mps_val, mps_pct, fmt="{:.0f}")}</li>

    <li><b>Max Teleport streak</b> (max_teleport_streak):
        {_fmt_val_and_pct(mts_val, mts_pct, fmt="{:.0f}")}</li>
  </ul>
</section>

<section>
  <h2>Main counts</h2>
  {counts_main_tbl}
</section>

<section>
  <h2>All action counts (with percentile)</h2>
  <div class="table-scroll">{counts_all_tbl}</div>
</section>

<section>
  <h2>Event timeline</h2>
  <div class="figure">
    <img src="{scat_b64}" alt="timeline scatter"/>
  </div>
</section>
""" + base_footer

        out_path = session_reports_dir / f"{sid}.html"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(page)

    print("[OK] Per-session reports saved in:", session_reports_dir.resolve())

# -----------------------
# HTML REPORT
# -----------------------
def build_html_report(
    feats_df, eda_payload, fs_X, fs_kept, fs_drop_tbl,
    clust_payload, seq_payload, sessions_raw, pat
):
    head_html = feats_df.head(10).to_html(index=False)

    kept_list = ", ".join(fs_kept) if fs_kept else "(all)"
    dropped_html = fs_drop_tbl.to_html(index=False, classes="drop-log") \
        if not fs_drop_tbl.empty else "<p>No dropped features (variance/correlation).</p>"

    # --- clustering tables + PCA ---
    res_html = clust_payload["res_df"].to_html(index=False)
    pca_b64 = pca_scatter_for_labels(
        clust_payload["Xs"],
        clust_payload["best_labels"],
        title=f"PCA scatter — {clust_payload['best_name']}"
    )

    # next 4 PCA plots
    top_next_html = "<p>No additional model available.</p>"
    try:
        res_df = clust_payload["res_df"]
        next4 = res_df.iloc[1:5].copy()
        imgs = []
        for _, r in next4.iterrows():
            mname = r["model"]
            lab = clust_payload.get("labels_by_model", {}).get(mname, None)
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

    # --- sequence analysis ---
    bigrams_html = seq_payload["bigrams"].to_html(index=False) \
        if not seq_payload["bigrams"].empty else "<p>No bigrams.</p>"
    trigrams_html = seq_payload["trigrams"].to_html(index=False) \
        if not seq_payload["trigrams"].empty else "<p>No trigrams.</p>"
    trans_b64 = seq_payload["trans_b64"]
    dwell_html = seq_payload["dwell_df"].to_html(index=False) \
        if not seq_payload["dwell_df"].empty else "<p>No dwell time can be computed.</p>"

    # --- cluster profiles (best model) ---
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
        profiles_html = "<p>Profiles not available (unaligned dimensions).</p>"

    # --- save CSV ---
    feats_df.to_csv(FEATURES_CSV, index=False)
    clust_payload["res_df"].to_csv(RES_CSV, index=False)
    pd.DataFrame({
        "sessionID": feats_df["sessionID"].values,
        "cluster": clust_payload["best_labels"]
    }).to_csv(ASSIGN_CSV, index=False)

    # --- per-session summary (sorted) ---
    per_session_html = build_per_session_section(
        sessions_raw,
        pat_df=pat["per_session"]
    )

    # ==========================
    # Sequence Pattern Detection
    # ==========================
    pat_sess_df = pat["per_session"].copy()

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
            "sessionID", "pattern_score",
            "archetype_primary", "archetypes",
            "effective_flags", "problem_flags",
            "touch_to_play_rate", "median_touch_to_play_sec",
            "pause_per_min", "teleport_per_min",
            "idle_gaps_gt60", "entropy_next_after_touch",
            "cluster"
        ]

        show_cols = [c for c in show_cols if c in pat_sess_df.columns]

        worst5 = pat_sess_df.sort_values("pattern_score", ascending=True).head(5)[show_cols].copy()
        best5 = pat_sess_df.sort_values("pattern_score", ascending=False).head(5)[show_cols].copy()

        for df_ in (worst5, best5):
            if "effective_flags" in df_.columns:
                df_["effective_flags"] = df_["effective_flags"].fillna("").replace("", "-")
            if "problem_flags" in df_.columns:
                df_["problem_flags"] = df_["problem_flags"].fillna("").replace("", "-")
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
        base_order = ["sessionID", "pattern_score", "archetype_primary", "archetypes", "effective_flags", "problem_flags"]
        other_cols = [c for c in pat_sess_df.columns if c not in base_order]
        pat_sess_df = pat_sess_df[base_order + other_cols]

        if "sessionID" in pat_sess_df.columns and "pattern_score" in pat_sess_df.columns:
            pat_sess_df["sessionID"] = [
                _color_session_id(sid, sc)
                for sid, sc in zip(pat_sess_df["sessionID"], pat_sess_df["pattern_score"])
            ]

    pat_sess_html = pat_sess_df.to_html(index=False, escape=False)

    # -----------------------
    # FLAG INTERPRETATION GUIDE
    # -----------------------
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
    <th>Quantitative meaning</th>
    <th>Behavioral interpretation</th>
  </tr>

  <tr>
    <td>High Touch→Play consistency</td>
    <td>touch_to_play_rate in the highest percentiles (≈ ≥80th)</td>
    <td>Ordered workflow: after Touch, the user often starts PlayAnnotation.</td>
  </tr>

  <tr>
    <td>Low Touch→Play consistency</td>
    <td>touch_to_play_rate in the lowest percentiles (≈ ≤20th)</td>
    <td>Scattered exploration or difficulty progressing after Touch.</td>
  </tr>

  <tr>
    <td>Fast Touch→Play time</td>
    <td>median_touch_to_play_sec in the lowest percentiles</td>
    <td>High responsiveness: the user quickly moves from Touch to Play.</td>
  </tr>

  <tr>
    <td>Slow Touch→Play time</td>
    <td>median_touch_to_play_sec in the highest percentiles</td>
    <td>Hesitation or slowdown before starting Play.</td>
  </tr>

  <tr>
    <td>Limited Pause usage</td>
    <td>pause_per_min below the median / low percentiles</td>
    <td>Smooth session with few interruptions.</td>
  </tr>

  <tr>
    <td>High Pause usage</td>
    <td>pause_per_min in the highest percentiles</td>
    <td>Fragmented or difficult session.</td>
  </tr>

  <tr>
    <td>Moderate Teleport usage</td>
    <td>teleport_per_min below the median / low percentiles</td>
    <td>Controlled, non-chaotic navigation.</td>
  </tr>

  <tr>
    <td>Frequent Teleport usage</td>
    <td>teleport_per_min in the highest percentiles</td>
    <td>Disordered exploration or continuous searching.</td>
  </tr>

  <tr>
    <td>No long inactive periods</td>
    <td>idle_gaps_gt60 = 0</td>
    <td>Continuously active session.</td>
  </tr>

  <tr>
    <td>Many long inactive periods</td>
    <td>idle_gaps_gt60 in the highest percentiles</td>
    <td>Long pauses / temporary abandonment.</td>
  </tr>

  <tr>
    <td>Stable post-Touch sequences</td>
    <td>entropy_next_after_touch low / low percentiles</td>
    <td>Actions after Touch are predictable and coherent.</td>
  </tr>

  <tr>
    <td>Unstable post-Touch sequences</td>
    <td>entropy_next_after_touch high / high percentiles</td>
    <td>Irregular workflow, trial and error.</td>
  </tr>
</table>

<h3 style="margin-top:16px;">Behavioral archetypes (guide)</h3>
<p>
  Archetypes are <b>multi-label</b> tags (max 2) assigned by combining sequence metrics.
  Thresholds are <b>data-driven</b> (quantiles/medians), therefore relative to your dataset.
</p>

<table>
  <tr>
    <th>Archetype</th>
    <th>Quantitative conditions (indicative)</th>
    <th>Behavioral interpretation</th>
  </tr>

  <tr>
    <td><b>Fast content consumer</b></td>
    <td>High Touch→Play (≥80th) + Fast Touch→Play (≤20th) + Low Pause usage (≤ median)</td>
    <td>Frequently starts content and does so quickly, with few interruptions.</td>
  </tr>

  <tr>
    <td><b>Reflective learner</b></td>
    <td>High Touch→Play (≥80th) + Slow Touch→Play (≥80th) + High Pause usage (≥80th)</td>
    <td>Interested in content but with longer decision times and more pause usage.</td>
  </tr>

  <tr>
    <td><b>Focused explorer</b></td>
    <td>High Touch→Play (≥80th) + Low entropy (≤ median) + Low Teleport usage (≤ median)</td>
    <td>Coherent and predictable path: follows a logical thread, with limited spatial exploration.</td>
  </tr>

  <tr>
    <td><b>Spatial explorer</b></td>
    <td>High Teleport usage (≥80th) + Low Touch→Play (≤20th) + Low Pause usage (≤ median)</td>
    <td>Moves a lot through space and rarely starts content; more “environmental” exploration.</td>
  </tr>

  <tr>
    <td><b>Zapping / scattered exploration</b></td>
    <td>High Teleport usage (≥80th) + High entropy (≥80th) + Fast Touch→Play (≤20th)</td>
    <td>Frenetic / variable behavior: jumps across actions and locations with low stability.</td>
  </tr>

  <tr>
    <td><b>Disoriented / struggling</b></td>
    <td>High entropy (≥80th) + High Pause usage (≥80th) + High Idle (≥80th and &gt;0)</td>
    <td>Unstable sequences with many interruptions and/or stalls: possible friction or confusion.</td>
  </tr>

  <tr>
    <td><b>Intermittent</b></td>
    <td>High Idle (≥80th and &gt;0) + High Pause usage (≥80th)</td>
    <td>Session in bursts: activity alternates with long stops and restarts.</td>
  </tr>

  <tr>
    <td><b>Passive observer</b></td>
    <td>Low Touch→Play (≤20th) + Low Teleport usage (≤ median) + Low entropy (≤ median)</td>
    <td>Limited and stable interaction: little exploration and few “strong” transitions.</td>
  </tr>
</table>
</div>
"""

    pat_cluster_html = "<p>No cluster summary available.</p>"
    if "cluster_summary" in pat and isinstance(pat["cluster_summary"], pd.DataFrame):
        if not pat["cluster_summary"].empty:
            pat_cluster_html = pat["cluster_summary"].to_html()

    best_k_txt = ""
    if not pd.isna(clust_payload.get("best_k", np.nan)):
        best_k_txt = f"(k={int(clust_payload['best_k'])})"

    # -----------------------
    # PCA DESCRIPTION BLOCK
    # -----------------------
    pca_explain_html = """
  <h3>What do PC1 and PC2 represent in the PCA plot?</h3>

  <p>
    The PCA plot shows sessions projected into two new dimensions called <b>PC1</b> and <b>PC2</b>.
    These dimensions <b>do not directly correspond to individual features</b> (such as Pause, Touch, or Teleport),
    but are combinations of all features used for clustering.<br/><br/>

    PCA (Principal Component Analysis) is used to reduce data complexity while preserving as much information
    as possible about the differences between sessions, so that they can be visualized in a 2D plot.
  </p>

  <p>
    <b>PC1 (First Principal Component)</b> is the dimension that explains most of the differences between
    sessions. It represents the main “direction” along which behaviors differ the most.
    Sessions far apart on PC1 have very different interaction strategies.
  </p>

  <p>
    <b>PC2 (Second Principal Component)</b> is a second dimension, independent from PC1, that captures
    other important differences not explained by the first component. It helps distinguish sessions that may
    be similar on PC1 but different in other behavioral aspects.
  </p>

  <p>
    <b>How to read the plot:</b><br/>
    • each point is a session;<br/>
    • nearby points = similar sessions;<br/>
    • distant points = different sessions;<br/>
    • color indicates the assigned cluster.<br/>
    If clusters are separated in PCA space, it means the model identified distinct behavioral groups.
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

<h1>Report — Preprocess + EDA + Feature Selection + Clustering + Sequence + Scoring</h1>

<section>
  <h2>Overview</h2>
  <p>Input: <code>{INPUT_DIR}/*.txt</code></p>
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
  <h3>Descriptive statistics</h3>

  <p>
    The table below shows a statistical summary of the numerical features extracted from all sessions.
    Each row represents a feature, while each column describes an aspect of its distribution:<br/><br/>
    <b>count</b>: number of sessions with a valid value.<br/>
    <b>mean</b>: average value of the feature.<br/>
    <b>std</b>: standard deviation, indicating variability across sessions.<br/>
    <b>min</b>: minimum observed value.<br/>
    <b>25%</b>: first quartile (25% of sessions have values ≤ this value).<br/>
    <b>50%</b>: median (50% of sessions have values ≤ this value).<br/>
    <b>75%</b>: third quartile (75% of sessions have values ≤ this value).<br/>
    <b>max</b>: maximum observed value.<br/><br/>
    This table helps identify outliers, understand session variability,
    and assess which features are useful for analysis or clustering.
  </p>

  {eda_payload["desc_html"]}

  <h3>Histograms</h3>

  <p>
    Histograms show the distribution of each numerical feature analyzed. Each plot represents a single feature and
    shows how its values are distributed across sessions:<br/><br/>
    <b>X-axis</b>: numerical values of the feature (e.g., duration, number of events, average time, etc.)<br/>
    <b>Y-axis</b>: frequency, i.e. how many sessions fall into that value interval<br/>
    <b>Legend</b>: name of the feature shown in the plot.<br/><br/>
    Histograms help identify outliers, concentrations, dispersion, and useful dataset patterns.
  </p>

  <div class="figure">
    <img src="{eda_payload['hist_b64']}" alt="Histograms"/>
  </div>

  <h3>Correlation matrix</h3>

  <p>
    The correlation matrix shows the linear relationship between all numerical features analyzed.
    Each cell contains a value between -1 and +1, where:<br/><br/>
    <b>+1</b>: perfect positive correlation (the two features increase together);<br/>
    <b>0</b>: no linear correlation;<br/>
    <b>-1</b>: perfect negative correlation (one increases while the other decreases).<br/><br/>
    In the heatmap, more intense colors indicate stronger relationships:
    warm tones represent positive correlations, while cool tones indicate negative correlations.
    This analysis helps identify redundant features, recurring patterns, and possible relationships between variables
    that are useful for understanding the dataset and for later stages (feature selection and clustering).
  </p>

  <div class="figure">
    <img src="{eda_payload['corr_b64']}" alt="Correlation matrix"/>
  </div>
</section>

<hr/>

<section>
  <h2>Feature Selection</h2>

  <p>
    This section keeps only the features that are truly useful for analysis and clustering.
    The procedure is based on two main stages:<br/><br/>

    <b>1) Variance Threshold</b>: removes features with near-zero variance, i.e. variables that take
    almost the same value all the time and therefore add no useful information.<br/><br/>

    <b>2) Correlation Threshold</b>: removes one of two highly correlated features
    (correlation &gt; 0.9), avoiding redundancy and preserving the more representative variable.<br/><br/>

    This selection reduces noise, removes redundancy, and improves the quality
    of subsequent analyses, especially clustering.
  </p>

  <p><b>Retained features:</b> {kept_list}</p>

  <h3>Drop log (variance/correlation)</h3>
  {dropped_html}
</section>

<hr/>

<section>
  <h2>Clustering</h2>

  <p>
    Clustering groups sessions automatically according to their behavioral characteristics.
    Each cluster represents a set of sessions that are similar to each other, identifying distinct
    patterns such as faster, more exploratory, more cautious, or more systematic users.<br/><br/>

    In this report, selected features are scaled with StandardScaler.
    Five algorithms are compared (KMeans, Agglomerative, Spectral, GMM, DBSCAN). For models that require
    a number of clusters, the following grid is tested: <b>k = 3, 4, 5, 6, 7</b>.<br/><br/>

    <b>Best model selection criterion</b>:
    results are ranked by:
    <ol>
      <li><b>Silhouette score</b> (higher is better)</li>
      <li><b>Number of clusters obtained</b> (higher is better, when silhouette is tied)</li>
      <li><b>Davies–Bouldin index</b> (lower is better)</li>
    </ol>

    The top-ranked model is labeled as the <b>"Best model"</b> and is used
    for cluster assignment and profile computation.
  </p>

  <p><b>Best model:</b> <code>{clust_payload['best_name']}</code> {best_k_txt}</p>

  {pca_explain_html}

  <h3>PCA scatter — best</h3>
  <div class="figure">
    <img src="{pca_b64}" alt="PCA scatter"/>
  </div>

  <h3>PCA scatter — next best 4</h3>
  <p class="small">
    The following plots show the next 4 models in the ranking, to visually compare
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
    The best model is chosen according to the criterion described above.
  </p>

  {res_html}

  <h3>Cluster profiles</h3>

  <p>
    This table shows the <b>average profile</b> of each cluster, obtained by computing the mean and median of
    the selected features for all sessions belonging to the same group.<br/><br/>

    <b>Note:</b> the profiles are based on the assignments of the <b>best model</b>
    (<code>{clust_payload['best_name']}</code>).<br/><br/>

    These values define the <b>centroid</b> of the cluster and help explain which features truly distinguish
    the groups.
  </p>

  {profiles_html}
</section>

<hr/>

<section>
  <h2>Sequence Analysis</h2>

  <p>
    Sequence Analysis examines the temporal order of events generated during sessions,
    helping understand how interaction evolves step by step. Unlike simple aggregate
    statistics, the goal here is to observe the <b>flow</b> of actions
    (Touch, PlayAnnotation, Grab, Release, Teleport, etc.) and the patterns
    with which they unfold over time.<br/><br/>

    The following are shown below:
    <ul>
      <li>the <b>transition matrix</b>, describing the probabilities of moving from one action to the next;</li>
      <li>the most frequent <b>bigrams</b> and <b>trigrams</b> (sequences of 2 or 3 consecutive actions);</li>
      <li><b>dwell time</b> statistics, i.e. time intervals between consecutive events of the same action.</li>
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
</section>

<hr/>

<section>
  <h2>Sequence Pattern Detection</h2>

  <p>
    This section analyzes interaction patterns at the individual session level using
    metrics derived from event sequences (e.g., Touch, PlayAnnotation, Pause, Teleport).
    Instead of fixed thresholds, this version uses <b>data-driven</b> thresholds based on quantiles.<br/><br/>

    Each session receives:
    <ul>
      <li><b>effective_flags</b>: positive patterns relative to the other sessions</li>
      <li><b>problem_flags</b>: potentially problematic patterns</li>
      <li><b>pattern_score</b>: sum (+1 for each effective flag, −1 for each problematic flag)</li>
    </ul>

    Flags also include the <b>percentile</b>, so the relative position in the distribution is clear.
  </p>

  <h3>Quick overview</h3>
  {pat_summary_html}

  <div class="figure">
    <img src="{pat_hist_b64}" alt="Pattern score distribution"/>
  </div>

  {flags_guide_html}

  <h3>Top 5 most critical sessions</h3>

  <div class="table-scroll">
    {top_worst_html}
  </div>

  <h3>Top 5 most linear sessions</h3>
  <div class="table-scroll">
    {top_best_html}
  </div>

  <h3>All sessions (complete metrics &amp; flags)</h3>
  <p class="small">
    The first column shows the <b>sessionID</b> colored according to pattern_score:
    green = more linear, yellow = intermediate, red = critical.
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
</section>

<hr/>

<section>
  <h2>Per-session summary</h2>
  <p class="small">
    All sessions are shown in sessionID order.
    Each panel includes cluster, pattern_score (colored badge), flags, sequence metrics,
    full counts, and timeline.
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

    """Run the full Mode 2 pipeline.

    Parameters
    ----------
    log_dir : str or None
        Path to the folder containing Mode 2 logs (.txt/.csv). If None, the user is prompted.
    out_root : str or None
        Path to the output root folder where the 'Report' folder will be created. If None, the user is prompted.
    skip_eda : bool
        If True, skips the heavy EDA plots section (useful on low-memory machines).
    export_processed_csv : bool
        If True, exports cleaned per-session logs as CSV to Report/ProcessedLogs.
    """

    if log_dir is None:
        log_dir = input("Enter the path to the log folder (.txt/.csv): ").strip().strip('"')
    if out_root is None:
        out_root = input("Enter the path to the folder where the 'Report' folder will be created: ").strip().strip('"')

    if not os.path.isdir(log_dir):
        print(f"[ERROR] Log folder does not exist: {log_dir}")
        return

    if not os.path.isdir(out_root):
        print(f"[INFO] Output folder does not exist, creating it: {out_root}")
        os.makedirs(out_root, exist_ok=True)

    INPUT_DIR = log_dir
    OUT_DIR = Path(out_root) / "Report"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FEATURES_CSV = OUT_DIR / "Mode2Logs.csv"
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

    num_cols = feats.select_dtypes(include=[np.number]).columns.tolist()
    X_full = feats[num_cols].copy()
    eda_payload = eda_section(X_full)

    X_sel, kept_cols, dropped_tbl = variance_corr_filter(X_full)

    clust = best_of_five_clustering(X_sel)

    seqs = sessions_to_sequences(sessions_raw)
    bi_df = ngram_counts(seqs, n=2, top=30)
    tri_df = ngram_counts(seqs, n=3, top=30)
    acts, P = transition_matrix(sessions_raw)
    trans_b64 = transition_heatmap_b64(acts, P)
    dwell_df = dwell_time_stats(sessions_raw)
    seq_payload = {
        "bigrams": bi_df,
        "trigrams": tri_df,
        "trans_b64": trans_b64,
        "dwell_df": dwell_df,
    }

    pat = sequence_pattern_detection(sessions_raw, feats, clust)

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

    parser = argparse.ArgumentParser(description="WP4 Learning Analytics — Mode 2 pipeline (D4.1).")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to Mode 2 logs folder (txt/csv). If omitted, you will be prompted.")
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