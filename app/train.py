# train.py
import os
import numpy as np
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.decomposition import PCA

from config import SETTINGS


# ----------------------------
# Helpers
# ----------------------------
def get_sensor_cols(df):
    return [c for c in df.columns if c.lower().startswith("sensor")]

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def fit_gmm_bic(Xp, k_range):
    best_bic = np.inf
    best_gmm = None
    rows = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=SETTINGS.COV_TYPE,
            init_params=SETTINGS.INIT_PARAMS,
            n_init=SETTINGS.N_INIT,
            max_iter=SETTINGS.MAX_ITER,
            reg_covar=SETTINGS.REG_COVAR,
            random_state=SETTINGS.RANDOM_SEED,
        ).fit(Xp)

        bic = gmm.bic(Xp)
        aic = gmm.aic(Xp)
        rows.append({"k": k, "bic": bic, "aic": aic})
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    return best_gmm, pd.DataFrame(rows)

def cluster_label_mapping(cluster_ids, y, labeled_mask, k):
    cluster_to_label, support, purity = {}, {}, {}
    y_vals = y.values
    lm = labeled_mask.values

    for c in range(k):
        idx = lm & (cluster_ids == c)
        s = int(idx.sum())
        support[c] = s
        if s == 0:
            cluster_to_label[c] = None
            purity[c] = None
        else:
            lbls = y_vals[idx].astype(int)
            vals, cnts = np.unique(lbls, return_counts=True)
            maj = int(vals[np.argmax(cnts)])
            cluster_to_label[c] = maj
            purity[c] = float(np.max(cnts) / s)
    return cluster_to_label, support, purity

def compute_cluster_cards(cluster_ids, max_prob, cluster_to_label, support, purity, k):
    sizes = pd.Series(cluster_ids).value_counts().sort_index()
    cards = pd.DataFrame({
        "cluster": range(k),
        "size_total": [int(sizes.get(c, 0)) for c in range(k)],
        "labeled_support": [support.get(c, 0) for c in range(k)],
        "purity_on_labeled": [purity.get(c, None) for c in range(k)],
        "major_label": [cluster_to_label.get(c, None) for c in range(k)],
        "avg_confidence": pd.Series(max_prob).groupby(cluster_ids).mean().reindex(range(k)).values,
    }).sort_values(["labeled_support","purity_on_labeled","size_total"], ascending=[False,False,False])
    return cards.reset_index(drop=True)

def stability_ari(Xp, base_assign, k, runs=8):
    rows = []
    for i in range(runs):
        seed = SETTINGS.RANDOM_SEED + 100 + i
        gmm_i = GaussianMixture(
            n_components=k,
            covariance_type=SETTINGS.COV_TYPE,
            init_params=SETTINGS.INIT_PARAMS,
            n_init=SETTINGS.N_INIT,
            max_iter=SETTINGS.MAX_ITER,
            reg_covar=SETTINGS.REG_COVAR,
            random_state=seed,
        ).fit(Xp)
        assign_i = gmm_i.predict(Xp)
        rows.append({
            "run": i,
            "seed": seed,
            "ari_vs_baseline": adjusted_rand_score(base_assign, assign_i)
        })
    return pd.DataFrame(rows)

def plot_bic_aic(sel_df, outdir):
    plt.figure(figsize=(7,4))
    plt.plot(sel_df["k"], sel_df["bic"], marker="o", label="BIC")
    plt.plot(sel_df["k"], sel_df["aic"], marker="o", label="AIC")
    plt.xlabel("K (components)")
    plt.ylabel("Score (lower is better)")
    plt.title("GMM model selection")
    plt.legend()
    save_fig(os.path.join(outdir, "01_bic_aic_curve.png"))

def plot_stability(stab_df, outdir):
    plt.figure(figsize=(7,4))
    plt.bar(stab_df["run"], stab_df["ari_vs_baseline"])
    plt.xlabel("Run")
    plt.ylabel("ARI vs baseline")
    plt.title("Stability: assignment consistency across seeds")
    save_fig(os.path.join(outdir, "02_stability_ari.png"))

def plot_operational_tradeoff(df_trade, outdir):
    plt.figure(figsize=(7,4))
    plt.plot(df_trade["threshold"], df_trade["labeled_coverage"], marker="o", label="labeled coverage")
    plt.plot(df_trade["threshold"], df_trade["labeled_accuracy"], marker="o", label="labeled accuracy")
    plt.plot(df_trade["threshold"], df_trade["operational_auto_rate_all"], marker="o", label="auto-rate (all)")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Value")
    plt.title("Operational trade-off: coverage, accuracy, auto-rate")
    plt.legend()
    save_fig(os.path.join(outdir, "03_tradeoff_curve.png"))

def plot_calibration(cal_df, outdir):
    cal_plot = cal_df[cal_df["count"] > 0].copy()
    plt.figure(figsize=(7,4))
    plt.plot(cal_plot["avg_conf"], cal_plot["accuracy"], marker="o")
    for _, r in cal_plot.iterrows():
        plt.text(r["avg_conf"], r["accuracy"], str(int(r["count"])), fontsize=9)
    plt.xlabel("Average confidence in bin")
    plt.ylabel("Empirical accuracy (labeled)")
    plt.title("Calibration check (counts annotated)")
    save_fig(os.path.join(outdir, "04_calibration.png"))

def plot_pca_option_a(Xp, df, labeled_mask, outdir):
    pca2 = PCA(n_components=2, random_state=SETTINGS.RANDOM_SEED)
    X2 = pca2.fit_transform(Xp)

    plt.figure(figsize=(7,5))
    plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.10, label="All / unlabeled events")

    if labeled_mask.sum() > 0:
        y_lab = df.loc[labeled_mask, SETTINGS.LABEL_COL].astype(int).values
        X2_lab = X2[labeled_mask.values]
        for lab in sorted(np.unique(y_lab)):
            idx = (y_lab == lab)
            plt.scatter(X2_lab[idx, 0], X2_lab[idx, 1], s=80, alpha=0.95, label=f"Label {lab}")

    plt.title("PCA scatter — all events (grey) + labeled points (colored by class)")
    plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend(fontsize=9)
    save_fig(os.path.join(outdir, "05_pca_all_bg_labeled_by_class.png"))

def plot_label_distribution_per_cluster(cluster_ids, df, labeled_mask, outdir):
    if labeled_mask.sum() == 0:
        return
    tmp = pd.DataFrame({
        "cluster": cluster_ids[labeled_mask.values],
        "label": df.loc[labeled_mask, SETTINGS.LABEL_COL].astype(int).values
    })
    ct = pd.crosstab(tmp["cluster"], tmp["label"]).sort_index()
    ct.plot(kind="bar", stacked=True, figsize=(8,4))
    plt.title("Labeled distribution per cluster (purity sanity check)")
    plt.xlabel("Cluster")
    plt.ylabel("Count (labeled only)")
    save_fig(os.path.join(outdir, "06_label_distribution_per_cluster.png"))


def train():
    outdir = SETTINGS.OUTDIR
    os.makedirs(outdir, exist_ok=True)
    np.random.seed(SETTINGS.RANDOM_SEED)

    df = pd.read_csv(SETTINGS.DATA_PATH)
    sensor_cols = get_sensor_cols(df)

    X = df[sensor_cols]
    y = df[SETTINGS.LABEL_COL]
    labeled_mask = y.notna()

    print(f"Rows={len(df)}, Features={len(sensor_cols)}")
    print(f"Labeled={int(labeled_mask.sum())}, Unlabeled={int((~labeled_mask).sum())}")

    preprocess = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    Xp = preprocess.fit_transform(X)

    # model selection
    gmm, sel = fit_gmm_bic(Xp, range(SETTINGS.K_MIN, SETTINGS.K_MAX + 1))
    sel.to_csv(os.path.join(outdir, "model_selection_bic_aic.csv"), index=False)
    K = int(gmm.n_components)
    print("Selected K:", K)

    plot_bic_aic(sel, outdir)

    cluster_ids = gmm.predict(Xp)
    probs = gmm.predict_proba(Xp)
    max_prob = probs.max(axis=1)

    cluster_to_label, support, purity = cluster_label_mapping(cluster_ids, y, labeled_mask, K)
    cards = compute_cluster_cards(cluster_ids, max_prob, cluster_to_label, support, purity, K)
    cards.to_csv(os.path.join(outdir, "cluster_cards.csv"), index=False)

    pred_class = pd.Series(cluster_ids).map(cluster_to_label)

    reliable_cluster = np.array([
        (support.get(c, 0) >= SETTINGS.MIN_SUPPORT)
        and (purity.get(c) is not None)
        and (float(purity[c]) >= SETTINGS.MIN_PURITY)
        for c in range(K)
    ], dtype=bool)
    reliable_mask_all = reliable_cluster[cluster_ids]

    auto_mask = (max_prob >= SETTINGS.CONF_THRESH) & reliable_mask_all & pred_class.notna()

    decisions = pd.DataFrame({
        "cluster": cluster_ids,
        "confidence": max_prob,
        "predicted_class": pred_class,
        "auto_assign": auto_mask,
        "true_label": y
    })
    decisions.to_csv(os.path.join(outdir, "decisions.csv"), index=False)

    # evaluate on labeled
    valid = labeled_mask & pred_class.notna()
    if valid.sum() > 0:
        y_true = y[valid].astype(int).values
        y_pred = pred_class[valid].astype(int).values
        acc = accuracy_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        print(f"Labeled-only: ACC={acc:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}")
    else:
        y_true = np.array([])
        y_pred = np.array([])

    auto_rate = float(auto_mask.mean())
    review_rate = 1.0 - auto_rate
    print(f"Operational: auto_rate={auto_rate:.3f}, review_rate={review_rate:.3f}")

    # stability
    stab = stability_ari(Xp, cluster_ids, K, runs=8)
    stab.to_csv(os.path.join(outdir, "stability_ari.csv"), index=False)
    plot_stability(stab, outdir)

    # trade-off curve
    CONF_GRID = np.arange(0.55, 0.96, 0.05)
    trade_rows = []
    if valid.sum() > 0:
        conf_l = max_prob[valid.values]
        for t in CONF_GRID:
            keep_l = conf_l >= t
            labeled_coverage = float(keep_l.mean())
            labeled_acc = float((y_true[keep_l] == y_pred[keep_l]).mean()) if keep_l.sum() > 0 else np.nan
            operational_auto_rate_all = float(((max_prob >= t) & reliable_mask_all & pred_class.notna().values).mean())
            trade_rows.append({
                "threshold": float(t),
                "labeled_coverage": labeled_coverage,
                "labeled_accuracy": labeled_acc,
                "operational_auto_rate_all": operational_auto_rate_all
            })
    trade_df = pd.DataFrame(trade_rows)
    trade_df.to_csv(os.path.join(outdir, "tradeoff_curve.csv"), index=False)
    if len(trade_df) > 0:
        plot_operational_tradeoff(trade_df, outdir)

    # calibration
    CAL_BINS = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if valid.sum() > 0:
        conf_l = max_prob[valid.values]
        correct = (y_true == y_pred).astype(int)
        bin_ids = np.digitize(conf_l, CAL_BINS, right=True)
        cal_rows = []
        for b in range(1, len(CAL_BINS)):
            in_bin = bin_ids == b
            if in_bin.sum() == 0:
                cal_rows.append({"bin": f"({CAL_BINS[b-1]:.1f},{CAL_BINS[b]:.1f}]",
                                 "count": 0, "avg_conf": np.nan, "accuracy": np.nan})
            else:
                cal_rows.append({"bin": f"({CAL_BINS[b-1]:.1f},{CAL_BINS[b]:.1f}]",
                                 "count": int(in_bin.sum()),
                                 "avg_conf": float(np.mean(conf_l[in_bin])),
                                 "accuracy": float(np.mean(correct[in_bin]))})
        cal_df = pd.DataFrame(cal_rows)
        cal_df.to_csv(os.path.join(outdir, "calibration_bins.csv"), index=False)
        plot_calibration(cal_df, outdir)

    # PCA
    plot_pca_option_a(Xp, df, labeled_mask, outdir)
    plot_label_distribution_per_cluster(cluster_ids, df, labeled_mask, outdir)

    # silhouette
    sil = silhouette_score(Xp, cluster_ids, metric="euclidean")
    pd.Series({"silhouette_full": float(sil)}).to_csv(os.path.join(outdir, "silhouette.csv"))
    print(f"Silhouette (full): {sil:.3f}")

    # save artifacts
    dump(preprocess, os.path.join(outdir, "preprocess.joblib"))
    dump(gmm, os.path.join(outdir, "gmm.joblib"))
    dump(cluster_to_label, os.path.join(outdir, "cluster_to_major_label.joblib"))
    dump(support, os.path.join(outdir, "cluster_support.joblib"))
    dump(purity, os.path.join(outdir, "cluster_purity.joblib"))

    print(f"\nSaved outputs to: {outdir}/")


if __name__ == "__main__":
    train()
