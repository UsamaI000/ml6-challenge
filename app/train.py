import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    f1_score,
)

from config import TrainConfig


# ----------------------------
# Helpers
# ----------------------------
def safe_is_not_none(arr_obj):
    return np.array([x is not None for x in arr_obj], dtype=bool)


def fit_gmm_select_k(Z_all, cfg: TrainConfig, random_state: int):
    """Fit GMMs for k in [k_min, k_max], return best_k by BIC and eval_df + gmms."""
    rows = []
    gmms = {}

    for k in range(cfg.k_min, cfg.k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cfg.cov_type,
            n_init=cfg.n_init,
            reg_covar=cfg.reg_covar,
            random_state=random_state,
        )
        gmm.fit(Z_all)

        bic = gmm.bic(Z_all)
        aic = gmm.aic(Z_all)
        hard = gmm.predict(Z_all)

        sil = np.nan
        if 1 < len(np.unique(hard)) < len(Z_all):
            sil = silhouette_score(Z_all, hard)

        rows.append({"k": k, "bic": bic, "aic": aic, "silhouette": sil})
        gmms[k] = gmm

    eval_df = pd.DataFrame(rows).sort_values("k")
    best_k = int(eval_df.loc[eval_df["bic"].idxmin(), "k"])
    return best_k, eval_df, gmms


def cluster_mapping_and_purity(cluster_train, y_train, k: int):
    """Majority-vote mapping + purity/support computed on *train anchors only*."""
    cluster_to_label = {}
    cluster_purity = {}
    cluster_support = {}

    for c in range(k):
        idx = np.where(cluster_train == c)[0]
        cluster_support[c] = int(len(idx))
        if len(idx) == 0:
            cluster_to_label[c] = None
            cluster_purity[c] = 0.0
            continue

        labels_here = y_train[idx]
        vals, counts = np.unique(labels_here, return_counts=True)
        maj_label = int(vals[np.argmax(counts)])
        purity = float(counts.max() / counts.sum())

        cluster_to_label[c] = maj_label
        cluster_purity[c] = purity

    return cluster_to_label, cluster_purity, cluster_support


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_curve(xs, ys, title, xlabel, ylabel, out_path: Path | None):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xs)
    plt.grid(True, linestyle="--", linewidth=0.5)
    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


# ----------------------------
# Core
# ----------------------------
def load_data(cfg: TrainConfig):
    df = pd.read_csv(cfg.csv_path)

    sensor_cols = [c for c in df.columns if c.startswith(cfg.sensor_prefix)]
    if not sensor_cols:
        raise ValueError(f"No sensor columns found with prefix '{cfg.sensor_prefix}'")

    if cfg.label_col not in df.columns:
        raise ValueError(f"Expected '{cfg.label_col}' column with NaN for unlabeled and ints for labeled.")

    X = df[sensor_cols].values.astype("float64")
    y_raw = df[cfg.label_col].values

    labeled_mask = ~pd.isna(y_raw)
    unlabeled_mask = ~labeled_mask

    if labeled_mask.sum() < 5:
        raise ValueError("Too few labeled points to train NCA meaningfully.")

    y_l = y_raw[labeled_mask].astype(int)
    classes, class_counts = np.unique(y_l, return_counts=True)

    if len(classes) < 2:
        raise ValueError("NCA needs at least 2 distinct labels in the labeled subset.")

    print(
        f"Loaded {len(df)} rows | Labeled {labeled_mask.sum()} | Unlabeled {unlabeled_mask.sum()} | "
        f"Labels: {sorted(classes.tolist())} | Counts: {dict(zip(classes.tolist(), class_counts.tolist()))}"
    )

    return df, sensor_cols, X, y_raw, labeled_mask, unlabeled_mask, y_l


def cv_evaluate(cfg: TrainConfig, X, y_raw, labeled_mask, unlabeled_mask, y_l):
    idx_labeled = np.where(labeled_mask)[0]
    classes, class_counts = np.unique(y_l, return_counts=True)

    # Adjust folds if necessary
    min_class = int(class_counts.min())
    cv_folds = cfg.cv_folds
    if cv_folds > min_class:
        cv_folds = max(2, min_class)
        print(f"[CV] Reducing cv_folds to {cv_folds} due to small class counts.")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=cfg.cv_shuffle, random_state=cfg.cv_seed)

    cv_rows = []
    fold_num = 0

    for train_pos, test_pos in skf.split(idx_labeled, y_l):
        fold_num += 1
        train_idx = idx_labeled[train_pos]
        test_idx = idx_labeled[test_pos]

        # Scale on TRAIN anchors only (leakage-safe)
        scaler = StandardScaler()
        scaler.fit(X[train_idx])
        X_scaled_all = scaler.transform(X)

        # NCA on TRAIN anchors only
        nca = NeighborhoodComponentsAnalysis(
            n_components=min(cfg.nca_dim, X_scaled_all.shape[1]),
            max_iter=cfg.nca_max_iter,
            tol=cfg.nca_tol,
            random_state=cfg.cv_seed + fold_num,
        )
        nca.fit(X_scaled_all[train_idx], y_raw[train_idx].astype(int))
        Z_all = nca.transform(X_scaled_all)

        # GMM selection on ALL points
        best_k, _, gmms = fit_gmm_select_k(Z_all, cfg, random_state=cfg.cv_seed + fold_num)
        best_gmm = gmms[best_k]

        cluster_all = best_gmm.predict(Z_all)
        post_all = best_gmm.predict_proba(Z_all)
        gmm_conf_all = post_all.max(axis=1)

        # Mapping/purity from TRAIN anchors only
        cluster_train = cluster_all[train_idx]
        y_train = y_raw[train_idx].astype(int)
        cluster_to_label, cluster_purity, cluster_support = cluster_mapping_and_purity(cluster_train, y_train, best_k)

        pred_label_all = np.array([cluster_to_label.get(c, None) for c in cluster_all], dtype=object)
        pred_is_valid_all = safe_is_not_none(pred_label_all)

        purity_arr = np.array([cluster_purity.get(c, 0.0) for c in range(best_k)], dtype=float)
        support_arr = np.array([cluster_support.get(c, 0) for c in range(best_k)], dtype=int)

        # Business confidence + hard gates
        business_conf_all = gmm_conf_all * purity_arr[cluster_all]
        bad = (support_arr[cluster_all] < cfg.min_support) | (purity_arr[cluster_all] < cfg.min_purity)
        business_conf_all[bad] = 0.0

        hc_unlabeled = unlabeled_mask & (business_conf_all >= cfg.conf_thresh) & pred_is_valid_all
        hc_unlabeled_count = int(hc_unlabeled.sum())

        # Evaluate on TEST anchors only
        y_test = y_raw[test_idx].astype(int)
        cluster_test = cluster_all[test_idx]

        ari_test = adjusted_rand_score(y_test, cluster_test)
        nmi_test = normalized_mutual_info_score(y_test, cluster_test)

        pred_test = pred_label_all[test_idx]
        pred_test_valid = safe_is_not_none(pred_test)

        acc = np.nan
        f1m = np.nan
        if pred_test_valid.sum() > 0:
            y_pred_test = pred_test[pred_test_valid].astype(int)
            y_true_test = y_test[pred_test_valid]
            acc = accuracy_score(y_true_test, y_pred_test)
            f1m = f1_score(y_true_test, y_pred_test, average="macro")

        cv_rows.append(
            {
                "fold": fold_num,
                "best_k_bic": best_k,
                "ari_test": ari_test,
                "nmi_test": nmi_test,
                "acc_test_mapped": acc,
                "macro_f1_test_mapped": f1m,
                "hc_unlabeled_count": hc_unlabeled_count,
                "test_mapped_coverage": float(pred_test_valid.mean()),
            }
        )

    cv_df = pd.DataFrame(cv_rows)
    print("\n====================")
    print("CV RESULTS (anchors only, leakage-safe)")
    print("====================")
    print(cv_df.to_string(index=False))

    for metric in ["ari_test", "nmi_test", "acc_test_mapped", "macro_f1_test_mapped", "test_mapped_coverage"]:
        vals = cv_df[metric].values.astype(float)
        print(f"[CV] {metric}: mean={np.nanmean(vals):.3f} std={np.nanstd(vals):.3f}")

    print(
        f"[CV] High-confidence unlabeled (>= {cfg.conf_thresh}) avg: {cv_df['hc_unlabeled_count'].mean():.1f} "
        f"(std={cv_df['hc_unlabeled_count'].std():.1f}) out of {int(unlabeled_mask.sum())}"
    )

    return cv_df


def final_train_and_export(cfg: TrainConfig, df, sensor_cols, X, y_raw, labeled_mask, unlabeled_mask, y_l):
    # Final training for deployment artifact (fit on ALL data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nca = NeighborhoodComponentsAnalysis(
        n_components=min(cfg.nca_dim, X_scaled.shape[1]),
        max_iter=cfg.nca_max_iter,
        tol=cfg.nca_tol,
        random_state=cfg.cv_seed,
    )
    nca.fit(X_scaled[labeled_mask], y_l)

    Z = nca.transform(X_scaled)
    print(f"\nNCA transformed shape: {Z.shape} (n_components={Z.shape[1]})")

    best_k, eval_df, gmms = fit_gmm_select_k(Z, cfg, random_state=cfg.cv_seed)
    best_gmm = gmms[best_k]

    print("\nModel selection table (final fit):")
    print(eval_df.to_string(index=False))
    print(f"\nChosen k by BIC (final): {best_k}")

    # Save selection table
    eval_df.to_csv(cfg.artifacts_path / cfg.final_model_selection_name, index=False)

    # Plots (saved)
    if cfg.save_plots:
        plots_dir = cfg.artifacts_path / "plots"
        ensure_dir(plots_dir)
        ks = eval_df["k"].values
        plot_curve(ks, eval_df["bic"].values, "GMM BIC vs k (final fit)", "k", "BIC", plots_dir / "bic.png")
        plot_curve(ks, eval_df["aic"].values, "GMM AIC vs k (final fit)", "k", "AIC", plots_dir / "aic.png")
        plot_curve(ks, eval_df["silhouette"].values, "Silhouette vs k (final fit)", "k", "Silhouette", plots_dir / "silhouette.png")

    # Assignments/confidence
    cluster = best_gmm.predict(Z)
    post = best_gmm.predict_proba(Z)
    gmm_conf = post.max(axis=1)

    # Mapping + purity using ALL anchors (deployment artifact)
    cl_lab = cluster[labeled_mask]
    y_lab = y_l
    cluster_to_label, cluster_purity, cluster_support = cluster_mapping_and_purity(cl_lab, y_lab, best_k)

    purity_arr = np.array([cluster_purity.get(c, 0.0) for c in range(best_k)], dtype=float)
    support_arr = np.array([cluster_support.get(c, 0) for c in range(best_k)], dtype=int)

    pred_label = np.array([cluster_to_label.get(c, None) for c in cluster], dtype=object)
    pred_is_valid = safe_is_not_none(pred_label)

    # Business confidence + hard gates
    business_conf = gmm_conf * purity_arr[cluster]
    bad = (support_arr[cluster] < cfg.min_support) | (purity_arr[cluster] < cfg.min_purity)
    business_conf[bad] = 0.0

    hc_unlabeled = unlabeled_mask & (business_conf >= cfg.conf_thresh) & pred_is_valid
    print(
        f"\nHigh-confidence unlabeled assignments (final) using BusinessConfidence (>= {cfg.conf_thresh}): "
        f"{int(hc_unlabeled.sum())} out of {int(unlabeled_mask.sum())}"
    )

    # Save results CSV
    result = df.copy()
    result["NCA_dim"] = Z.shape[1]
    result["GMM_k"] = best_k
    result["Cluster"] = cluster
    result["GMMConfidence"] = gmm_conf
    result["ClusterPurity"] = purity_arr[cluster]
    result["ClusterSupport"] = support_arr[cluster]
    result["BusinessConfidence"] = business_conf
    result["PredictedLabel"] = pred_label

    out_csv = cfg.artifacts_path / cfg.results_csv_name
    result.to_csv(out_csv, index=False)
    print("\nSaved results CSV:", out_csv)

    # Save model bundle for inference
    bundle = {
        "sensor_cols": sensor_cols,
        "label_col": cfg.label_col,
        "sensor_prefix": cfg.sensor_prefix,
        "scaler": scaler,
        "nca": nca,
        "gmm": best_gmm,
        "best_k": best_k,
        "cluster_to_label": cluster_to_label,
        "cluster_purity": cluster_purity,
        "cluster_support": cluster_support,
        "min_support": cfg.min_support,
        "min_purity": cfg.min_purity,
        "conf_thresh": cfg.conf_thresh,
    }

    joblib.dump(bundle, cfg.model_path)
    print("Saved model bundle:", cfg.model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to training CSV.")
    parser.add_argument("--artifacts", type=str, default=None, help="Artifacts output directory.")
    args = parser.parse_args()

    cfg = TrainConfig(
        csv_path=args.csv or TrainConfig().csv_path,
        artifacts_dir=args.artifacts or TrainConfig().artifacts_dir,
    )

    ensure_dir(cfg.artifacts_path)

    df, sensor_cols, X, y_raw, labeled_mask, unlabeled_mask, y_l = load_data(cfg)

    cv_df = cv_evaluate(cfg, X, y_raw, labeled_mask, unlabeled_mask, y_l)
    cv_df.to_csv(cfg.artifacts_path / cfg.cv_metrics_name, index=False)
    print("Saved CV metrics:", cfg.artifacts_path / cfg.cv_metrics_name)

    final_train_and_export(cfg, df, sensor_cols, X, y_raw, labeled_mask, unlabeled_mask, y_l)


if __name__ == "__main__":
    main()
