# inference.py
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import load

from config import SETTINGS


@dataclass
class PredictionResult:
    cluster: int
    predicted_class: int | None
    confidence: float
    auto_assign: bool
    suggest_assign: bool
    reason: str
    cluster_support: int
    cluster_purity: float | None


class InferenceEngine:
    """
    Loads artifacts produced by train.py and provides predict_one()/predict_batch().
    Reliability gates are computed from TRUE-label support/purity only
    (cluster_support_true/joblib + cluster_purity_true.joblib).
    """

    def __init__(self, model_dir: str = SETTINGS.OUTDIR):
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        req = [
            "preprocess.joblib",
            "gmm.joblib",
            "cluster_to_major_label.joblib",
            "sensor_cols.joblib",
            "cluster_support_true.joblib",
            "cluster_purity_true.joblib",
        ]
        for f in req:
            path = os.path.join(self.model_dir, f)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing artifact: {path}. Run `python train.py` first.")

        self.preprocess = load(os.path.join(self.model_dir, "preprocess.joblib"))
        self.gmm = load(os.path.join(self.model_dir, "gmm.joblib"))
        self.cluster_to_label = load(os.path.join(self.model_dir, "cluster_to_major_label.joblib"))
        self.sensor_cols = load(os.path.join(self.model_dir, "sensor_cols.joblib"))

        # TRUE-only reliability stats
        self.support_true = load(os.path.join(self.model_dir, "cluster_support_true.joblib"))
        self.purity_true = load(os.path.join(self.model_dir, "cluster_purity_true.joblib"))

        self.K = int(self.gmm.n_components)

        # Precompute reliable clusters gate (TRUE-only)
        self.reliable_cluster = np.array(
            [
                (self.support_true.get(c, 0) >= SETTINGS.MIN_SUPPORT)
                and (self.purity_true.get(c) is not None)
                and (float(self.purity_true[c]) >= SETTINGS.MIN_PURITY)
                for c in range(self.K)
            ],
            dtype=bool,
        )

        self.suggest_thresh = float(getattr(SETTINGS, "SUGGEST_CONF_THRESH", 0.75))

    def _normalize_payload_to_training_cols(self, payload: dict) -> dict:
        """
        Map incoming payload keys (case-insensitive) onto the exact training sensor column names.
        Missing sensors become NaN (then imputed by preprocess).
        """
        lower_map = {k.lower(): k for k in payload.keys()}
        out = {}
        for col in self.sensor_cols:
            src = lower_map.get(col.lower())
            out[col] = payload[src] if src is not None else np.nan
        return out

    def _decision_for_row(self, cluster: int, conf: float) -> tuple[bool, bool, str, int, float | None]:
        pred_class = self.cluster_to_label.get(cluster, None)
        has_mapping = pred_class is not None
        is_reliable_cluster = bool(self.reliable_cluster[cluster])

        cluster_support = int(self.support_true.get(cluster, 0))
        cluster_purity = self.purity_true.get(cluster, None)
        cluster_purity = float(cluster_purity) if cluster_purity is not None else None

        auto_assign = bool(has_mapping and is_reliable_cluster and (conf >= SETTINGS.CONF_THRESH))
        suggest_assign = bool(has_mapping and is_reliable_cluster and (conf >= self.suggest_thresh))

        if not has_mapping:
            reason = "no_cluster_to_class_mapping"
        elif not is_reliable_cluster:
            reason = "cluster_not_reliable_support_or_purity"
        elif auto_assign:
            reason = "auto_assign"
        elif suggest_assign:
            reason = "suggest_assign"
        else:
            reason = "low_confidence"

        return auto_assign, suggest_assign, reason, cluster_support, cluster_purity

    def predict_one(self, payload: dict) -> PredictionResult:
        Xrow = self._normalize_payload_to_training_cols(payload)
        X = pd.DataFrame([Xrow], columns=self.sensor_cols)
        Xp = self.preprocess.transform(X)

        cluster = int(self.gmm.predict(Xp)[0])
        probs = self.gmm.predict_proba(Xp)[0]
        conf = float(np.max(probs))

        pred_class = self.cluster_to_label.get(cluster, None)
        auto_assign, suggest_assign, reason, cluster_support, cluster_purity = self._decision_for_row(cluster, conf)

        return PredictionResult(
            cluster=cluster,
            predicted_class=int(pred_class) if pred_class is not None else None,
            confidence=conf,
            auto_assign=auto_assign,
            suggest_assign=suggest_assign,
            reason=reason,
            cluster_support=cluster_support,
            cluster_purity=cluster_purity,
        )

    def predict_batch(self, rows: list[dict]) -> list[PredictionResult]:
        if len(rows) == 0:
            return []

        Xrows = [self._normalize_payload_to_training_cols(r) for r in rows]
        X = pd.DataFrame(Xrows, columns=self.sensor_cols)
        Xp = self.preprocess.transform(X)

        clusters = self.gmm.predict(Xp)
        probs = self.gmm.predict_proba(Xp)
        confs = probs.max(axis=1)

        out: list[PredictionResult] = []
        for i in range(len(rows)):
            c = int(clusters[i])
            conf = float(confs[i])
            pred_class = self.cluster_to_label.get(c, None)

            auto_assign, suggest_assign, reason, cluster_support, cluster_purity = self._decision_for_row(c, conf)

            out.append(
                PredictionResult(
                    cluster=c,
                    predicted_class=int(pred_class) if pred_class is not None else None,
                    confidence=conf,
                    auto_assign=auto_assign,
                    suggest_assign=suggest_assign,
                    reason=reason,
                    cluster_support=cluster_support,
                    cluster_purity=cluster_purity,
                )
            )
        return out
