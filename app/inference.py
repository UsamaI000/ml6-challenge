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
    reason: str
    cluster_support: int
    cluster_purity: float | None

class InferenceEngine:
    """
    Loads artifacts produced by train.py and provides predict() for single/batch.
    """
    def __init__(self, model_dir: str = SETTINGS.OUTDIR):
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        req = [
            "preprocess.joblib",
            "gmm.joblib",
            "cluster_to_major_label.joblib",
            "cluster_support.joblib",
            "cluster_purity.joblib",
        ]
        for f in req:
            path = os.path.join(self.model_dir, f)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing artifact: {path}. Run `python train.py` first."
                )

        self.preprocess = load(os.path.join(self.model_dir, "preprocess.joblib"))
        self.gmm = load(os.path.join(self.model_dir, "gmm.joblib"))
        self.cluster_to_label = load(os.path.join(self.model_dir, "cluster_to_major_label.joblib"))
        self.support = load(os.path.join(self.model_dir, "cluster_support.joblib"))
        self.purity = load(os.path.join(self.model_dir, "cluster_purity.joblib"))

        self.K = int(self.gmm.n_components)

        # Precompute reliable clusters gate
        self.reliable_cluster = np.array([
            (self.support.get(c, 0) >= SETTINGS.MIN_SUPPORT)
            and (self.purity.get(c) is not None)
            and (float(self.purity[c]) >= SETTINGS.MIN_PURITY)
            for c in range(self.K)
        ], dtype=bool)

    @staticmethod
    def _sensor_cols_from_payload(payload: dict) -> list[str]:
        # Expect keys like Sensor 0..Sensor 19 (case-insensitive accepted)
        # Normalize to match training columns if needed.
        cols = []
        for k in payload.keys():
            if k.lower().startswith("sensor"):
                cols.append(k)
        # Sort "Sensor 0..19" safely:
        def keyf(c):
            parts = c.split()
            try:
                return int(parts[-1])
            except Exception:
                return c
        cols = sorted(cols, key=keyf)
        return cols

    def predict_one(self, payload: dict) -> PredictionResult:
        sensor_cols = self._sensor_cols_from_payload(payload)
        if len(sensor_cols) == 0:
            raise ValueError("Payload must include keys like 'Sensor 0', 'Sensor 1', ...")

        X = pd.DataFrame([{c: payload[c] for c in sensor_cols}])
        Xp = self.preprocess.transform(X)

        cluster = int(self.gmm.predict(Xp)[0])
        probs = self.gmm.predict_proba(Xp)[0]
        conf = float(np.max(probs))

        pred_class = self.cluster_to_label.get(cluster, None)

        # gating
        is_reliable_cluster = bool(self.reliable_cluster[cluster])
        passes_conf = conf >= SETTINGS.CONF_THRESH
        has_mapping = pred_class is not None
        cluster_support = int(self.support.get(cluster, 0))
        cluster_purity = self.purity.get(cluster, None)
        cluster_purity = float(cluster_purity) if cluster_purity is not None else None
        auto_assign = bool(passes_conf and is_reliable_cluster and has_mapping)

        if not has_mapping:
            reason = "no_cluster_to_class_mapping"
        elif not is_reliable_cluster:
            reason = "cluster_not_reliable_support_or_purity"
        elif not passes_conf:
            reason = "low_confidence"
        else:
            reason = "auto_assign"

        return PredictionResult(
            cluster=cluster,
            predicted_class=int(pred_class) if pred_class is not None else None,
            confidence=conf,
            auto_assign=auto_assign,
            reason=reason,
            cluster_support=cluster_support,
            cluster_purity=cluster_purity,
        )

    def predict_batch(self, rows: list[dict]) -> list[PredictionResult]:
        if len(rows) == 0:
            return []

        # Find union of sensor cols across rows; missing handled by imputer in preprocess
        all_cols = set()
        for r in rows:
            for k in r.keys():
                if k.lower().startswith("sensor"):
                    all_cols.add(k)

        if not all_cols:
            raise ValueError("Batch payload must include keys like 'Sensor 0', ...")

        # stable order
        def keyf(c):
            parts = c.split()
            try:
                return int(parts[-1])
            except Exception:
                return c
        sensor_cols = sorted(list(all_cols), key=keyf)

        X = pd.DataFrame([{c: r.get(c, np.nan) for c in sensor_cols} for r in rows])
        Xp = self.preprocess.transform(X)

        clusters = self.gmm.predict(Xp)
        probs = self.gmm.predict_proba(Xp)
        confs = probs.max(axis=1)

        out: list[PredictionResult] = []
        for i in range(len(rows)):
            c = int(clusters[i])
            conf = float(confs[i])
            pred_class = self.cluster_to_label.get(c, None)

            is_reliable_cluster = bool(self.reliable_cluster[c])
            passes_conf = conf >= SETTINGS.CONF_THRESH
            has_mapping = pred_class is not None
            auto_assign = bool(passes_conf and is_reliable_cluster and has_mapping)
            cluster_support = int(self.support.get(c, 0))
            cluster_purity = self.purity.get(c, None)
            cluster_purity = float(cluster_purity) if cluster_purity is not None else None

            if not has_mapping:
                reason = "no_cluster_to_class_mapping"
            elif not is_reliable_cluster:
                reason = "cluster_not_reliable_support_or_purity"
            elif not passes_conf:
                reason = "low_confidence"
            else:
                reason = "auto_assign"

            out.append(PredictionResult(
                cluster=c,
                predicted_class=int(pred_class) if pred_class is not None else None,
                confidence=conf,
                auto_assign=auto_assign,
                reason=reason,
                cluster_support=cluster_support,
                cluster_purity=cluster_purity
            ))
        return out
