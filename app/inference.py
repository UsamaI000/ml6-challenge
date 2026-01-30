from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Prediction:
    cluster: int
    predicted_label: Optional[int]
    gmm_confidence: float
    cluster_purity: float
    cluster_support: int
    business_confidence: float
    is_high_confidence: bool


class ModelRunner:
    def __init__(self, bundle: Dict[str, Any]):
        self.sensor_cols: List[str] = bundle["sensor_cols"]
        self.scaler = bundle["scaler"]
        self.nca = bundle["nca"]
        self.gmm = bundle["gmm"]
        self.best_k: int = int(bundle["best_k"])
        self.cluster_to_label: Dict[int, Optional[int]] = bundle["cluster_to_label"]
        self.cluster_purity_map: Dict[int, float] = bundle["cluster_purity"]
        self.cluster_support_map: Dict[int, int] = bundle["cluster_support"]
        self.min_support: int = int(bundle["min_support"])
        self.min_purity: float = float(bundle["min_purity"])
        self.conf_thresh: float = float(bundle["conf_thresh"])

        # Arrays for fast indexing
        self.purity_arr = np.array([self.cluster_purity_map.get(c, 0.0) for c in range(self.best_k)], dtype=float)
        self.support_arr = np.array([self.cluster_support_map.get(c, 0) for c in range(self.best_k)], dtype=int)

    @staticmethod
    def load(model_path: str) -> "ModelRunner":
        bundle = joblib.load(model_path)
        return ModelRunner(bundle)

    def _to_dataframe(self, x: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            df = x.copy()
        elif isinstance(x, dict):
            df = pd.DataFrame([x])
        else:
            df = pd.DataFrame(x)

        missing = [c for c in self.sensor_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing sensor columns: {missing}")

        return df[self.sensor_cols]

    def predict(self, x: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame]) -> List[Prediction]:
        X_df = self._to_dataframe(x)
        X = X_df.values.astype("float64")

        X_scaled = self.scaler.transform(X)
        Z = self.nca.transform(X_scaled)

        cluster = self.gmm.predict(Z)
        post = self.gmm.predict_proba(Z)
        gmm_conf = post.max(axis=1)

        purity = self.purity_arr[cluster]
        support = self.support_arr[cluster]

        # mapped label (may exist even when gate blocks it)
        mapped = [self.cluster_to_label.get(int(c), None) for c in cluster]

        business_conf = gmm_conf * purity

        # hard gates: if not validated, force business_conf=0 and predicted_label=None
        bad = (support < self.min_support) | (purity < self.min_purity)
        business_conf = business_conf.copy()
        business_conf[bad] = 0.0

        pred_label = []
        for i in range(len(mapped)):
            if bad[i]:
                pred_label.append(None)
            else:
                pred_label.append(mapped[i])

        is_hc = business_conf >= self.conf_thresh

        out: List[Prediction] = []
        for i in range(len(cluster)):
            out.append(
                Prediction(
                    cluster=int(cluster[i]),
                    predicted_label=pred_label[i],
                    gmm_confidence=float(gmm_conf[i]),
                    cluster_purity=float(purity[i]),
                    cluster_support=int(support[i]),
                    business_confidence=float(business_conf[i]),
                    is_high_confidence=bool(is_hc[i]),
                )
            )
        return out
