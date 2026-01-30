from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # Data
    csv_path: str = "./data_sensors.csv"
    sensor_prefix: str = "Sensor"
    label_col: str = "Label"

    # NCA
    nca_dim: int = 8
    nca_max_iter: int = 2000
    nca_tol: float = 1e-5

    # GMM selection
    k_min: int = 3
    k_max: int = 10
    cov_type: str = "diag"      # "diag" or "full"
    n_init: int = 10
    reg_covar: float = 1e-6

    # CV
    cv_folds: int = 5
    cv_seed: int = 0
    cv_shuffle: bool = True

    # Business confidence (Confidence Gating)
    conf_thresh: float = 0.90
    min_support: int = 5
    min_purity: float = 0.80

    # Outputs
    artifacts_dir: str = "./app/artifacts"
    model_filename: str = "model_bundle.joblib"
    results_csv_name: str = "nca_gmm_bic_results.csv"
    cv_metrics_name: str = "cv_metrics.csv"
    final_model_selection_name: str = "final_model_selection.csv"
    save_plots: bool = True

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def model_path(self) -> Path:
        return self.artifacts_path / self.model_filename
