# config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    DATA_PATH: str = "./data/data_sensors.csv"
    LABEL_COL: str = "Label"
    OUTDIR: str = "baseline_outputs"

    # Model selection
    K_MIN: int = 3
    K_MAX: int = 10
    COV_TYPE: str = "diag"
    N_INIT: int = 10
    MAX_ITER: int = 300
    REG_COVAR: float = 1e-6
    INIT_PARAMS: str = "kmeans"
    RANDOM_SEED: int = 42

    # Operational gates
    CONF_THRESH: float = 0.85
    MIN_SUPPORT: int = 4
    MIN_PURITY: float = 0.80

SETTINGS = Settings()
