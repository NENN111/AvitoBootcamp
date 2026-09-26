from dataclasses import dataclass

SEED = 100


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    train_size: int = 100_000
    validation_size: int = 4000
    calibration_size: int = 2000
    selection_size: int = 2000
    holdout_size: int = 2000
    run_dir: str = 'results/mixed'

# Половина обучающих примеров реальная, половина синтетическая, классы сбалансированы.
MIXED_CONFIG = Config()
