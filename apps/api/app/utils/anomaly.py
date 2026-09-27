from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np


Sensor = Literal["vibration", "current"]
ARTIFACT_DIR = Path(__file__).resolve().parents[4] / "artifacts"


@lru_cache(maxsize=2)
def load_baseline(sensor: Sensor) -> tuple[np.ndarray, float]:
    """센서별 정상 평균 스펙트럼과 임계값을 한 번만 불러온다."""
    with np.load(ARTIFACT_DIR / f"{sensor}_baseline.npz") as baseline:
        return baseline["mean_spectrum"], float(baseline["threshold"])


def calculate_anomaly(sensor: Sensor, values: list[float]) -> tuple[float, float, bool]:
    """측정값의 이상 점수를 계산하고 임계값과 비교한다."""
    mean_spectrum, threshold = load_baseline(sensor)
    sample = np.asarray(values, dtype=float)

    if sample.shape != mean_spectrum.shape:
        raise ValueError(
            f"{sensor} 측정값은 {mean_spectrum.size}개여야 하지만 {sample.size}개입니다."
        )

    score = float(np.mean(np.abs(sample - mean_spectrum)))
    return score, threshold, score > threshold
