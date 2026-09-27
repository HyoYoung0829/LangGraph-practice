import csv
from itertools import islice
from pathlib import Path
from typing import Literal


DATA_DIR = Path(__file__).resolve().parents[4] / "data"


def load_signal(
    sensor: Literal["vibration", "current"],
    dataset: Literal["normal", "anomaly"],
    index: int,
) -> dict:
    """CSV에서 센서 데이터 한 건을 읽어 반환한다."""
    if index < 0:
        raise ValueError("index는 0 이상이어야 합니다.")

    path = DATA_DIR / f"{sensor}_{dataset}.csv"

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        row = next(islice(reader, index, index + 1), None)

    if row is None:
        raise ValueError(f"{path.name}에 {index}번 샘플이 없습니다.")

    return {
        "sensor": sensor,
        "dataset": dataset,
        "index": index,
        "timestamp": row[0],
        "frequencies": [float(value) for value in header[1:]],
        "values": [float(value) for value in row[1:]],
    }
