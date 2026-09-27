import csv
from itertools import islice
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graph import graph

app = FastAPI(title="LangGraph API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GraphRequest(BaseModel):
    message: str


class SignalResponse(BaseModel):
    sensor: str
    dataset: str
    index: int
    timestamp: str
    frequencies: list[float]
    values: list[float]


DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# 서버가 정상적으로 실행 중인지 확인하는 상태 점검 엔드포인트.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

# CSV에서 요청된 데이터 반환하는 엔드포인트
# 프론트에 그래프 보여줘야 해서.
@app.get("/signals/{sensor}/{dataset}/{index}", response_model=SignalResponse)
def get_signal(
    sensor: Literal["vibration", "current"],
    dataset: Literal["normal", "anomaly"],
    index: int,
) -> SignalResponse:
    if index < 0:
        raise HTTPException(status_code=400, detail="Index must be zero or greater")

    path = DATA_DIR / f"{sensor}_{dataset}.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        row = next(islice(reader, index, index + 1), None)

    if row is None:
        raise HTTPException(status_code=404, detail="Signal sample not found")

    return SignalResponse(
        sensor=sensor,
        dataset=dataset,
        index=index,
        timestamp=row[0],
        frequencies=[float(value) for value in header[1:]],
        values=[float(value) for value in row[1:]],
    )

# 랭그래프 호출
@app.post("/graph")
async def run_graph(request: GraphRequest) -> dict[str, str]:
    return await graph.ainvoke({"message": request.message, "response": ""})
