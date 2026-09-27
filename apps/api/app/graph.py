import csv
from langgraph.graph import END, START, StateGraph
from langchain_core.tools import tool
from typing import Literal
from itertools import islice
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from .state import GraphState

load_dotenv()

DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def respond(state: GraphState) -> dict[str, str]:
    return {"response": f"LangGraph received: {state['message']}"}

# csv 읽는 유틸 함수
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


# @tool
# def read_signal(
#     sensor: Literal["vibration", "current"],
#     dataset: Literal["normal", "anomaly"],
#     index: int,
# ) -> dict:
#     """센서 종류, 데이터셋 종류, 샘플 번호로 CSV에서 측정 데이터 한 건을 조회한다."""
#     return load_signal(sensor, dataset, index)


@tool
def detect_anomaly(
    sensor: Literal["vibration", "current"],
    dataset: Literal["normal", "anomaly"],
    index: int,
) -> any:
    """지정한 센서 샘플의 이상 여부를 분석한다."""

    signal = load_signal(sensor, dataset, index)

    values = signal["values"]

    # 이상 현상 감지 로직

    return {
        "sensor": sensor,
        "index": index,
        "timestamp": signal["timestamp"],
    }



model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

tools = [detect_anomaly]
model_with_tools = model.bind_tools(tools)

builder = StateGraph(GraphState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile()


# if __name__ == "__main__":
#     result = model_with_tools.invoke(
#         "current normal 데이터의 8201번 샘플을 읽어줘."
#     )

#     print(result.tool_calls)
