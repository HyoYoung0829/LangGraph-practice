from typing import Literal

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .utils.anomaly import calculate_anomaly
from .utils.signal import load_signal


load_dotenv()


# 1. 사용할 도구 정의
@tool
def detect_anomaly(
    sensor: Literal["vibration", "current"],
    dataset: Literal["normal", "anomaly"],
    index: int,
) -> dict:
    """지정한 센서 샘플의 이상 여부를 분석한다."""
    signal = load_signal(sensor, dataset, index)

    score, threshold, is_anomaly = calculate_anomaly(
        sensor,
        signal["values"],
    )

    return {
        "sensor": sensor,
        "dataset": dataset,
        "index": index,
        "timestamp": signal["timestamp"],
        "score": score,
        "threshold": threshold,
        "is_anomaly": is_anomaly,
    }


# 2. 모델 생성
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)


# 3. 모델에 도구 알려주기
tools = [detect_anomaly]
model_with_tools = model.bind_tools(tools)


# 4. 모델 노드 함수
def call_model(state: MessagesState) -> dict:
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# 5. 도구 실행 노드 생성
tool_node = ToolNode(tools)


# 6. 그래프 생성
builder = StateGraph(MessagesState)


# 7. 그래프에 노드 등록
builder.add_node("model", call_model)
builder.add_node("tools", tool_node)


# 8. 노드 연결
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")


# 9. 실행 가능한 그래프로 완성
graph = builder.compile()