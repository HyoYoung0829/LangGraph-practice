import operator
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict


class OverallState(TypedDict):
    topic: str
    sections: list[str]
    drafts: Annotated[list[str], operator.add]
    revision_count: int
    final_report: str


class WorkerState(TypedDict):
    topic: str
    section: str


def plan_sections(state: OverallState) -> dict:
    # 전체 State 중 변경할 필드만 반환한다.
    return {
        "sections": ["핵심 개념", "그래프 흐름", "실무 적용"],
        "revision_count": 0,
    }


def route_to_writers(state: OverallState) -> list[Send]:
    # sections 길이에 맞춰 write_section 노드를 동적으로 여러 번 실행한다.
    return [
        Send(
            "write_section",
            {"topic": state["topic"], "section": section},
        )
        for section in state["sections"]
    ]


def write_section(state: WorkerState) -> dict:
    # drafts는 reducer(operator.add)로 각 워커 결과가 누적된다.
    return {
        "drafts": [
            f"- {state['section']}: {state['topic']}을/를 기준으로 간단히 정리했습니다."
        ]
    }


def review_drafts(state: OverallState) -> dict:
    # 검토만 하는 노드라서 State를 바꾸지 않는다.
    return {}


def needs_revision(state: OverallState) -> Literal["revise", "finish"]:
    # 실행 시점의 State를 보고 다음 노드를 선택한다.
    if state["revision_count"] == 0:
        return "revise"
    return "finish"


def revise_drafts(state: OverallState) -> dict:
    # reducer가 있으므로 기존 drafts를 덮어쓰지 않고 보완 문장이 추가된다.
    return {
        "revision_count": state["revision_count"] + 1,
        "drafts": ["- 보완: 각 섹션을 하나의 최종 보고서로 합칠 준비를 마쳤습니다."],
    }


def finish_report(state: OverallState) -> dict:
    # 마지막 노드에서 누적된 drafts를 최종 결과로 변환한다.
    return {"final_report": "\n".join(state["drafts"])}


builder = StateGraph(OverallState)

builder.add_node("plan_sections", plan_sections)
builder.add_node("write_section", write_section)
builder.add_node("review_drafts", review_drafts)
builder.add_node("revise_drafts", revise_drafts)
builder.add_node("finish_report", finish_report)

builder.add_edge(START, "plan_sections")
builder.add_conditional_edges("plan_sections", route_to_writers)
builder.add_edge("write_section", "review_drafts")
builder.add_conditional_edges(
    "review_drafts",
    needs_revision,
    {"revise": "revise_drafts", "finish": "finish_report"},
)
builder.add_edge("revise_drafts", "review_drafts")
builder.add_edge("finish_report", END)

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {
            "topic": "LangGraph 기초",
            "sections": [],
            "drafts": [],
            "revision_count": 0,
            "final_report": "",
        }
    )

    print(result["final_report"])
