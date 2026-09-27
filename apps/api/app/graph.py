from langgraph.graph import END, START, StateGraph

from .state import GraphState


def respond(state: GraphState) -> dict[str, str]:
    return {"response": f"LangGraph received: {state['message']}"}


builder = StateGraph(GraphState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)

graph = builder.compile()
