from fastapi import FastAPI
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/graph")
async def run_graph(request: GraphRequest) -> dict[str, str]:
    return await graph.ainvoke({"message": request.message, "response": ""})
