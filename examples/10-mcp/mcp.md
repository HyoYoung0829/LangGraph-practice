# LangGraph × MCP 정리

> 교재: `01-LangGraph-MCP-Tutorial.ipynb` + 학습 중 나눈 Q&A 정리

---

## 1. MCP가 뭔가

**MCP(Model Context Protocol)** = "애플리케이션이 LLM에 도구와 컨텍스트를 제공하는 방식"을 표준화한 오픈 프로토콜. 예전엔 도구마다 각자 다른 연동 코드를 짜야 했는데, MCP는 하나의 표준 인터페이스로 통일함.

- **표준화된 도구 인터페이스**: 서버마다 다르게 안 짜도 됨
- **여러 전송 방식**: stdio, Streamable HTTP
- **동적 도구 검색**: 런타임에 서버가 제공하는 도구 목록을 자동으로 가져옴
- **다중 서버 연결**: 여러 서버를 한 번에 묶어서 씀

---

## 2. 전송 방식 — STDIO vs Streamable HTTP

핵심 차이는 **"서버가 어디서 도는가"**.

| | STDIO | Streamable HTTP |
|---|---|---|
| 통신 위치 | 같은 컴퓨터 안 (프로세스 간 통신) | 네트워크 너머 |
| 서버 실행 주체 | 클라이언트가 자식 프로세스로 직접 실행 | 이미 떠있는 서버에 접속만 |
| 다중 클라이언트 | 불가 (1:1) | 가능 (1서버:N클라이언트) |
| 인증 | 없음 (로컬 신뢰) | 필요 (헤더/OAuth) |
| 용도 | 로컬 개발, IDE 도구 연결 | 팀/여러 사용자에게 호스팅 |

```python
from fastmcp import FastMCP

mcp = FastMCP("Weather", instructions="날씨 정보를 제공하는 어시스턴트입니다.")

@mcp.tool
async def get_weather(location: str) -> str:
    """지정된 위치의 현재 날씨 정보를 가져옵니다."""
    return f"It's always Sunny in {location}"

if __name__ == "__main__":
    mcp.run(transport="stdio")  # HTTP: mcp.run(transport="http", port=8002)
```

**참고**: 예전 원격 방식이던 HTTP+SSE(POST/SSE 엔드포인트 두 개)는 deprecated. 지금은 **Streamable HTTP**(엔드포인트 하나로 요청·응답·세션 스트림 다 처리)가 표준.

---

## 3. MCPAdapter — target 하나로 전송 방식 자동 추론

`MCPAdapter`는 MCP 서버에 연결해서 도구를 LangChain 도구로 바꿔주는 **비동기 컨텍스트 매니저**. `target`의 형태만 보고 전송 방식을 알아서 결정함.

| target | 전송 방식 |
|---|---|
| `Path("server.py")` | stdio — 서브프로세스로 실행 |
| `"http://.../mcp"` (URL 문자열) | Streamable HTTP |
| `FastMCP` 인스턴스 | 인메모리 (테스트용) |
| `{"mcpServers": {...}}` (MCPConfig) | 여러 서버를 하나의 어댑터로 |

```python
async def load_mcp_tools(target):
    async with MCPAdapter(target) as adapter:
        tools = await adapter.list_tools()
    return tools  # 컨텍스트 종료 후에도 도구는 계속 호출 가능 (호출마다 세션 열고 닫음)
```

**다중 서버 구성**

```python
server_config = {
    "mcpServers": {
        "weather": {"command": "uv", "args": ["run", "python", "server/mcp_server_local.py"]},  # stdio
        "current_time": {"url": "http://127.0.0.1:8002/mcp"},  # Streamable HTTP
    }
}
```

- `command`/`args` → stdio, `url` → Streamable HTTP로 **키만 보고 자동 판별**
- 도구 이름 충돌 방지를 위해 서버명이 접두사로 자동으로 붙음 (`weather_get_weather`, `current_time_get_current_time`)
- stdio와 HTTP를 섞어도 에이전트 입장에선 구분 없이 똑같은 도구 리스트로 취급됨

---

## 4. 에이전트에 MCP 도구 연결하기

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    llm,
    tools,                       # MCP 도구든 일반 @tool 함수든 섞어도 됨
    checkpointer=InMemorySaver(),
)
```

- 이전 챕터에서 배운 `checkpointer` + `thread_id`는 도구 출처(로컬 함수 vs MCP 서버, stdio vs HTTP)와 완전히 독립적으로 작동함
- 같은 `thread_id`로 여러 질문 → 대화 맥락 유지 (MCP 도구를 썼든 안 썼든 동일)

**RAG도 MCP 도구 중 하나일 뿐**

```python
rag_server = Path("server/mcp_server_rag.py")
rag_tools = await load_mcp_tools(rag_server)
rag_agent = create_agent(llm, rag_tools, checkpointer=InMemorySaver())
```

PDF 검색 같은 RAG 기능도 그냥 "도구 하나"로 MCP로 노출해서 쓰는 것. 에이전트가 필요할 때 알아서 이 검색 도구를 호출함 — Agentic RAG의 실제 구현 형태 중 하나.

---

## 5. ToolNode — create_agent와 달리 직접 조립하는 방식

`ToolNode`는 도구가 아니라 **"LLM이 요청한 도구 호출을 실제로 실행해주는 그래프 노드"**.

```
agent 노드: LLM 호출 → "이 도구를 이런 인자로 불러줘" 요청 생성
    ↓
tools 노드 (ToolNode): 그 요청 실행 → 결과 반환
    ↓
agent 노드: 결과 보고 다시 판단
```

| | `create_agent` (prebuilt) | `StateGraph` + `ToolNode` (직접 조립) |
|---|---|---|
| agent/tools 노드, 라우팅 | 프레임워크가 이미 구현 | 직접 작성 |
| State 구조 | 정해진 형태 | 커스텀 필드 자유롭게 추가 (`context` 등) |
| 중간 로직 끼워넣기 | 어려움 (블랙박스) | 검증/로깅/승인 노드 등 자유롭게 삽입 가능 |

**MCP 자체는 똑같이 씀 — 차이는 그 도구를 감싸는 에이전트 루프를 얼마나 세밀하게 제어하냐**

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: Dict[str, Any]   # create_agent엔 없는 커스텀 필드

tools = await load_mcp_tools(target)
tools.append(TavilySearch(max_results=2))   # MCP 도구 + 일반 도구 섞기
llm_with_tools = llm.bind_tools(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
```

---

## 6. 외부(3rd Party) MCP 서버 — 직접 만든 서버와의 결정적 차이

| | 직접 만든 서버 (Part 2~5) | 3rd Party 서버 (Part 6, 예: Context7) |
|---|---|---|
| 서버 코드 작성 | 내가 함 | 안 함 — 이미 배포된 걸 그대로 씀 |
| 도구 동작 신뢰 | 내 코드니까 검증됨 | 남을 믿어야 함 (블랙박스) |
| 인증 | 필요 없거나 직접 설정 | 대부분 API 키/토큰 필요 |
| 커스터마이징 | 자유 | 불가능 |

**Context7 예시** — 최신 라이브러리 문서를 실시간 검색해서 프롬프트에 넣어주는 MCP 서버 (RAG를 MCP로 표준화해서 배포한 형태)

```python
"context7": {
    "command": "npx",
    "args": ["-y", "@upstash/context7-mcp@latest"],  # stdio, Node.js 필요
}
```

붙이는 코드 자체(`load_mcp_tools`)는 자체 서버 때와 문법상 동일함. 교재가 강조하려는 건 **코드가 아니라 "누가 만든 걸 쓰느냐"의 개념적 전환** — 내가 통제 가능한 자체 서버에서, 남이 운영하는 외부 생태계 서버로 넘어가는 지점.

Smithery AI 같은 MCP 서버 레지스트리에서 다른 3rd party 서버도 같은 방식으로 검색해서 붙일 수 있음.

---

## 7. 교재 코드에서 실무 기준으로 챙길 점

이 노트북은 `langchain.mcp`(신규 네임스페이스, `langchain[mcp]>=1.4.0`, 현재 **beta**), `MCPAdapter`, FastMCP 4.x, `create_agent` 등 **이미 최신 API 기준**으로 짜여 있음. 옛날 문법을 새로 고쳐야 할 부분은 없고, 대신 실무 적용 시 유의할 점만 짚음.

### 7-1. beta 네임스페이스라는 점 인지

```python
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)
```

`langchain.mcp`는 아직 beta라 API가 바뀔 수 있음. 프로덕션에 박아넣기 전에 버전 고정(`langchain[mcp]==1.4.0` 식으로 핀)해두는 게 안전함. 경고를 무시하는 건 학습 편의를 위한 것이지, "완전히 안정된 API"라는 뜻은 아님.

### 7-2. `MultiServerMCPClient` → `MCPAdapter`로 이미 통합됨

예전엔 `langchain-mcp-adapters`라는 별도 패키지의 `MultiServerMCPClient`를 썼는데, 지금은 LangChain 본체 안의 단일 `MCPAdapter` 클래스로 흡수됨. 다른 자료/블로그 보다가 `MultiServerMCPClient` 나오면 옛날 글임.

### 7-3. `create_react_agent` → `create_agent`로 이미 deprecated

```python
from langchain.agents import create_agent   # 권장
# from langgraph.prebuilt import create_react_agent  # deprecated
```

이전 챕터(메모리)에서 쓰던 `create_react_agent`도 이 노트북에서는 `create_agent`로 바뀌어 있음. 새 자료를 찾을 때 `create_react_agent` 기준 설명이면 구버전 기준일 확률이 높음.

### 7-4. 연결 생명주기 — `async with` 블록 밖에서도 도구가 살아있는 이유

```python
async with MCPAdapter(target) as adapter:
    tools = await adapter.list_tools()
# 여기서 벗어나도 tools는 계속 호출 가능
```

`list_tools()`로 받은 도구 객체가 클라이언트 정보를 내부에 들고 있어서, `with` 블록이 끝난 뒤에도 에이전트가 그 도구를 호출할 수 있음. 대신 **도구를 호출할 때마다 세션을 새로 열고 닫음** — 즉 "연결을 계속 유지하는" 게 아니라 "필요할 때마다 짧게 재연결"하는 구조. 호출 빈도가 아주 높은 프로덕션이면 이 재연결 비용이 누적될 수 있어서, 그럴 땐 `ClientGroup`(서버별 독립 연결을 계속 유지)을 쓰는 게 나을 수 있음 — 규모 커지면 검토 대상.

### 7-5. 다중 서버 묶을 때 프로토콜 버전 하향 주의

`MCPConfig`로 여러 서버를 하나의 어댑터로 묶으면, 전체 fleet이 **가장 낮은 프로토콜 버전에 맞춰짐**. 레거시 서버 하나만 섞여 있어도 최신 기능(예: elicitation)을 쓰는 다른 서버까지 구버전 방식으로 동작할 수 있음. 서로 다른 세대의 서버를 묶어야 하면 `MCPConfig` 대신 `ClientGroup`으로 서버별 독립 연결을 쓰는 게 맞음.

### 7-6. 에러 처리

- MCP 도구 자체의 실패(`isError=True`) → `status="error"`인 `ToolMessage`로 모델에 전달됨 (모델이 인지하고 재시도/다른 방법 시도 가능)
- 전송 계층 자체의 실패(연결 끊김 등) → 예외로 발생 (개발자가 try/except로 직접 처리해야 함)

이 둘을 구분해서 처리해야 함 — "도구가 실패한 것"과 "연결 자체가 끊긴 것"은 복구 전략이 다름.

---

## 8. 한화 제조 데이터 해커톤 적용 아이디어

**MCP로 표준화할 만한 것들**
- 설비 센서 데이터 조회, 이상 탐지 로그 검색, 생산 이력 DB 쿼리 같은 걸 각각 MCP 서버(stdio)로 분리해두면, 에이전트 쪽 코드는 안 건드리고 서버만 교체/확장 가능
- 로컬 개발 중엔 stdio로 빠르게 테스트하고, 심사/데모 때 여러 팀원이 접근해야 하면 Streamable HTTP로 전환

**멀티 서버 구조 예시**
```python
server_config = {
    "mcpServers": {
        "sensor_query": {"command": "uv", "args": ["run", "python", "server/sensor.py"]},
        "anomaly_log": {"command": "uv", "args": ["run", "python", "server/anomaly.py"]},
        "production_db": {"url": "http://internal-server:8000/mcp"},
    }
}
```

**ToolNode 직접 조립이 유용한 지점**
- 심사 기준상 "왜 이 판단을 했는지"를 보여줘야 한다면, `create_agent`(블랙박스)보다 `StateGraph` + `ToolNode`로 직접 조립해서 각 단계(센서 조회 → 이상 판단 → 리포트 생성)를 노드로 명시적으로 나누는 게 설명하기 쉬움
- State에 `context` 같은 커스텀 필드를 둬서 "지금까지 조회한 설비 목록", "확정된 이상 여부" 같은 걸 단계별로 누적시키는 것도 가능

**3rd Party MCP 활용**
- 코드 생성 단계에서 Context7 같은 걸 붙여두면, LangGraph/langchain 최신 문법으로 코드를 짜는 데 도움됨 (2주짜리 해커톤에서 버전 꼬임으로 시간 버리는 걸 줄일 수 있음)

**짧은 기간 고려한 순서**
1. 자체 stdio 서버 1~2개(센서 조회 등)로 `create_agent` 기반 단순 에이전트 먼저 동작시키기
2. 시간 남으면 `ToolNode` 직접 조립으로 전환해서 단계별 제어/설명 가능하게 확장
3. 필요하면 Streamable HTTP로 전환해서 팀원 간 공유 가능한 형태로

---

## 9. Q&A로 정리한 핵심 개념

- **MCP vs LangGraph**: MCP는 "완성된 도구를 설정으로 연결하는 프로토콜", LangGraph는 "그 도구들을 어떤 순서/조건으로 쓸지 직접 짜는 프레임워크". 대립 관계가 아니라 LangGraph 노드 하나가 MCP 도구를 호출하는 식으로 같이 씀.
- **MCP 서버를 내가 띄워야 하나**: stdio는 클라이언트가 설정대로 알아서 자식 프로세스로 띄워줌(직접 터미널에서 실행 안 해도 됨). 원격 서버는 이미 어딘가에 떠있는 걸 URL로 접속만 하는 것.
- **STDIO vs Streamable HTTP**: 로컬 프로세스 통신이냐 네트워크 통신이냐가 핵심 차이. 혼자 로컬에서 쓸 거면 STDIO, 여러 사용자에게 원격으로 서비스할 거면 Streamable HTTP.
- **에이전트 개발 = RAG 개발이 아님**: RAG는 에이전트가 쓸 수 있는 도구 중 하나(검색 기법)일 뿐. RAG를 MCP로 노출할 수도 있고, 그냥 로컬 함수 도구로 구현할 수도 있음 — MCP는 필수가 아니라 "도구를 어떻게 배포할지"의 선택지 중 하나.
- **ReAct vs Agentic RAG**: ReAct는 도구 종류 불문하고 "생각→행동→관찰"을 반복하는 범용 에이전트 루프 패턴. Agentic RAG는 "검색을 얼마나 똑똑하게 할지"에 초점 맞춘 아키텍처 — ReAct로 구현할 수도, 검색 전용 커스텀 그래프로 구현할 수도 있음.
- **ToolNode vs 도구**: 도구는 "무슨 일을 하는지" 정의된 낱개 함수, `ToolNode`는 "LLM이 요청한 도구 호출을 실제로 실행해주는" 그래프 노드. `create_agent`는 이 구조를 이미 조립해둔 완제품, `ToolNode` 직접 조립은 부품 단위로 커스텀하는 것.
