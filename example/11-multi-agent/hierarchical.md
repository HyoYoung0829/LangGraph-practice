# LangGraph 멀티에이전트 — Hierarchical(계층적) 패턴 정리 + 3패턴 종합 비교

> 교재: `04-LangGraph-Hierarchial-Agent-Team.ipynb` + 학습 중 나눈 Q&A 정리

---

## 1. 이건 새로운 패턴인가 — 결론부터

**새로운 4번째 패턴이 아니라, Supervisor 패턴을 "그래프를 여러 겹으로 중첩"해서 확장한 것**이에요. 핵심 아이디어 자체(감독자가 작업자에게 위임 → 작업자가 감독자로 복귀)는 그대로고, 달라지는 건 **"감독자 밑에 또 다른 supervisor 그래프가 통째로 들어간다"**는 점.

```
Super-Graph
 ├── Supervisor (총 감독자)
 ├── ResearchTeam ─── 이 자체가 하나의 완결된 Supervisor 그래프
 │     ├── Supervisor (Research 팀 감독자)
 │     ├── Searcher
 │     └── WebScraper
 └── PaperWritingTeam ─── 이것도 하나의 완결된 Supervisor 그래프
       ├── Supervisor (Doc Writing 팀 감독자)
       ├── DocWriter
       ├── NoteTaker
       └── ChartGenerator
```

이전 챕터의 "워커 노드"가, 여기서는 **그 자체로 완전한 서브그래프(Supervisor + 여러 워커)**로 교체된 것. AutoGen 논문 아이디어를 LangGraph로 구현한 사례.

**왜 필요한가**
- 작업자 수가 늘어나면 감독자 하나가 전부 관리하기엔 판단 부담이 커짐 (라우팅 정확도 저하)
- 서로 다른 전문 영역(웹 조사 vs 문서 작성)을 각각 독립된 팀으로 캡슐화하고 싶을 때
- 팀 내부 구현(어떤 도구를 쓰는지, 몇 명인지)을 상위 레벨이 몰라도 되게 **추상화**하고 싶을 때

---

## 2. 팀별 도구와 에이전트 정의

### Research Team

```python
tavily_tool = TavilySearch(max_results=5)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """웹 페이지 스크래핑 도구"""
    loader = WebBaseLoader(web_path=urls, header_template={"User-Agent": "..."})
    docs = loader.load()
    return "\n\n".join([f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>' for doc in docs])
```

### Doc Writing Team — 파일시스템 접근 도구

```python
@tool
def write_document(content: Annotated[str, "..."], file_name: Annotated[str, "..."]) -> str:
    """텍스트 내용을 받아 파일로 저장합니다."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"
```

`create_outline`, `read_document`, `write_document`, `edit_document` — 에이전트가 로컬 파일시스템에 직접 읽고 쓰게 하는 도구들. 교재도 "파일 시스템 접근은 보안상 위험할 수 있다"고 명시적으로 경고함.

**Python REPL 도구**

```python
from langchain_experimental.tools import PythonREPLTool
python_repl_tool = PythonREPLTool()
```

`langchain_experimental`의 도구는 내부적으로 코드를 그대로 `exec`하는 방식이라, 공식 보안 문서(`langchain-ai/langchain/SECURITY.md`)에서도 **"프로덕션에 쓸 땐 반드시 샌드박스 처리하라"**고 명시하고 있음. 지난 챕터에서 얘기했던 `exec()` 보안 이슈와 동일한 맥락 — 학습/데모용으로만 그대로 쓰고, 실제 배포 전엔 격리 계층(도커, 별도 프로세스, 리소스 제한 등)을 반드시 추가해야 함.

---

## 3. `AgentFactory` — 반복되는 에이전트 생성/노드 변환을 재사용 가능하게

```python
class AgentFactory:
    def __init__(self, model_name):
        self.llm = init_chat_model(model_name, temperature=0)

    def create_agent_node(self, agent, name: str):
        def agent_node(state):
            result = agent.invoke(state)
            return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}
        return agent_node
```

- 팀이 여러 개, 워커도 여러 개다 보니 "에이전트 만들고 → 노드로 감싸고 → HumanMessage로 변환"하는 반복 작업을 클래스로 캡슐화
- `HumanMessage(..., name=name)` 변환 — 이전 Network 챕터에서 배운 것과 완전히 동일한 이유. 발신자를 명시(`name`)하고, 다음 노드(팀 내 supervisor)가 "새로운 입력"처럼 자연스럽게 받아들이게 변환

---

## 4. 팀 내부 Supervisor — 라우팅 방식이 이전 챕터와 다름

```python
def create_team_supervisor(model_name, system_prompt, members) -> str:
    options_for_next = ["FINISH"] + members

    class RouteResponse(BaseModel):
        next: Literal[*options_for_next]

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Given the conversation above, who should act next? ... Select one of: {options}"),
    ]).partial(options=str(options_for_next))

    llm = init_chat_model(model_name, temperature=0)
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)
    return supervisor_chain
```

**이전 Supervisor 챕터와의 결정적 차이**: 이전엔 `assign_to_research_agent`처럼 **핸드오프 도구를 호출**시켜서 `Command(goto=...)`로 라우팅했음. 이 교재는 그 대신 **구조화된 출력(structured output)**으로 라우팅함:

- LLM한테 `Literal["Searcher", "WebScraper", "FINISH"]` 타입의 `next` 필드 하나만 뽑아내게 강제 (`with_structured_output`)
- 리턴된 `next` 값을 라우팅 함수(`get_next_node`)가 그대로 읽어서 조건부 엣지로 분기

```python
def get_next_node(x):
    return x["next"]

web_research_graph.add_conditional_edges(
    "Supervisor", get_next_node,
    {"Searcher": "Searcher", "WebScraper": "WebScraper", "FINISH": END},
)
```

**두 방식 비교**

| | 핸드오프 도구 방식 (Supervisor 챕터) | Structured Output 방식 (이 챕터) |
|---|---|---|
| 라우팅 신호 생성 | LLM이 도구(`transfer_to_X`)를 호출 | LLM이 `next` 필드에 값을 채운 구조화 응답 생성 |
| 실제 이동 실행 | 도구 함수 안의 `Command(goto=...)` | 그래프의 `add_conditional_edges` 매핑 |
| Task description 등 부가 정보 전달 | 도구 파라미터로 자유롭게 확장 가능 | `next` 하나만 리턴 — 부가 정보는 별도 처리 필요 |
| 다른 도구와 공존 | 자연스러움 (도구 목록에 handoff 도구를 얹으면 됨) | Supervisor 전용 체인이라 별도로 분리됨 |

**정리**: 둘 다 유효한 구현 방식이고 어느 게 "구식"인 건 아님. 이 교재는 **팀 내부 supervisor처럼 "다음 실행자 하나만 정확히 골라야 하는" 단순 라우팅**에는 구조화 출력이 코드가 더 간결해서 이 방식을 택한 것으로 보임. 반면 "위임하면서 구체적인 작업 지시까지 전달해야 하는" 경우(이전 챕터의 `task_description` 패턴)엔 도구 호출 방식이 자연스럽게 확장하기 좋음.

---

## 5. 팀 그래프 조립 — Supervisor 챕터와 동일한 형태

```python
web_research_graph = StateGraph(ResearchState)
web_research_graph.add_node("Searcher", search_node)
web_research_graph.add_node("WebScraper", web_scraping_node)
web_research_graph.add_node("Supervisor", supervisor_agent)

web_research_graph.add_edge("Searcher", "Supervisor")     # 워커는 항상 팀 supervisor로 복귀
web_research_graph.add_edge("WebScraper", "Supervisor")
web_research_graph.add_conditional_edges("Supervisor", get_next_node, {...})
web_research_graph.set_entry_point("Supervisor")

web_research_app = web_research_graph.compile(checkpointer=InMemorySaver())
```

이 구조 자체는 지금까지 배운 Supervisor 패턴과 완전히 동일함. **차이는 다음 단계 — 이 컴파일된 그래프(`web_research_app`) 자체가 상위 그래프의 노드 하나로 통째로 들어간다는 것.**

### Doc Writing Team의 `preprocess` — 팀별 커스텀 전처리

```python
def preprocess(state):
    written_files = [f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")]
    return {**state, "current_files": "..." + "\n".join([f" - {f}" for f in written_files])}

context_aware_doc_writer_agent = preprocess | doc_writer_agent
```

- `preprocess | doc_writer_agent` — 파이프 연산자로 "전처리 함수 → 에이전트" 체인을 구성. 에이전트를 호출하기 전에 항상 "지금 작업 디렉토리에 어떤 파일이 있는지"를 State에 채워 넣어서, 문서 작성 에이전트가 현재 상황을 인지하고 작업하게 함
- 이건 각 팀의 필요에 따라 자유롭게 끼워넣을 수 있는 부분 — 팀 그래프가 서로 다른 State 스키마(`ResearchState` vs `DocWritingState`)를 갖는 것도 이 유연성 때문

---

## 6. Super-Graph — 팀을 서브그래프로 감싸서 상위 레벨에 연결

```python
def get_last_message(state: State) -> str:
    """마지막 메시지를 추출하여 하위 그래프에 전달"""
    last_message = state["messages"][-1]
    ...

def join_graph(response: dict):
    """하위 그래프의 마지막 메시지를 추출하여 반환"""
    return {"messages": [response["messages"][-1]]}

super_graph = StateGraph(State)
super_graph.add_node("ResearchTeam", get_last_message | web_research_app | join_graph)
super_graph.add_node("PaperWritingTeam", get_last_message | authoring_app | join_graph)
super_graph.add_node("Supervisor", supervisor_node)

super_graph.add_edge("ResearchTeam", "Supervisor")
super_graph.add_edge("PaperWritingTeam", "Supervisor")
super_graph.add_conditional_edges("Supervisor", get_next_node, {...})
super_graph.set_entry_point("Supervisor")
```

**핵심: `get_last_message | web_research_app | join_graph`**

이게 이 챕터의 진짜 핵심 패턴이에요. 파이프(`|`)로 세 개를 이어 붙였는데:

1. **`get_last_message`**: Super-Graph의 `State`(전체 대화 맥락 등)를, 팀 서브그래프가 이해하는 입력 형태로 변환 (전체를 다 넘기지 않고 마지막 메시지만 추출해서 넘김 — 이전에 배운 "명시적 위임" 개념과 같은 이유: 팀 내부 입장에서 필요한 것만 깔끔하게 전달)
2. **`web_research_app`**: Research Team 서브그래프 자체를 통째로 실행 (내부에서 팀 supervisor가 Searcher/WebScraper 사이를 몇 번이고 오가며 작업)
3. **`join_graph`**: 서브그래프 실행이 끝난 결과에서 필요한 부분(마지막 메시지)만 뽑아서 Super-Graph의 State 형식으로 다시 변환

**왜 이 변환이 필요한가 — State 스키마가 레벨마다 다르기 때문**

```python
class ResearchState(TypedDict):
    messages: ...
    team_members: List[str]
    next: str

class State(TypedDict):   # Super-Graph
    messages: ...
    next: str
```

Super-Graph의 `State`와 팀 내부의 `ResearchState`/`DocWritingState`는 **필드 구성이 다름**. 그래서 팀 서브그래프를 상위 그래프에 그냥 노드로 꽂으면 타입이 안 맞음 — `get_last_message`(입구 어댑터)와 `join_graph`(출구 어댑터)가 이 스키마 차이를 중간에서 변환해주는 역할을 함. 계층 구조를 만들 때 거의 항상 필요한 "레벨 간 인터페이스" 패턴.

**총 감독자(Super-Graph의 Supervisor) 프롬프트에 있는 추가 안전장치**

```python
"IMPORTANT: Assign each team at most once. Once a team has reported"
" its results, do NOT assign it again. When all teams have completed"
" their tasks, respond with FINISH immediately."
```

계층이 깊어질수록(팀 내부에서도 여러 번 순환, 그 위에서 팀 간에도 순환 가능) 무한 루프/중복 작업 위험이 커짐. 그래서 "각 팀은 최대 한 번만 배정해라"처럼 이전 챕터보다 더 엄격한 종료 규칙을 프롬프트에 명시함.

---

## 7. 실행 — `invoke_graph` 사용 이유

```python
def run_graph(app, message, recursive_limit=50):
    config = RunnableConfig(recursion_limit=recursive_limit, configurable={"thread_id": random_uuid()})
    inputs = {"messages": [HumanMessage(content=message)]}
    # 계층적 Supervisor 체인은 stream_mode="messages"와 호환되지 않으므로 invoke_graph 사용
    invoke_graph(app, inputs, config)
    return app.get_state(config).values
```

- `with_structured_output`을 쓰는 supervisor 체인은 토큰 단위 스트리밍(`stream_mode="messages"`)과 궁합이 안 맞음 — 구조화 출력은 완성된 객체 하나로 나오는 방식이라 중간 토큰을 흘려보내는 스트리밍과는 안 맞기 때문. 그래서 스트리밍 대신 `invoke_graph`(내부적으로 각 스텝 완료 후 상태를 출력)를 사용
- `recursion_limit`: 계층 구조라 팀 내부 순환 + 팀 간 순환이 겹치면서 스텝 수가 단일 Supervisor보다 훨씬 많이 소요될 수 있음 (Super-Graph 실행 예시에서 `recursive_limit=20`으로 늘려서 실행)

---

## 8. 세 가지 멀티에이전트 패턴 종합 비교

| | **Tool Calling / Supervisor (단일 계층)** | **Network** | **Hierarchical** |
|---|---|---|---|
| 통제 구조 | 중앙 감독자 하나가 모든 워커 관리 | 중앙 통제 없음, 에이전트끼리 순환 | 감독자의 감독자 — 다단계 중앙집중 |
| 라우팅 주체 | Supervisor (도구 호출 또는 구조화 출력) | 각 에이전트 응답의 키워드를 그래프가 감지 | 각 레벨의 Supervisor가 자기 레벨만 책임 |
| 확장 방식 | 워커를 계속 추가 (Supervisor 부담 증가) | 새 에이전트를 순환 경로에 추가 | 팀 단위로 추가 (Supervisor 부담 분산) |
| 관찰성/추적 | 높음 | 낮음 (순환 경로 추적이 상대적으로 어려움) | 매우 높음 (레벨별로 명확히 구분됨) |
| 구현 복잡도 | 낮음 | 중간 | 높음 (State 변환, 서브그래프 어댑터 필요) |
| 무한 루프 위험 | 낮음 (감독자가 매번 판단) | 높음 (순환 구조 자체가 위험 요소) | 중간~높음 (레벨마다 관리 필요, 프롬프트로 강하게 제약해야 함) |

### 8-1. 실무에서 언제 뭘 쓰는가

**Tool Calling / Supervisor (단일 계층)** — 가장 많이 쓰임, 기본 선택지
- 워커 수가 적당함 (대략 2~5개 내외)
- 역할 구분이 명확하고, 위임 흐름이 한 단계로 충분히 설명됨
- 대부분의 프로덕션 에이전트 시스템이 이 형태로 시작함

**Network** — 협업적이고 비선형적인 워크플로우
- 에이전트들이 서로의 결과물을 보고 계속 다듬어야 하는 창작/리뷰 워크플로우 (Writer ↔ Editor ↔ Fact-checker)
- "누가 감독자인지" 딱 정하기 애매한 대등한 관계의 협업
- 대신 관찰성이 떨어지니, 실무에선 종료 조건과 `recursion_limit`을 특히 신경 써서 설계해야 함

**Hierarchical** — 규모가 커질 때만
- 워커 수가 많아져서(6개 이상) 단일 Supervisor의 판단 부담이 실제로 문제가 되는 경우
- 서로 다른 전문 영역이 명확히 "팀" 단위로 나뉘고, 팀 내부 구현을 상위에서 몰라도 되게 캡슐화하고 싶을 때
- 구현/디버깅 비용이 확 늘어나므로(레벨 간 State 어댑터, 다단계 종료 조건), **실제로 필요할 때만** — 워커가 몇 개 안 되는데 미리 계층 구조로 설계하는 건 과한 엔지니어링

### 8-2. 판단 순서 (실무 체크리스트)

1. 워커가 5개 이하이고 역할이 명확함 → **Supervisor(단일 계층)**로 시작
2. 에이전트들이 순서 없이 서로 결과를 주고받으며 다듬어야 함 → **Network** 고려
3. 위 1번으로 시작했는데 감독자 프롬프트가 너무 길어지고 라우팅 실수가 잦아짐 (보통 워커 6개 이상) → **Hierarchical**로 리팩터링
4. 세 패턴은 배타적이지 않음 — 예: 상위는 Hierarchical, 특정 팀 내부는 Network로 조합하는 것도 가능 (이 교재의 Doc Writing Team도 실질적으로 팀 내부는 Supervisor형 순환)

---

## 9. 한화 제조 데이터 해커톤 적용 아이디어

**언제 Hierarchical까지 갈 필요가 있는지 먼저 판단**
- 2주짜리 프로젝트에서 워커가 2~3개(예: 센서 조회, 이상 분석, 리포트 생성) 수준이면 굳이 계층 구조까지 안 가는 게 맞음 — Supervisor 단일 계층으로 충분
- 만약 "설비 진단 팀"(센서 조회 + 이상 탐지 + 근본 원인 분석)과 "리포트 팀"(요약 + 차트 생성 + 문서화)처럼 **두 개의 뚜렷이 구분되는 하위 목적**이 생기고 각 팀 내부에 워커가 여러 개라면, 이 교재의 Research Team / Doc Writing Team 구조를 그대로 벤치마크할 만함

**Doc Writing Team 구조가 특히 유용**
- `write_document`, `create_outline`, `python_repl_tool`(차트) 조합은 "분석 결과를 최종 리포트로 정리"하는 해커톤 산출물 생성 단계에 그대로 적용 가능
- `preprocess` 패턴(현재 작업 디렉토리 파일 목록을 State에 미리 채워주는 것)도 "지금까지 생성된 분석 결과 파일 목록을 다음 에이전트가 인지하게" 하는 데 응용 가능

**보안 체크리스트**
- `python_repl_tool`을 실제 제조 데이터 분석/차트 생성에 쓴다면, `exec()` 기반이라는 걸 인지하고 데모 환경이라도 최소한의 안전장치(허용 모듈 제한, 타임아웃)는 고려
- 파일시스템 접근 도구(`write_document` 등)도 작업 디렉토리를 명확히 격리(`WORKING_DIRECTORY` 같은 고정 경로)해서 의도치 않은 경로 접근을 막아야 함

---

## 10. Q&A로 정리한 핵심 개념

- **이 패턴의 정체**: 완전히 새로운 아키텍처가 아니라, Supervisor 패턴을 "서브그래프로 중첩"해서 규모를 키운 것. 각 팀 자체가 하나의 완결된 Supervisor 그래프이고, 그 팀들을 다시 상위 Supervisor(Super-Graph)가 관리함.
- **구조화 출력(`with_structured_output`) 라우팅 vs 핸드오프 도구 라우팅**: 둘 다 유효한 Supervisor 구현 방식. 구조화 출력은 "다음 실행자 하나만 고르면 되는" 단순 라우팅에 코드가 간결하고, 핸드오프 도구는 위임 시 부가 정보(작업 지시문 등)까지 함께 전달하기 좋음. 구조화 출력 방식은 토큰 스트리밍과 궁합이 안 맞아 `invoke_graph`처럼 완료 후 상태를 받는 방식을 씀.
- **레벨 간 State 변환(`get_last_message` / `join_graph`)**: 계층마다 State 스키마가 다르므로(Super-Graph의 `State` vs 팀의 `ResearchState`), 팀 서브그래프를 상위에 노드로 끼워 넣을 때 입구/출구에서 형식을 맞춰주는 어댑터가 필요함. 파이프(`|`) 연산자로 "변환 → 서브그래프 실행 → 변환"을 자연스럽게 연결.
- **3패턴 선택 기준**: 워커 수가 적고 역할이 명확하면 Supervisor(단일 계층)부터 시작하는 게 기본. 순서 없는 협업/다듬기 워크플로우면 Network. 워커가 많아져서 단일 감독자의 판단 부담이 실제로 문제될 때만 Hierarchical로 확장 — 처음부터 계층 구조로 설계하는 건 대개 과한 엔지니어링.
