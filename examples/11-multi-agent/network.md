# LangGraph 멀티에이전트 — Network(협업) 패턴 정리

> 교재: `02-LangGraph-Multi-Agent-Collaboration.ipynb` + 학습 중 나눈 Q&A 정리

---

## 1. Supervisor와 뭐가 다른가

이전 챕터(Supervisor)는 감독자가 모든 라우팅을 결정하고, 워커는 끝나면 무조건 감독자한테 복귀하는 **중앙집중형** 구조였음. 이번 챕터는 그 대비되는 축인 **Network(분산형) 패턴** — 중앙 감독자 없이 에이전트들이 서로 순환하며 협업함.

> AutoGen 논문에서 영감을 받은 구조. "분할 정복(divide-and-conquer)" — 각 에이전트가 자기 전문 분야에만 집중하고, 필요하면 다른 전문 에이전트한테 넘김.

| | Supervisor | Network |
|---|---|---|
| 통제 주체 | 중앙의 감독자 노드 | 없음 — 에이전트끼리 순환 |
| 라우팅 방식 | `Command(goto=...)` (에이전트가 직접 목적지 지정) | 조건부 엣지 + 라우터 함수 (그래프가 사전에 정의한 경로만 순환) |
| 종료 판단 | supervisor가 판단해서 END 결정 | 각 에이전트 응답에 담긴 키워드(`FINAL ANSWER`)를 라우터가 감지 |
| 발신자 구분 | 그래프 구조(항상 supervisor를 거침)로 암시적으로 구분됨 | 메시지의 `name` 필드로 명시적으로 구분해야 함 |

이 교재의 구현은 "완전 자유 연결"이 아니라 **researcher ↔ chart_generator 2개 노드가 서로를 향해 조건부 엣지로 순환**하는 단순한 형태의 Network. 개념적으로는 에이전트가 몇 개든 서로 순환 가능한 구조로 확장 가능.

---

## 2. 도구 정의

```python
tavily_tool = TavilySearch(max_results=5)

def run_python_code(code: str) -> str:
    """Python 코드를 실행하고 stdout 출력을 반환합니다."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        exec(code, {})
    except Exception as e:
        sys.stdout = old_stdout
        return f"Error: {repr(e)}"
    sys.stdout = old_stdout
    return buffer.getvalue()

@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."]):
    """Python 코드를 실행합니다."""
    result = run_python_code(code)
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
```

- `Annotated[str, "..."]`로 파라미터 설명을 달아둔 것 — 이 설명이 LLM한테 그대로 노출되는 도구 스키마 정보가 됨
- 도구 반환값 끝에 **"FINAL ANSWER로 답하라"는 지시를 박아둔 게 포인트** — 도구 실행 결과 자체가 "이제 끝났으면 이 키워드를 써라"고 LLM을 유도하는 장치. 라우터가 감지할 신호를 도구 레벨에서도 상기시켜주는 것

**실무 주의점 — `exec()`는 프로덕션에 그대로 쓰면 안 됨**

```python
exec(code, {})
```

이건 LLM이 생성한 코드를 **그대로 실행**하는 것. 학습/데모용으로는 괜찮지만, 실제 서비스에 넣으려면:
- 샌드박스 격리(도커 컨테이너, 별도 프로세스, `subprocess` + 리소스 제한 등) 없이 이렇게 직접 `exec`하면 임의 코드 실행 위험(파일 시스템 접근, 네트워크 호출, 무한 루프 등)이 그대로 열려있음
- LangChain 공식 `PythonREPLTool`도 문서에 "신뢰할 수 없는 코드 실행 위험"을 명시하고 있음 — 실제 배포 전엔 격리 계층을 꼭 추가해야 함

---

## 3. 에이전트 생성 — 공통 시스템 프롬프트 패턴

```python
def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )
```

모든 에이전트가 공유하는 규칙(협업 방식, 종료 신호)을 헬퍼 함수로 빼두고, 각자 다른 `suffix`(역할별 지시)만 덧붙이는 구조. Supervisor 패턴에서 "각 워커는 자기 역할만 하고 다른 영역은 침범하지 않는다"를 매번 길게 썼던 것과 비슷한데, 여기선 공통 부분을 재사용 가능하게 함수화한 게 차이점.

```python
research_agent = create_agent(
    model,
    tools=[tavily_tool],
    system_prompt=make_system_prompt("You can only do research. You are working with a chart generator colleague."),
)

chart_agent = create_agent(
    model,
    [python_repl_tool],
    system_prompt=make_system_prompt(chart_generator_system_prompt),
    name="chart_generator",
)
```

---

## 4. 노드 함수 — 결과를 HumanMessage로 변환하는 이유

```python
def research_node(state: MessagesState) -> MessagesState:
    result = research_agent.invoke(state)
    last_message = HumanMessage(content=result["messages"][-1].content, name="researcher")
    return {"messages": [last_message]}
```

**왜 `AIMessage`가 아니라 `HumanMessage`로 바꾸나**: 다음 에이전트(chart_generator) 입장에서, 이걸 "동료가 방금 나한테 보낸 요청/정보"처럼 자연스럽게 받아들이게 하려는 의도. `AIMessage`로 그대로 두면 다음 에이전트가 "이건 내가(AI) 한 말인가?"로 헷갈릴 수 있음 — 역할 혼동을 막기 위한 변환.

**`name` 필드가 왜 필수인가 (Network 패턴 고유의 이유)**: Supervisor 패턴은 감독자가 항상 중간에 껴 있어서 "누가 이 결과를 냈는지" 그래프 구조 자체가 암시해줌. Network는 중앙 감독자가 없이 같은 `messages` 리스트를 여러 에이전트가 공유하면서 순환하기 때문에, `role`만으로는 어느 에이전트의 발언인지 구분이 안 됨. `name="researcher"`, `name="chart_generator"`처럼 명시적으로 박아둬야:
- 다음 에이전트가 "이 정보는 누가 만든 건지" 구분해서 읽을 수 있음
- 같은 작업을 중복 수행하는 낭비를 막을 수 있음
- 로그/디버깅 시 발신자 추적이 가능함

---

## 5. 라우터 함수 — LLM 없이 "이미 내려진 판단"을 읽기만 함

```python
def router(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    content = last_message.content

    if isinstance(content, list):
        content = " ".join(block.get("text", "") for block in content if isinstance(block, dict))

    if "FINAL ANSWER" in content:
        return END
    return "continue"
```

**이 함수 자체엔 LLM이 없음** — 판단은 이미 그 전 단계(에이전트 노드의 LLM 호출)에서 끝났고, 라우터는 그 결과 텍스트에 박혀있는 약속된 키워드(`FINAL ANSWER`)를 문자열 매칭으로 확인만 함.

```
1. agent 노드: LLM 호출 → "작업 끝났다" 판단되면 응답 앞에 FINAL ANSWER 붙여서 생성
   (이 판단 자체는 system_prompt의 "prefix your response with FINAL ANSWER" 지시에 따라 LLM이 매 턴 스스로 함)
2. router: 그 텍스트에 "FINAL ANSWER"가 있는지 그냥 확인
3. 있으면 END, 없으면 "continue" 라벨 반환
```

`content`가 리스트(content block) 형태일 수도 있어서 처리해준 부분도 있음 — 모델에 따라 응답이 단순 문자열이 아니라 `[{"type": "text", "text": "..."}]` 같은 블록 리스트로 올 수 있어서, 그 경우 텍스트만 추출해서 합침.

**라벨과 실제 노드는 매핑 테이블에서 연결됨**

```python
workflow.add_conditional_edges(
    "researcher", router, {"continue": "chart_generator", END: END},
)
workflow.add_conditional_edges(
    "chart_generator", router, {"continue": "researcher", END: END},
)
```

`router`가 리턴하는 `"continue"`는 노드 이름이 아니라 그냥 라벨. 어디로 갈지는 `add_conditional_edges`의 세 번째 인자(매핑 딕셔너리)가 미리 정해둠 — "누구한테 갈 수 있는지"는 그래프 엣지로 사전에 고정돼 있고, LLM은 그 틀 안에서 "끝났다/계속한다"라는 신호만 준다는 게 이 Network 구현의 특징. (참고로 Supervisor의 `Command(goto=agent_name)` 방식은 LLM이 목적지 자체를 직접 고르는 것과 대비됨)

---

## 6. 그래프 조립

```python
workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)

workflow.add_conditional_edges("researcher", router, {"continue": "chart_generator", END: END})
workflow.add_conditional_edges("chart_generator", router, {"continue": "researcher", END: END})

workflow.add_edge(START, "researcher")
app = workflow.compile(checkpointer=InMemorySaver())
```

- 두 노드가 **서로를 향해** 조건부 엣지를 걺 — `researcher`가 끝나면 다음은 `chart_generator`, 거기서 안 끝나면 다시 `researcher`. 이게 Supervisor의 "워커는 항상 supervisor로 복귀"와 다른, Network 특유의 순환 구조
- `visualize_graph(app, xray=True)` — `xray=True`는 내부 서브그래프 구조까지 펼쳐서 시각화해줌 (`subgraphs=True` 스트리밍 옵션과 유사한 개념을 그래프 다이어그램 쪽에 적용한 것)
- `recursion_limit=10`: 순환 구조라 무한 루프 위험이 있으므로, 최대 몇 번까지 순회를 허용할지 제한. Network 패턴에서 특히 중요한 안전장치 — 라우터의 종료 판단이 실패하면 계속 왔다갔다 할 수 있기 때문

---

## 7. 실행

```python
config = RunnableConfig(recursion_limit=10, configurable={"thread_id": "thread-1"})
inputs = {"messages": [HumanMessage(content="2010년~2025년까지의 대한민국의 1인당 GDP 추이를 그래프로 시각화 해주세요.")]}
stream_graph(app, inputs=inputs, config=config)
```

흐름 예상: researcher가 GDP 데이터 검색 → chart_generator한테 순환 → chart_generator가 python_repl_tool로 차트 코드 실행 → 데이터가 부족하면 다시 researcher로 → 충분해지면 어느 한쪽이 `FINAL ANSWER`로 응답하고 종료.

---

## 8. 한화 제조 데이터 해커톤 적용 아이디어

**Network가 어울리는 경우 vs 아닌 경우**

- Supervisor 패턴 정리 문서에서 다뤘듯, 심사에서 "판단 근거"를 설명해야 하는 경우엔 Network보다 Supervisor가 유리함 (중앙 통제 지점이 있어야 추적이 쉬움)
- 다만 "센서 데이터 조사 ↔ 이상 여부 분석"처럼 **두 역할이 서로의 결과를 보고 되돌아가며 다듬어야 하는** 성격의 작업(예: 분석 agent가 "이 구간 데이터 더 필요하다"고 조사 agent한테 되묻는 구조)이라면 Network의 순환 구조가 자연스러움

**차트 생성 에이전트 패턴 자체가 바로 활용 가능**

이 교재의 `researcher ↔ chart_generator` 구조는 "데이터 조사 → 시각화"라는 조합이라, 제조 데이터 해커톤에서 그대로 응용 가능:
- researcher 역할 → 센서/생산 로그 조회 전담 에이전트
- chart_generator 역할 → 조회된 수치를 그래프/리포트로 시각화하는 에이전트
- `FINAL ANSWER` 신호 대신, 팀 상황에 맞는 명확한 종료 조건 문구를 시스템 프롬프트에 정의

**`exec()` 보안 이슈는 해커톤에서도 챙길 것**

만약 LLM이 생성한 분석 코드를 실행하는 도구를 만든다면, 데모 환경이라도 `exec()`를 무방비로 쓰지 말고 최소한의 안전장치(허용 모듈 화이트리스트, 타임아웃 등)는 고려하는 게 좋음 — 특히 실제 제조 데이터/설비 제어와 연결되는 맥락이라면 더 중요함.

**`recursion_limit` 설계**

순환형 구조를 쓴다면 실제 제조 이상 분석이 몇 라운드 안에 결론이 나야 하는지 가늠해서 `recursion_limit`을 넉넉하지만 무한하지 않게 설정 — 데모 중 무한 루프로 멈추는 상황을 방지.

---

## 9. Q&A로 정리한 핵심 개념

- **Network 패턴이란**: 중앙 supervisor 없이 각 에이전트가 서로 handoff하거나(도구 기반) 조건부 엣지로 순환하는(이 교재 방식) 분산형 멀티에이전트 구조. Supervisor 대비 유연하지만 통제력·관찰성은 떨어지고, 무한 루프 위험 관리가 더 중요해짐.
- **메시지를 HumanMessage로 바꾸는 것**: Network냐 Supervisor냐라는 아키텍처 선택과는 별개로, "핸드오프 도구/노드를 어떻게 구현하느냐"에 달린 선택. 기본 handoff는 role 그대로 이어붙이고, `task_description`이나 이번 교재의 `research_node`처럼 다음 에이전트가 "새로운 요청"처럼 받아들이게 하고 싶을 때 `HumanMessage`로 재구성함.
- **`name` 필드**: Supervisor는 그래프 구조가 발신자를 암시해주지만, Network는 공유 메시지 리스트를 여러 에이전트가 순환하며 쓰기 때문에 `name` 없이는 서로 발언을 구분할 방법이 없음 — Network 패턴에서 사실상 필수.
- **라우터 함수에 LLM이 없는 이유**: 라우터는 판단을 새로 하는 게 아니라, 이미 LLM이 응답 생성 시점에 내린 판단(system_prompt로 약속한 `FINAL ANSWER` 키워드)을 문자열 검사로 읽어서 그래프 이동으로 번역하는 역할만 함.
- **`Annotated`와 `@tool`의 관계**: `Annotated`는 파이썬 표준 타입힌트 문법이라 `@tool` 유무와 무관하게 항상 쓸 수 있음. `@tool`(또는 `create_agent`의 자동 도구 변환 로직)은 그 안의 정보(설명 문자열, `InjectedState`/`InjectedToolCallId` 마커)를 읽어서 도구 스키마를 만드는 데 활용하는 소비자(consumer) 쪽이지, `Annotated` 사용을 가능하게 해주는 주체가 아님.
