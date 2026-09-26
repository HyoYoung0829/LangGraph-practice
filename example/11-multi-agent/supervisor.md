# LangGraph 멀티에이전트 — Supervisor 패턴 정리

> 교재: `01-LangGraph-Supervisor.ipynb` + 학습 중 나눈 Q&A 정리

---

## 1. 멀티에이전트가 필요한 상황

단일 에이전트에 도구를 계속 몰아넣으면 생기는 문제:

- **도구 과부하**: 도구가 너무 많아지면 LLM이 어떤 걸 써야 할지 잘못 판단함
- **컨텍스트 오버플로우**: 대화/작업 범위가 커지면 한 에이전트가 추적하기엔 컨텍스트가 너무 커짐
- **전문화 필요**: 계획, 조사, 계산처럼 역할이 다른 작업을 한 프롬프트에 다 욱여넣으면 품질이 떨어짐

**비유**: 회사 조직에서 프로젝트 매니저(Supervisor)가 개발팀(Research), 재무팀(Math)에 업무를 배분하는 구조와 같음. 각 팀은 자기 전문 분야에만 집중하고, 매니저가 전체를 조율.

---

## 2. 멀티에이전트 두 가지 패턴

| 패턴 | 작동 방식 | 제어 흐름 | 사용처 |
|---|---|---|---|
| **Tool Calling (Subagents)** | supervisor가 다른 에이전트를 도구로 호출. 도구 에이전트는 결과만 반환 | 중앙 집중식 — 모든 라우팅이 호출 에이전트를 통과 | 작업 오케스트레이션, 구조화된 워크플로우 |
| **Handoffs (Supervisor)** | 현재 에이전트가 다음 에이전트로 제어권 자체를 전달 | 분산형 — 에이전트가 활성 에이전트를 직접 바꿈 | 다중 도메인 대화, 전문가 인계 |

### 2-1. Tool Calling 패턴 — 가장 단순한 형태

```python
@tool
def calculator(expression: str) -> str:
    """수학 표현식 계산 도구"""
    return str(eval(expression))

tc_math_agent = create_agent(model, tools=[calculator], system_prompt="...")

@tool("math_expert", description="수학 계산과 문제 해결에 이 도구를 사용하세요.")
def call_math_agent(query: str) -> str:
    """수학 전문 에이전트 호출 도구"""
    result = tc_math_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

main_agent = create_agent(model, tools=[call_math_agent], system_prompt="...")
```

- 서브 에이전트를 그냥 `@tool`로 감싸서, 메인 에이전트 입장에선 "함수 호출 하나"로 보임
- 서브 에이전트가 실행되는 동안 메인 에이전트는 개입 못 함 — `.invoke()` 끝날 때까지 블로킹, 결과 문자열만 돌려받음
- 구현이 간단하지만, 에이전트 간 유연한 제어(중간에 다른 에이전트로 바로 넘기기 등)는 안 됨

### 2-2. Handoffs(Supervisor) 패턴 — 이 챕터의 메인 주제

```
1. Supervisor: "조사가 필요하다" → Research Agent로 핸드오프
2. Research Agent: 정보 수집 완료 → Supervisor로 복귀
3. Supervisor: "계산이 필요하다" → Math Agent로 핸드오프
4. Math Agent: 계산 완료 → Supervisor로 복귀
5. Supervisor: 최종 결과 종합
```

`Command` 객체로 실제 제어권 자체를 다음 에이전트에게 넘김 — Tool Calling처럼 "결과만 받아오는" 게 아니라 그래프의 실행 흐름 자체가 이동함.

---

## 3. 워커 에이전트 만들기 — 복습

```python
research_agent = create_agent(
    model="gpt-5.4",
    tools=[web_search],
    system_prompt=(
        "You are a research specialist agent.\n"
        "STRICT GUIDELINES:\n"
        "- Focus ONLY on research and information gathering tasks\n"
        "- DO NOT perform mathematical calculations\n"
        "- When task is complete, report findings directly to the supervisor"
    ),
    name="research_agent",
)
```

- 각 워커는 "자기 역할만 하고, 다른 영역은 절대 침범하지 않는다"는 걸 system_prompt에 명시 — 전문화 경계가 흐려지면 라우팅 의미가 없어짐
- `math_agent`의 도구 함수(`add`, `multiply` 등)는 `@tool` 없이 그냥 `def`로 정의 — `create_agent`가 자동으로 도구 변환 (독스트링이 곧 LLM한테 전달되는 설명이라는 점은 그대로 적용)

---

## 4. `langgraph-supervisor` 라이브러리로 빠르게 구현

```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    model=init_chat_model("gpt-5.4"),
    agents=[research_agent, math_agent],
    system_prompt="...",
    add_handoff_back_messages=True,   # 에이전트 복귀 시 핸드오프 메시지 자동 추가
    output_mode="full_history",        # 전체 대화 히스토리를 출력에 포함
).compile()
```

- `create_supervisor` 하나로 handoff 도구 생성, 그래프 구성, 워커 연결까지 자동 처리
- `output_mode="full_history"` vs `"last_message"`: 최종 출력에 중간 과정 전체를 포함할지, 마지막 메시지만 포함할지 선택 가능

**주의 — 지금은 공식 권장 패턴이 아님**

`langgraph-supervisor` 레포(`langchain-ai` 공식 관리)에 이런 공지가 붙어 있음:

> "대부분의 경우, 이 라이브러리보다 supervisor 패턴을 **tool-calling 방식으로 직접 구현**하는 걸 권장한다. 그게 컨텍스트 엔지니어링에 대한 통제력이 더 높다."

즉 이 교재의 "커스텀 Supervisor 직접 구현" 파트(5장)가 **지금 실무에서 실제로 권장되는 패턴**이고, 라이브러리 버전은 "빠른 프로토타입" 또는 "라이브러리가 해결 못 하는 문제가 없는 단순한 경우"에만 적합하다고 보면 됨.

---

## 5. 커스텀 Supervisor 직접 구현 — Handoff 도구

### 5-1. 기본 핸드오프 도구

```python
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],       # 현재 State 자동 주입
        tool_call_id: Annotated[str, InjectedToolCallId],      # 이 호출의 id 자동 주입
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,               # 다음 노드 지정
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,          # 부모(supervisor) 그래프 레벨에서 라우팅
        )
    return handoff_tool
```

**핵심 개념 3가지**

| 개념 | 역할 |
|---|---|
| `InjectedState` | LLM은 모르게, 도구 실행 시점의 그래프 State(기본: 전체)를 자동으로 함수 인자에 주입 |
| `InjectedToolCallId` | 이 도구 호출의 고유 id를 자동 주입 — `ToolMessage`를 만들 때 "어느 호출에 대한 응답인지" 짝짓는 데 필수 |
| `Command(goto=..., graph=Command.PARENT)` | 다음 노드로 실제 이동을 일으키는 라우팅 지시. 서브그래프 안에서 부모 그래프의 흐름을 바꿔야 하므로 `PARENT` 지정 |

**왜 전역 변수로 안 꺼내 쓰고 굳이 주입 방식을 쓰나**: LLM이 한 스텝에서 도구 호출을 여러 개 동시에(병렬로) 요청할 수 있음. 전역에서 "지금 실행 중인 값"을 꺼내 쓰면 병렬 실행 중 레이스 컨디션이 생길 수 있음. 주입 방식은 각 호출 전용 값을 함수 인자로 직접 넘겨서 이 문제를 원천 차단함 (부수 효과로 순수 함수처럼 단위 테스트도 쉬워짐).

### 5-2. 그래프로 조립

```python
supervisor_agent = create_agent(
    model="gpt-5.4",
    tools=[assign_to_research_agent, assign_to_math_agent],
    system_prompt="...",
    name="supervisor",
)

custom_supervisor = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent, destinations=("research_agent", "math_agent", END))
    .add_node(research_agent)
    .add_node(math_agent)
    .add_edge(START, "supervisor")
    .add_edge("research_agent", "supervisor")   # 워커는 항상 supervisor로 복귀
    .add_edge("math_agent", "supervisor")
    .compile()
)
```

- `destinations=(...)`는 라우팅 로직이 아니라 **시각화용 힌트**. 실제 라우팅은 handoff 도구 안의 `Command(goto=...)`가 함
- 워커→supervisor 엣지가 고정돼 있어서, 멀티스텝 작업에서 매번 "다음엔 뭘 해야 하나"를 supervisor가 다시 판단할 기회를 가짐

---

## 6. 고급 작업 위임 — `Send`로 컨텍스트 정제해서 넘기기

### 기존 방식(암묵적 위임) vs 이 방식(명시적 위임)

| | 암묵적 위임 (5장) | 명시적 위임 (6장) |
|---|---|---|
| 워커가 받는 것 | 전체 대화 히스토리 (`state["messages"] + [...]`) | supervisor가 작성한 작업 지시문 하나 |
| 워커의 일 | 전체 맥락에서 "내가 뭘 해야 하는지" 스스로 파악 | 이미 정리된 지시를 바로 실행 |
| 장점 | 정보 손실 없음 | 컨텍스트 낭비 없음, 명확한 단일 작업만 전달 |
| 단점 | 대화가 길어질수록 워커 입력도 계속 커짐 | supervisor가 지시를 부실하게 쓰면 정보가 통째로 증발 |

```python
def create_task_description_handoff_tool(*, agent_name: str, description=None):
    name = f"transfer_to_{agent_name}"

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str, "Detailed description of what the agent should do..."],  # LLM이 직접 작성
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}  # 메시지를 1개로 교체
        return Command(
            goto=[Send(agent_name, agent_input)],   # 이 노드를 이 입력값으로 실행하라
            graph=Command.PARENT,
        )
    return handoff_tool
```

- `task_description`은 (Injected가 아닌) **일반 파라미터** — supervisor의 LLM이 위임 시점에 직접 작문해서 채움
- `Send(agent_name, agent_input)`: "그 노드로 가라"뿐 아니라 "**이 커스텀 입력값으로** 실행해라"까지 지정 — `update`로 State를 바꾸는 것과 달리 그 노드 하나만을 위한 독립적인 입력을 만들 수 있음

**트레이드오프 판단 기준** (트리밍 vs 요약과 같은 논리)

| 조건 | 유리한 방식 |
|---|---|
| 대화가 짧고 단순 | 암묵적 위임 |
| 같은 워커가 세션 내 여러 번 재호출돼 연속성이 필요 | 암묵적 위임 |
| 긴 대화라 노이즈가 많고 토큰 비용이 실제 문제 | 명시적 위임 |
| supervisor 모델이 요약을 신뢰할 만큼 강력함 | 명시적 위임 |

---

## 7. 계층적 에이전트 (Tool Calling 패턴으로 3단계 중첩)

```python
# Level 3: 최하위 — 기본 산술
basic_math = create_agent(model, tools=[add_numbers], ...)
@tool("basic_math", ...)
def call_basic_math(query): return basic_math.invoke(...)

# Level 2: 중간 — 기본 도구를 다시 도구로 사용
intermediate_math = create_agent(model, tools=[call_basic_math], ...)
@tool("intermediate_math", ...)
def call_intermediate_math(query): return intermediate_math.invoke(...)

# Level 1: 최상위 — 문제를 분해해서 중간 레벨에 위임
advanced_math = create_agent(model, tools=[call_intermediate_math], ...)
```

- Handoff가 아니라 **Tool Calling 패턴을 계층으로 쌓은 것** — 각 레벨이 자기 하위 레벨을 그냥 "도구"로 취급
- 복잡한 문제를 재귀적으로 쪼개는 구조가 필요할 때 쓰는 패턴. Supervisor(Handoff) 패턴보다 구현이 단순하지만, 레벨 간 유연한 제어 이전은 안 됨

---

## 8. 실전 응용: 고객 지원 시스템

- 기술 지원 / 청구 지원 / 일반 지원, 세 전문 에이전트를 Tool Calling 패턴으로 구성
- 각 에이전트는 자기 도메인 도구만 가짐 (기술: `check_system_status`, `restart_service` / 청구: `check_invoice`, `process_refund`)
- Supervisor가 문의 내용을 분석해서 적절한 전문 에이전트로 라우팅

**패턴 선택 포인트**: 이 예제가 Handoff 대신 Tool Calling을 쓴 이유는, "사용자가 새 에이전트와 직접 상호작용해야 하는" 상황이 아니라 "결과만 받아서 supervisor가 정리해서 답하면 되는" 상황이기 때문. 사용법 선택 기준:
- 사용자가 전문가와 **직접 이어서 대화**해야 함 → Handoff
- Supervisor가 결과만 취합해서 **하나의 최종 답변**으로 정리하면 됨 → Tool Calling

---

## 9. 베스트 프랙티스

### 9-1. 에이전트 전문화 원칙

- **단일 책임**: 에이전트 하나는 하나의 명확한 역할만
- **상호 보완적**: 에이전트들이 서로 다른 강점을 가져야 함 (겹치는 역할 X)
- **명확한 경계**: "언제 어떤 에이전트를 써야 하는지"가 system_prompt만 봐도 애매하지 않아야 함
- 피해야 할 패턴: `Research Agent 1, 2, 3...`처럼 유사 기능이 중복되거나, "모든 걸 다 하는 Agent"처럼 역할이 너무 넓은 경우

### 9-2. 오류 처리

| 상황 | 대응 |
|---|---|
| API 한도 초과, 네트워크 오류 | 재시도 메커니즘 (최대 2~3회) |
| Research Agent 실패 | 대체 정보 소스 준비 |
| 일부 정보만 수집됨 | 완전하지 않아도 사용 가능한 부분 결과는 전달 |
| 문제 발생 시 | 사용자에게 투명하게 상황 공유 |

### 9-3. 성능 최적화

| 전략 | 설명 | 주의점 |
|---|---|---|
| 병렬 처리 | 독립적인 작업은 동시에 여러 에이전트에 위임 | 상태 충돌 방지용 동기화 필요 |
| 캐싱 | 반복 요청 결과를 재사용 | 시간에 민감한 데이터는 TTL 필요 |
| 작업 배치 | 유사 작업을 묶어서 한 번에 처리 | 배치 크기·지연시간 트레이드오프 |
| 조기 종료 | 충분한 정보 모이면 즉시 종료 | 완전성과 효율성 균형 |

---

## 10. 모델 수준 선택 — Supervisor vs Worker

기준은 "supervisor냐 아니냐"가 아니라 **"이 노드가 얼마나 복잡한 판단을 하는가"**.

- **Supervisor가 보통 상위 모델인 이유**: 누구한테 위임할지 판단, 컨텍스트 요약/재구성(`task_description` 작성), 여러 결과 종합, 멀티스텝에서 "다음엔 뭘 해야 하나" 연쇄 판단 — 여기서 실수하면 전체가 틀어짐
- **Worker는 작업 성격에 따라 갈림**: 단순 도구 호출·포맷 변환은 경량 모델로 충분, 복잡한 전문 추론(수학 증명, 코드 리뷰)이 필요하면 supervisor급 이상도 필요할 수 있음
- **멀티스텝 문제**란 한 번의 위임으로 안 끝나고 여러 라운드의 위임-복귀를 반복해야 하는 작업 성격을 말함 (예: "GDP 검색 → 검색 → 비율 계산" 3단계). 이런 작업일수록 supervisor의 판단 정교함이 더 중요해짐

**해커톤처럼 짧은 기간엔**: 처음엔 전부 동일한 가벼운 모델로 굴러가게 만들고, 실제로 라우팅/요약 품질이 아쉬운 지점(대개 supervisor)만 골라서 업그레이드하는 게 효율적.

---

## 11. `subgraphs=True` — 스트리밍 부가 옵션

```python
for chunk in advanced_supervisor.stream({"messages": [...]}, subgraphs=True):
```

- 멀티에이전트는 그래프 안에 그래프가 들어있는 구조 (워커 에이전트 각각이 이미 하나의 ReAct 루프 그래프)
- 기본값(`False`)은 "supervisor → research_agent → supervisor"처럼 **노드 단위 요약**만 보여줌
- `subgraphs=True`는 각 워커 **내부**에서 일어나는 세부 스텝(어떤 도구를 몇 번 호출했는지 등)까지 펼쳐서 보여줌
- 그래프 실행 로직 자체엔 영향 없는, 순전히 "얼마나 자세히 보여줄지" 결정하는 스트리밍 전용 파라미터 — 디버깅/투명성 확보 목적일 때 켬
- `subgraphs=True`를 켜면 `chunk`가 `(namespace, data)` 튜플 형태가 되므로, `pretty_print_messages` 같은 헬퍼로 파싱해서 출력하는 게 일반적

---

## 12. 한화 제조 데이터 해커톤 적용 아이디어

**에이전트 분리 후보**
- **Sensor Research Agent**: 센서/로그 조회 전담 (MCP 도구로 노출한 센서 쿼리, 이상 로그 검색)
- **Analysis Agent**: 수치 계산·통계 분석 전담 (이상 수치 비율, 트렌드 계산)
- **Report Agent**: 최종 리포트 포맷팅 전담

**패턴 선택**
- 심사에서 "왜 이 판단을 했는지" 설명 가능해야 한다면 → **Handoff(Supervisor) 패턴**으로 직접 구현. 각 핸드오프 시점이 명확한 단계 구분점이 되어 발표 때 그래프 시각화(`visualize_graph`)로 보여주기 좋음
- 단순히 "여러 전문 기능을 조합"하는 수준이면 → **Tool Calling 패턴**이 구현이 빠르고 안전함 (서브에이전트가 사용자와 직접 상호작용할 필요가 없다면 이쪽이 더 간단)

**컨텍스트 관리 선택**
- 원본 로그가 길다면 암묵적 위임(전체 맥락 전달)보다 **명시적 위임(`Send` + `task_description`)**이 유리 — supervisor가 "몇 번 라인의 몇 시 이상 데이터를 분석해줘" 식으로 요약해서 넘기면 워커가 불필요한 노이즈 없이 작업 가능
- 단, supervisor 요약이 부실하면 중요 조건(측정 단위, 기준 임계값 등)이 통째로 빠질 위험 — 이 경우 원본 데이터 일부를 함께 첨부하는 하이브리드 방식도 고려

**짧은 기간 고려한 순서**
1. Tool Calling 패턴으로 워커 1~2개(조사, 계산)부터 빠르게 동작 확인
2. 시간 남으면 Handoff 패턴 + `Send` 기반 명시적 위임으로 전환해서 판단 근거를 명확히 보여줄 수 있게 확장
3. `subgraphs=True`로 실행 과정을 캡처해두면 발표 자료로 그대로 활용 가능

---

## 13. Q&A로 정리한 핵심 개념

- **`convert_to_messages`**: 튜플/딕셔너리/문자열 등 제각각인 메시지 표현을 `BaseMessage` 리스트로 통일해주는 유틸. 헬퍼 함수(`pretty_print_messages`)에서 출력 포맷 통일용으로 사용됨. (반대 방향은 `convert_to_openai_messages`)
- **도구 자동 변환**: `create_agent`에 `@tool` 없이 일반 `def` 함수를 넘겨도 자동으로 도구로 변환됨. 단, 독스트링은 장식이 아니라 **LLM이 도구 사용 여부를 판단하는 유일한 근거**라서 잘 써야 함. 이름/설명/스키마를 세밀히 제어하고 싶을 때만 `@tool`을 명시적으로 붙임.
- **`langgraph-supervisor`**: LangChain 공식 라이브러리(`langchain-ai`)가 맞지만, 현재는 "대부분의 경우 tool-calling 방식으로 직접 구현하는 걸 권장"하는 상태 — 컨텍스트 엔지니어링 통제력 때문.
- **`InjectedState` / `InjectedToolCallId`**: LLM이 채우는 값이 아니라 시스템이 도구 실행 시점에 자동으로 주입하는 값. Command를 리턴하는 도구(State를 직접 갱신)에서 `InjectedToolCallId`는 거의 항상 필요하고, `InjectedState`는 그중 "이전 맥락을 읽어야 판단 가능한" 경우에만 추가로 필요 — 현업에서 이 둘을 쓰는 도구 자체가 전체 도구 중 소수(단순 값 리턴 도구가 대다수).
- **`InjectedState` 주입 범위**: `field` 미지정 시 도구가 호출되는 그 순간의 최신 State 전체(스키마에 정의된 모든 필드)가 딕셔너리로 옴. `InjectedState("messages")`처럼 지정하면 그 필드 값만 좁혀서 받음.
- **`create_handoff_tool`**: 예약어 아님, 그냥 사용자 정의 함수 이름. `langgraph_supervisor.create_handoff_tool`이라는 동일 이름의 라이브러리 함수가 실제로 있어서, 교재가 "라이브러리 없이 직접 구현하면 이런 모습"이라는 걸 보여주려 같은 이름을 붙인 것으로 보임.
- **암묵적 위임 vs 명시적 위임(`Send`)**: 무조건 후자가 낫지 않음. 트리밍 vs 요약과 같은 구조의 트레이드오프 — "정보 손실 없이 노이즈 감수" vs "깔끔하지만 요약 품질에 의존".
