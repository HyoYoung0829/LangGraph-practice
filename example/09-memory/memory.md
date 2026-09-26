# LangGraph 메모리 정리

> 교재: `01-LangGraph-Add-Memory.ipynb` + 학습 중 나눈 Q&A 정리

---

## 1. 왜 메모리가 필요한가

LLM 자체는 stateless. 매 호출마다 아무것도 기억 못 함. 여러 턴에 걸친 대화나, 그래프가 분기·루프를 도는 복잡한 실행 흐름을 다루려면 "지금까지 뭐가 있었는지"를 어딘가에 저장하고 다시 불러올 방법이 필요함.

랭그래프는 이걸 **Checkpointer**로 해결함. 이때 핵심은, 저장되는 게 대화 로그 한 줄이 아니라 **그래프의 State 전체**라는 것. 메시지 목록뿐 아니라 현재 실행 위치, State에 정의된 커스텀 필드, pending task까지 다 포함됨. 그래서 interrupt/resume, time travel(체크포인트로 되돌아가기), 장애 복구 같은 게 가능해짐.

---

## 2. 단기 메모리: Checkpointer + thread_id

### 기본 사용

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### thread_id는 checkpointer와 항상 세트

- **checkpointer** = 저장 인프라 (파일 캐비닛)
- **thread_id** = 그 안에서 어느 세션에 저장/조회할지 지정하는 키 (서랍 이름표)

```python
config = {"configurable": {"thread_id": "conversation_1"}}
graph.invoke({"messages": [...]}, config)
```

- checkpointer 있음 + thread_id 없음 → 에러 (어디에 저장할지 알 수 없음)
- checkpointer 없음 → thread_id 줘도 의미 없음, 매번 새 대화로 취급
- 같은 thread_id로 다시 invoke → 이전 State를 불러와서 이어감
- 다른 thread_id → 완전히 새로운 세션 (여러 사용자/세션 분리에 사용)

### 저장된 State 조회

```python
snapshot = graph.get_state(config)          # 현재 상태 (values, config, next)
graph.get_state_history(config)              # 전체 체크포인트 이력 (시간 역순)
```

---

## 3. State 정의와 `add_messages` 리듀서

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**주의**: 여기서 독스트링(`"""..."""`)은 사람이 읽는 설명일 뿐, 동작에는 아무 영향 없음. 실제로 메시지가 쌓이게 만드는 건 `Annotated[list, add_messages]`의 `add_messages` 리듀서.

- 리듀서가 없으면 노드가 반환한 값이 State를 **덮어씀**
- `add_messages`가 있으면 반환된 메시지를 기존 리스트에 **id 기준으로 병합**:
  - 새 id → append
  - 기존 id → 해당 메시지 교체
  - `RemoveMessage(id=...)` → 해당 id 메시지 제거

이 id 기반 병합 로직 하나가 트리밍, 삭제, 요약 전부의 기반이 됨.

---

## 4. 컨텍스트 관리 3전략 — 핵심 차이

세 가지 다 "LLM에 들어가는/State에 쌓이는 걸 줄인다"는 목적은 같은데, **무엇을 어디까지 건드리냐**가 다름.

| | 대상 | State/checkpointer 영향 | LLM 관여 |
|---|---|---|---|
| **Trimming** | 이번 LLM 호출 1회 | 없음 (원본 그대로 보존) | 없음 (토큰 카운팅만) |
| **RemoveMessage** | State 자체 | 영구 삭제 | 없음 |
| **Summarization** | State 자체 | 원본 삭제, 압축본으로 대체 | 있음 (요약 생성 호출 1회) |

### 4-1. Trimming

```python
from langchain.messages import trim_messages

def chatbot(state: State):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        max_tokens=500,
        token_counter=token_counter,
        start_on="human",
        include_system=True,
    )
    response = llm.invoke(trimmed)          # ← 트리밍된 사본만 LLM에 전달
    return {"messages": [response]}          # ← State에는 응답 1개만 반환
```

**핵심**: `trimmed`는 이 함수 안에서만 존재하는 임시 변수. `state["messages"]`(원본, checkpointer 저장분)는 전혀 안 건드림. → "트리밍은 상태를 변경하지 않는다"는 말의 정확한 의미.

**한계**: 컨텍스트 윈도우 초과 문제는 트리밍만으로 완전히 해결됨. 근데 잘려나간 부분에 있던 정보는 **이번 호출에서 LLM이 그냥 못 봄**. State엔 남아있어도 안 보내면 없는 거나 마찬가지.

### 4-2. RemoveMessage (진짜 삭제)

```python
from langchain.messages import RemoveMessage

# 수동 삭제
app.update_state(config, {"messages": RemoveMessage(id=messages[0].id)})

# 그래프 노드 안에서 자동 삭제
def delete_messages(state: MessagesState):
    messages = state["messages"]
    if len(messages) > 3:
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-3]]}
    return {}
```

`update_state()`는 `invoke()` 없이 checkpointer의 State를 직접 수정하는 API. `RemoveMessage(id=...)`를 messages 필드에 넣으면 `add_messages` 리듀서가 "이건 삭제 요청이구나" 인식하고 해당 id 메시지를 State에서 제거함. **영구적**이라 time travel로도 복구 불가능.

**용도**: 저장 공간 절감, 민감정보 파기처럼 "저장소 자체 크기"가 문제일 때.

### 4-3. Summarization

```python
class SummaryState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str

def maybe_summarize(state: SummaryState):
    if len(state["messages"]) > 6:
        summary = summarize_conversation(state["messages"][:4])  # LLM 호출로 요약 생성
        # 요약을 시스템 메시지로 유지하고 원본은 RemoveMessage로 삭제
```

**요약이 필요한 상황**: "컨텍스트 윈도우가 유한함(트리밍으로 해결 가능)" 만으로는 요약이 필요한 이유가 안 됨. 요약은 다음 두 조건이 **동시에** 맞을 때 씀:

1. 대화가 길어져서 뭔가는 반드시 잘라야 함
2. 그 잘리는 부분에 나중 답변에도 계속 필요한 정보가 섞여 있음

(1)만 참이면 트리밍으로 충분. (2)까지 참이어야 압축이라도 해서 정보를 살려두는 요약이 의미 있음. 대신 요약은 LLM 호출이 추가로 드니 비용·레이턴시가 늘어남 — "요약이 트리밍보다 싸서" 쓰는 게 아니라 정보 손실을 막으려고 추가 비용을 감수하는 것.

**주의**: 이름·알레르기 같은 개별 사실은 요약(뭉뚱그려진 텍스트 한 덩어리)보다 장기 메모리 **Store**에 구조화된 필드로 따로 저장하는 게 더 안전함. 요약은 매턴 갱신되면서 "요약의 요약"을 거치면 디테일이 흐려질 수 있음.

---

## 5. 프로덕션 체크포인터

| 체크포인터 | 특징 | 적합한 상황 |
|---|---|---|
| `InMemorySaver` | 프로세스 메모리 상주, 재시작 시 소실 | 개발/테스트 |
| `SqliteSaver` | 가벼운 로컬 파일 저장소 | 소규모, 단일 서버 |
| `PostgresSaver` | ACID 준수, 영속 보장 | 엔터프라이즈, 감사 필요한 데이터 |
| `RedisSaver` | 인메모리 기반, 매우 빠름, 영속성은 옵션 | 활발히 진행 중인 세션, TTL 걸어 자동 만료시킬 단기 State |

**Redis는 "일반 DB"가 아님** — RDBMS는 디스크 기반 + SQL + ACID 우선 설계, Redis는 메모리 상주 + key-value + 속도 우선 설계(영속성은 부차적). "얼마나 오래, 얼마나 안전하게 보관해야 하는가"로 판단하면 선택이 명확해짐.

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # 최초 1회 테이블 생성
    graph = graph_builder.compile(checkpointer=checkpointer)
    graph.invoke(..., config)
```

`with`는 컨텍스트 매니저 — 블록을 벗어나면 (정상이든 에러든) DB 커넥션이 자동으로 close됨. try/finally를 대신 처리해주는 문법.

---

## 6. 교재 코드에서 실무 기준으로 바꿔야 할 부분

노트북 자체는 개념 설명용으로는 틀린 게 없음. 다만 실제 프로덕션에서 쓰는 패턴과는 차이가 있어서 정리함.

### 6-1. 트리밍을 노드마다 직접 짜는 것 → `pre_model_hook` / `@before_model` 미들웨어로 중앙화

교재처럼 `call_model_with_trimming` 안에 트리밍 로직을 박아두면, LLM 호출 지점이 늘어날 때마다 같은 코드를 계속 복붙해야 함. 실무에서는 이걸 그래프/에이전트 레벨의 훅으로 뺌:

```python
# langgraph.prebuilt.create_react_agent 사용 시
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

def pre_model_hook(state):
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,  # 정확한 토큰 계산용 커스텀 함수 대신 표준 유틸
        max_tokens=384,
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"llm_input_messages": trimmed}  # State엔 반영 안 되고 이번 호출에만 사용

agent = create_react_agent(model, tools=[...], pre_model_hook=pre_model_hook, checkpointer=memory)
```

최신 `langchain.agents.create_agent`(langchain 1.0 계열)를 쓰는 경우엔 `@before_model` 데코레이터 미들웨어로 동일한 역할을 함.

### 6-2. `summarize_conversation` 직접 구현 → 내장 요약 모듈 사용

교재의 `SummaryState` + 수동 `maybe_summarize` 함수는 요약 전략의 원리를 배우기엔 좋은데, 실무에서는 이미 만들어진 모듈을 씀:

- `langmem.short_term.SummarizationNode` — `pre_model_hook`으로 붙여서 자동 요약
- `langchain.agents.middleware.SummarizationMiddleware` — 최신 `create_agent` 계열에서 사용

직접 구현하면 프롬프트 튜닝, 요약 트리거 조건, 원본 삭제 타이밍 같은 걸 다 손수 관리해야 하는데, 이 모듈들은 이미 검증된 형태로 제공됨.

### 6-3. `RemoveMessage`를 루프 돌려서 개별 삭제 → 전체 초기화 시 `REMOVE_ALL_MESSAGES` 활용

메시지 전체를 지우고 새로 시작하고 싶은 경우 (예: 요약 후 원본 통째로 비우기) `langgraph.graph.message.REMOVE_ALL_MESSAGES` 센티널을 쓰면 개별 id를 순회할 필요 없이 한 번에 처리 가능.

### 6-4. `with PostgresSaver...` 를 매 호출마다 여는 패턴 → 실제 서비스에선 커넥션을 앱 시작 시 1회만 열기

교재는 셀 단위로 `with` 블록을 계속 새로 여는데, 이건 "재연결해도 데이터 유지되는지" 보여주려는 교육용 구성. 실제 서비스(API 서버 등)에서는 요청마다 DB 커넥션을 새로 열고 닫으면 오버헤드가 큼 → 앱 시작 시 커넥션(or 커넥션 풀)을 한 번 만들어서 계속 재사용하는 구조로 감.

---

## 7. 한화 제조 데이터 해커톤 적용 아이디어

11월 1일 시작, 2주 진행. 제조 데이터 기반 AX 에이전트를 만든다면 메모리 설계는 대략 이런 축으로 나눌 수 있음:

**단기 메모리 (thread_id 단위)**
- 설비/라인 하나에 대한 진단 세션 하나를 thread_id로 매핑 (`thread_id = f"line_{line_id}_{session_date}"`)
- 센서 로그·이상 탐지 결과를 여러 턴에 걸쳐 주고받으면서 "이 설비, 아까 그 이상값이랑 연관 있어?" 같은 후속 질문을 이어갈 수 있게 함
- 데모에서는 `InMemorySaver`로 충분, 팀 내 데모/재현이 필요하면 `SqliteSaver`로 가볍게 영속화

**컨텍스트 관리**
- 센서 로그는 raw 데이터량이 크니까, LLM에 넘기기 전에 트리밍/요약 조합 고려 (예: 최근 N개 이상 이벤트는 원문 유지, 그 이전 건 요약해서 "지난주 A라인에서 반복된 이상 패턴: ..." 형태로 압축)
- 여기서 "정보 손실 감수 가능한가"가 판단 기준 — 이상 탐지 원인 분석처럼 과거 패턴이 계속 필요한 태스크면 요약, 단순 질의응답이면 트리밍으로 충분

**장기 메모리 (Store, thread 넘나듦)**
- 설비별 고유 특성 (예: "3호기는 재작년부터 진동 이슈 반복"), 담당자별 선호 리포트 포맷 같은 건 thread 하나에 묶이는 게 아니라 여러 세션에 걸쳐 계속 필요한 정보 → 장기 메모리 Store에 구조화해서 저장
- 요약문 하나로 뭉뚱그리기보다 `store.put(namespace, "equipment_3_known_issues", {...})` 식으로 필드화해두면 나중에 정확히 꺼내 쓸 수 있음

**심사 관점에서 챙기면 좋을 것**
- "왜 이 메모리 전략을 택했는지"를 설명할 수 있으면 좋음 (트리밍 vs 요약 vs Store 선택 근거)
- 짧은 해커톤 기간엔 `InMemorySaver` + 단순 트리밍으로 먼저 동작하는 걸 만들고, 시간 남으면 Store 기반 장기 메모리나 요약 미들웨어로 확장하는 순서가 안전

---

## 8. Q&A로 정리한 헷갈렸던 지점

- **Checkpointer ≠ Retriever**: Retriever는 외부에서 정보를 "검색"해오는 도구(그래프의 노드), Checkpointer는 그래프 State를 "저장/복원"하는 인프라(compile 시 붙이는 설정). 저장이냐 검색이냐가 근본적으로 다름.
- **`add_messages`는 checkpointer 선언이 아님**: 리듀서는 "한 번의 invoke 안에서 messages를 어떻게 병합할지"를 정하는 것. checkpointer는 "invoke가 끝난 뒤에도 State를 남길지"를 정하는 것. 서로 독립적으로 동작하고, 둘 다 있어야 "여러 턴에 걸쳐 메시지가 누적되는" 결과가 나옴.
- **컨텍스트 윈도우 = 한 번의 invoke 시점에 모델에 들어가는 입력 토큰 한도**. 이건 트리밍만으로 완전히 해결 가능. 요약이 별도로 필요해지는 건 "잘려나가는 부분에도 계속 필요한 정보가 있을 때"뿐.
