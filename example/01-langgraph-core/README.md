# 랭그래프(LangGraph) 기초 정리

## 1. 랭그래프의 포지션

- **랭그래프 = 랭체인 생태계 안의 오케스트레이션 프레임워크**
  - 랭체인: LLM 호출, 프롬프트, 체인, 리트리버 등 컴포넌트 라이브러리
  - 랭그래프: 그 컴포넌트들을 **상태 기반 그래프**로 어떻게 연결하고 흐름을 제어할지 담당
  - 랭체인의 체인(Chain)이 "A → B → C" 직렬 구조라면, 랭그래프는 분기 + 루프(cycle)까지 표현 가능한 구조

- **랭체인과의 관계: 단독 사용 가능, 다만 실무에선 같이 쓰는 편**
  - 최근 버전은 랭체인 의존성이 분리되어 있어 랭체인 없이 OpenAI/Anthropic SDK만으로도 노드 작성 가능
  - 그럼에도 실무에서 같이 쓰는 이유: 기존 랭체인 컴포넌트(PromptTemplate, 리트리버, 구조화 출력 파서 등) 재사용이 편해서
  - 랭체인을 추가로 설치하는 대표 시점: 여러 LLM 프로바이더 통일 인터페이스 필요 시 / 기존 체인·리트리버 재사용 시
  - 비슷한 포지션의 타 프레임워크: CrewAI, AutoGen, LlamaIndex Workflows

---

## 2. 핵심 개념 ①: State (상태)

그래프 전체에서 여러 노드가 공유하는 데이터 구조. 랭그래프의 모든 동작은 결국 "State를 어떻게 정의하고, 어떻게 업데이트하는가"로 귀결됨.

### 2.1 State 정의 방식

```python
from typing_extensions import TypedDict

class State(TypedDict):
    input: str
    result: str
```

| | TypedDict | Pydantic (BaseModel) |
|---|---|---|
| 런타임 검증 | 안 함 (타입 힌트일 뿐) | 함 (타입 안 맞으면 에러/변환) |
| 오버헤드 | 없음 | 있음 |
| 적합한 곳 | 노드끼리 주고받는 내부 State | 외부 입력(API, 사용자 입력, LLM structured output) 검증 지점 |

→ 실무 기본값: **State는 TypedDict로 가볍게, 외부 입력 검증이 필요한 지점만 Pydantic을 섞어 쓰는** 조합

### 2.2 State 업데이트 방식

- 노드는 **State 전체를 반환하지 않고, 바뀐 부분(딕셔너리)만 반환** → 랭그래프가 기존 State에 자동 병합
- Zustand의 `set((state) => ({ ... }))`과 동일한 부분 업데이트·병합 패턴. 다만 Zustand는 UI 렌더링용, 랭그래프 State는 노드 간 실행 흐름 전달용이라는 목적 차이가 있음

### 2.3 Reducer — 필드별 병합 규칙 (State 업데이트의 하위 개념)

기본 동작은 **덮어쓰기(overwrite)**. 그런데 메시지 히스토리처럼 "누적"이 필요한 필드도 있어서, 필드마다 병합 규칙을 지정할 수 있음.

```python
from typing import Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]
```

- `Annotated[타입, reducer함수]`: "이 필드는 덮어쓰지 말고 이 함수로 기존값·새값을 합쳐라"
- `operator.add`는 `+`를 함수 형태로 감싼 것뿐 (타입 힌트 자리엔 연산자가 아니라 callable이 필요해서 씀)

**`add_messages` — 대화 전용 reducer (operator.add의 특수화 버전)**

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`operator.add` 대비 추가 기능:
1. 같은 `id`의 메시지는 **덮어쓰기(수정)**, 새 id는 **추가**
2. 딕셔너리/튜플로 들어온 메시지를 자동으로 `HumanMessage`, `AIMessage` 객체로 변환
3. `RemoveMessage(id=...)`로 메시지 삭제 지원

**`MessagesState` — 위 reducer를 미리 세팅해둔 내장 템플릿**

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    extra_field: str  # 필요한 필드만 추가
```

즉 계층으로 보면: `operator.add`(범용 병합 함수) → `add_messages`(대화 전용으로 특수화) → `MessagesState`(그걸 미리 넣어둔 State 템플릿)

---

## 3. 핵심 개념 ②: 그래프 구조 (노드 · 엣지 · 흐름 제어)

### 3.1 기본 구성 요소

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)

def process_node(state: State):
    return {"result": f"처리됨: {state['input']}"}

builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

graph = builder.compile()
```

- `StateGraph(State)`: State 스키마를 지정해 그래프 빌더 생성
- `add_node(이름, 함수)`: 노드 등록 — 노드 = State를 받아 업데이트 딕셔너리를 반환하는 함수
- `add_edge(A, B)`: A → B **정적** 연결
- `compile()`: 설계도를 실행 가능한 그래프로 변환

### 3.2 흐름 제어 방식 (상위 개념) — 정적 → 동적 순으로 정리

| 방식 | 결정 시점 | 특징 |
|---|---|---|
| `add_edge` | 컴파일 시점 (고정) | A → B 무조건 연결 |
| `add_conditional_edges` | 실행 시점 | State를 보고 **다음 노드 하나**를 선택 |
| `Send` | 실행 시점 | **몇 개가 될지 모르는 노드 실행**을 동적으로 fan-out (→ 4장) |
| `Command` | 실행 시점 | State 업데이트 + 라우팅을 **한 반환값**으로 통합 처리 |

**3.2.1 조건부 분기 — `add_conditional_edges`**

```python
def route_after_process(state: State) -> str:
    return "retry" if state["result"] == "" else "finish"

builder.add_conditional_edges("process", route_after_process)
```
이 메커니즘 덕분에 랭그래프가 **분기와 루프**를 표현할 수 있음 (랭체인 체인과의 결정적 차이).

**3.2.2 통합형 — `Command` (State 업데이트 + 라우팅)**

```python
from langgraph.types import Command
from typing import Literal

def process_node(state: State) -> Command[Literal["next_node", "retry_node"]]:
    if state["input"] == "":
        return Command(update={"result": "실패"}, goto="retry_node")
    return Command(update={"result": "처리됨"}, goto="next_node")
```

- 기존 방식은 "State 업데이트(dict 반환)"와 "라우팅(`add_conditional_edges` + 별도 함수)"이 분리되어 있었음
- `Command`는 이 둘을 한 곳에 모음 → 로직이 밀접할수록, 특히 멀티 에이전트·서브그래프 구조에서 유용
- `graph=Command.PARENT` 옵션으로 서브그래프에서 부모 그래프로 빠져나가는 것도 가능

---

## 4. Map-Reduce 패턴과 `Send`

### 4.1 Map-Reduce란 (일반 개념)

원래 대용량 데이터 병렬 처리를 위한 프로그래밍 모델(구글 논문 기원, Hadoop 등으로 대중화)로, 두 단계로 구성됨:

- **Map 단계**: 큰 작업을 독립적인 작은 단위로 쪼개서, 각 단위를 **병렬로** 동시 처리
- **Reduce 단계**: Map에서 나온 결과들을 **하나로 집계·병합**

랭그래프에서는 이 두 단계가 각각 아래 개념에 대응됨:

| Map-Reduce 단계 | 랭그래프 대응 개념 |
|---|---|
| Map (분산 실행) | `Send` — 동적으로 노드를 여러 번, 각기 다른 State로 병렬 실행 |
| Reduce (결과 집계) | Reducer (예: `operator.add`) — 병렬 결과를 State의 한 필드로 병합 |

### 4.2 `Send` — Map 단계의 구현체

일반 `add_conditional_edges`는 "다음 노드 하나"만 고르지만, 컴파일 시점엔 **몇 번 실행될지조차 모르는** 경우(리스트 길이가 런타임에만 정해짐)엔 대응이 안 됨. 이걸 위한 것이 `Send`.

```python
from langgraph.types import Send

def route_to_workers(state: OverallState):
    return [Send("worker", {"item": item}) for item in state["items"]]

builder.add_conditional_edges("dispatcher", route_to_workers)
```

### 4.3 워커 State 분리 — Send 사용 시 원칙

```python
class OverallState(TypedDict):        # Reduce 대상: 전체 현황판
    items: list[str]
    results: Annotated[list[str], operator.add]

class WorkerState(TypedDict):          # Map 단위: 워커 한 명의 작업 지시서
    item: str
```

- `Send`의 두 번째 인자로 전체 State를 통째로 넘기면, 워커 입장에서 "내가 뭘 처리해야 하는지"가 불명확해짐
- 워커 전용 State(`WorkerState`)를 따로 정의해서 **딱 그 워커가 처리할 조각만** 넘기는 게 원칙
- 워커가 반환한 키 이름이 `OverallState`의 reducer 지정 필드와 같으면 자동으로 Reduce(병합)됨

즉 4장 전체를 한 문장으로: **Map-Reduce라는 일반 패턴을, 랭그래프에서는 `Send`(분산 Map) + `Reducer`(집계 Reduce) 조합으로 구현한다.**

---

## 5. 노드 함수 작성 시 참고사항

**docstring의 용도는 노드의 종류에 따라 다름**

| 상황 | docstring 용도 |
|---|---|
| 일반 노드 함수 (`add_node`로 등록) | 사람이 읽는 문서. 실행에 영향 없음 |
| `@tool`로 감싼 함수 (에이전트가 호출) | **LLM이 읽고 도구 선택에 사용** — tool description 자체가 됨. 실행에 직접 영향 |

에이전트가 도구를 직접 호출하는 패턴(`ToolNode` + `@tool`)에서는 docstring 품질이 도구 선택 정확도에 직결됨.

---

## 6. 전체 계층 요약

```
랭그래프 (오케스트레이션 프레임워크)
├─ State (공유 데이터)
│   ├─ 정의 방식: TypedDict / Pydantic
│   └─ 업데이트 규칙: Reducer
│       ├─ operator.add (범용 누적)
│       └─ add_messages (대화 전용) → MessagesState (내장 템플릿)
│
└─ 그래프 구조 (노드 + 흐름 제어)
    ├─ 정적 연결: add_edge
    ├─ 조건부 분기: add_conditional_edges
    ├─ 동적 병렬 실행: Send  ─┐
    │                         ├─ Map-Reduce 패턴 구현
    │   (Reducer가 결과 집계)─┘
    └─ 통합형(업데이트+라우팅): Command
```

## 7. 실무 적용 흐름

1. 기존 랭체인 RAG 체인이 있다면, 이를 그대로 노드 함수 안에 넣어 재사용
2. State는 `TypedDict`로 정의, 대화 이력이 필요하면 `MessagesState` 상속
3. 단순 순차 흐름은 `add_edge`, 분기가 필요하면 `add_conditional_edges`
4. 리스트 항목별 병렬 처리(map-reduce)가 필요하면 `Send` + 별도 `WorkerState`
5. State 업데이트와 라우팅이 얽혀 있거나 멀티 에이전트/서브그래프 구조라면 `Command` 고려
6. 에이전트가 도구를 직접 판단해서 호출해야 하면 `@tool` + docstring을 프롬프트 수준으로 신경 써서 작성

## 참고
- 공식 문서: https://docs.langchain.com/oss/python/langgraph/use-graph-api
- 검색 시점(2026년 9월) 기준 `StateGraph`, `add_node`, `add_conditional_edges`, `Send`, `Command` API는 안정적으로 유지되고 있음. Python 기준 State는 TypedDict, Pydantic, dataclass 모두 지원.
