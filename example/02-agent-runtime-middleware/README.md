# 에이전트: 모델 · 프롬프트 · 미들웨어 · 런타임 정리

> `langchain` v1 기준 (`create_react_agent` → `create_agent`로 이름과 구조가 바뀐 최신 문법). 구버전 `create_react_agent`를 쓰던 코드는 `prompt` 파라미터, pre/post-model hook 같은 게 있었는데, 지금은 전부 **미들웨어**로 통합됨.

## 0. 전체 그림 — 이 개념들이 어디에 위치하는가

```
create_agent(model, tools, middleware, context_schema, ...)
│
├─ Model        : 실제로 텍스트를 생성하는 LLM
├─ Prompt       : 모델에게 주는 지시문 (시스템 프롬프트)
├─ Middleware   : 에이전트 실행 흐름 중간에 개입하는 후크(hook)들
│   ├─ 이 미들웨어들이 Model도 바꾸고(wrap_model_call)
│   ├─ Prompt도 만들고(dynamic_prompt)
│   └─ 흐름 자체(jump_to)도 제어함
└─ Runtime      : 위 모든 것이 실행 중에 참조하는 "이번 호출의 정보 꾸러미"
    ├─ Context (설정값)
    ├─ Store (장기 메모리)
    └─ Stream Writer (실시간 송출)
```

즉 **Model·Prompt는 "무엇을 실행할지"**, **Middleware는 "언제·어떻게 개입할지"**, **Runtime은 "그 개입에 필요한 정보를 어디서 가져올지"**를 담당한다고 보면 전체 구조가 잡힘.

---

## 1. Model

### 1.1 기본 사용

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5.4")

agent = create_agent(
    model=model,       # 객체로 넣어도 되고
    tools=[...],
)
# 또는
agent = create_agent(
    model="gpt-5.4",   # 문자열로 바로 넣어도 create_agent 내부에서 init_chat_model 호출해줌
    tools=[...],
)
```

### 1.2 동적 모델 선택 — Model의 하위 개념: `wrap_model_call` 미들웨어

Model을 고정으로 하나만 쓰지 않고, **모델 호출 시점마다 조건에 따라 바꾸고** 싶으면 미들웨어(4장)의 `wrap_model_call`을 씀.

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = init_chat_model("gpt-5.4-mini")
advanced_model = init_chat_model("gpt-5.4")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """대화 복잡도에 따라 모델 선택"""
    message_count = len(request.state["messages"])
    request.model = advanced_model if message_count > 10 else basic_model
    return handler(request)  # 실제 모델 호출 실행

agent = create_agent(model=basic_model, tools=[], middleware=[dynamic_model_selection])
```

- `handler(request)`를 호출해야 실제로 LLM API가 불림 — 안 부르면 모델 호출 자체를 건너뛰는(short-circuit) 것도 가능 (캐싱, 사전 차단 등에 활용)

**실무 활용**: 짧은 질문엔 저렴한 모델, 긴 대화·복잡한 판단엔 고급 모델로 자동 전환해 비용을 절감하는 패턴으로 흔히 씀.

---

## 2. Prompt

### 2.1 정적 프롬프트

```python
agent = create_agent(
    model=model,
    tools=[...],
    system_prompt="You are a helpful assistant",   # 구버전은 prompt= 였는데 v1부터 system_prompt=
)
```
- 실행 내내 고정된 문자열 — 가장 단순한 형태

### 2.2 동적 프롬프트 — Prompt의 하위 개념: `dynamic_prompt` 미들웨어

Context(6장)에 따라 프롬프트를 매번 다르게 만들고 싶으면 `dynamic_prompt`를 씀.

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_name: str
    language: str

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context.user_name
    language = request.runtime.context.language
    if language == "Korean":
        return f"You are a helpful assistant. Address the user as '{user_name}'. Always respond in Korean."
    return f"You are a helpful assistant. Address the user as '{user_name}'. Always respond in English."

agent = create_agent(model=model, tools=[...], middleware=[dynamic_system_prompt], context_schema=Context)
```

- State(`request.state["messages"]`)를 기반으로도 동적 프롬프트 가능 (예: 대화가 10턴 넘으면 "간결하게 답하라" 추가) — Context는 "누가 요청했는지", State는 "지금 대화가 어떤 상태인지" 기준으로 분기할 때 씀

**실무 활용**: 사용자 역할(관리자/일반), 언어, 답변 형식(SNS/기사체) 등에 따라 시스템 프롬프트를 개인화할 때 표준적으로 쓰는 패턴.

---

## 3. Middleware — 에이전트 실행에 개입하는 후크

### 3.1 두 가지 스타일

```
Middleware
├─ Node 스타일 — 그래프의 "특정 시점"에 한 번 끼어듦
│   ├─ before_agent  : 에이전트 시작 전 (실행당 1회)
│   ├─ before_model   : 매 모델 호출 전
│   ├─ after_model    : 매 모델 호출 후
│   └─ after_agent    : 에이전트 종료 후 (실행당 1회)
│
└─ Wrap 스타일 — 호출 "자체를 감싸서" 가로챔
    ├─ wrap_model_call : 모델 호출을 감쌈 (1장에서 본 것)
    └─ wrap_tool_call   : 도구 호출을 감쌈
```

| | 매개변수 | 반환 | 특징 |
|---|---|---|---|
| Node 스타일 | `state, runtime` | `dict \| None` | 그래프 노드처럼 State를 직접 받고, 값을 반환하면 State에 병합 |
| Wrap 스타일 | `request: ModelRequest, handler` (또는 `dynamic_prompt`는 `handler` 없이 `request`만) | `ModelResponse` / `str` 등 | `handler`를 호출해야 실제 호출이 일어남 → 재시도·캐싱·차단(short-circuit) 가능 |

### 3.2 Node 스타일 예시 — 로깅

```python
from langchain.agents.middleware import before_model, after_model
from langgraph.runtime import Runtime

@before_model
def log_before_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"[Before Model] User: {runtime.context.user_name}")
    print(f"[Before Model] Messages count: {len(state['messages'])}")
    return None

@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"[After Model] Response generated for session: {runtime.context.session_id}")
    return None
```
- 도구를 쓰는 에이전트는 "모델 → 도구 → 모델" 루프를 타기 때문에, `before_model`/`after_model`은 **한 번의 사용자 질문에도 여러 번** 실행될 수 있음

### 3.3 Node 스타일 + 라우팅 제어 — `jump_to` / `can_jump_to`

Middleware도 결국 그래프 위의 노드이기 때문에, State 업데이트뿐 아니라 **"다음에 어디로 갈지"**까지 결정할 수 있음.

```python
from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime

@before_agent(can_jump_to=["end"])
def check_permissions(state: AgentState, runtime: Runtime[AuthContext]) -> dict[str, Any] | None:
    """Check if user has required permissions"""
    permissions = runtime.context.permissions
    if state["messages"]:
        content = state["messages"][0].content.lower()
        if ("delete" in content or "remove" in content) and "admin" not in permissions:
            return {
                "messages": [{"role": "assistant", "content": "You don't have permission to perform this action."}],
                "jump_to": "end",
            }
    return None
```

- `can_jump_to=["end"]`: 데코레이터에 붙이는 **선언** — "이 미들웨어는 `end` 노드로 점프할 수 있다"고 그래프 컴파일 시점에 미리 알려줌. 유효한 값은 `"tools"`, `"model"`, `"end"`
- `"jump_to": "end"`: 반환 딕셔너리 안에 담긴 **예약된 키(reserved key)** — 랭그래프가 이 키를 발견하면 일반 State 필드 병합이 아니라 **라우팅 명령**으로 해석해서, 지정한 노드로 즉시 이동시킴
- 이는 랭그래프 기초에서 배운 `Command(goto=...)`와 본질적으로 같은 메커니즘. `Command`는 `update`/`goto`가 별도 필드로 분리된 형태, 미들웨어의 `jump_to`는 하나의 dict 안에 특수 키로 섞여 들어가는 형태라는 표현 방식 차이만 있음

**실무 활용**: 가드레일(guardrail) 구현의 핵심 패턴 — 권한 없는 요청, 금칙어, PII 등을 **LLM 호출 전에** `before_agent`/`before_model` 단계에서 차단해서 불필요한 API 비용을 막음.

### 3.4 미들웨어의 docstring — 누구를 위한 것인가

```python
@before_model
def log_before_model(state, runtime) -> dict | None:
    """모델 호출 전 로깅"""   # ← LLM에게 안 감
```
- `@tool`의 docstring은 **LLM이 도구 선택 판단에 쓰는 프롬프트 일부**
- 미들웨어(`before_model`, `after_model`, `dynamic_prompt` 등)는 LLM이 "쓸지 말지" 판단하는 게 아니라 **시스템이 정해진 시점에 무조건 실행**하므로, docstring은 순수하게 **사람이 읽는 문서** — 실행에 영향 없음

---

## 4. Runtime — Context / Store / Stream Writer

### 4.1 Runtime이란

`create_agent`는 내부적으로 랭그래프 런타임 위에서 동작하고, 이 런타임은 도구·미들웨어가 공통으로 참조할 수 있는 **"이번 호출 한정 정보 꾸러미"**를 제공함.

```
Runtime (agent.invoke() 한 번 호출될 때마다 새로 생성, 끝나면 소멸)
├─ Context      : 정적 설정값 (사용자 ID, 권한, DB 연결, 요청별 옵션 등)
├─ Store        : 장기 메모리 (Runtime 자체는 휘발되어도 저장된 데이터는 영속)
└─ Stream Writer : 실행 도중 실시간으로 정보를 밖으로 흘려보내는 통로
```

- 비유: 매 호출마다 새로 발급되는 "방문증"(Context/Stream Writer)과, 방문증으로 들어가는 "서고"(Store) — 방문증은 나가면 폐기되지만 서고 안의 책은 남아있음

### 4.2 접근 경로는 "누가 접근하느냐"에 따라 다름 (같은 Runtime, 다른 통로)

| 접근 주체 | 접근 방법 |
|---|---|
| 도구 (`@tool`) | `runtime: ToolRuntime[ContextType]` 매개변수 (state, tool_call_id, config도 추가로 포함) |
| Wrap 스타일 미들웨어 (`dynamic_prompt`, `wrap_model_call`) | `request.runtime` (`ModelRequest` 안에 포함) |
| Node 스타일 미들웨어 (`before_model` 등) | `runtime: Runtime[ContextType]` 별도 매개변수 |

`ToolRuntime`과 `Runtime`은 이름은 비슷하지만 다른 타입 — `ToolRuntime`은 도구 전용으로 `state`, `tool_call_id`, `config` 등 도구에만 필요한 정보가 추가로 들어있음.

### 4.3 Context — 정적 설정값

```python
@dataclass
class Context:
    user_id: str

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Get information about the current user."""
    return f"User ID: {runtime.context.user_id}"

agent = create_agent(model=model, tools=[get_user_info], context_schema=Context)
agent.invoke({"messages": [...]}, context=Context(user_id="user_123"))
```

- `context_schema`: "이 에이전트는 이런 모양의 Context를 받는다"는 틀 선언
- `context=Context(...)`: 실행 시점에 실제 값 주입
- **`ToolRuntime`은 LLM에게 노출되지 않음** — LLM은 도구가 `runtime`이라는 매개변수를 받는지조차 모르고, 시스템이 자동으로 채워줌

#### Context 활용 패턴 ① — 외부 자원 접근 수단 전달

```python
@dataclass
class DatabaseContext:
    db_pool: Any   # 연결 객체가 아니라 "커넥션 풀"을 담는 게 실무 표준
    user_id: str

@tool
def query_database(sql: str, runtime: ToolRuntime[DatabaseContext]) -> str:
    """Execute SQL query on the database."""
    pool_ref = runtime.context.db_pool
    conn = pool_ref.getconn()      # 풀에서 연결 빌림
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        return str(cursor.fetchall())
    finally:
        pool_ref.putconn(conn)     # 반드시 반납
```
- Context에 넣는 건 **데이터 자체가 아니라 접근 수단**(연결/풀) — 실제 조회는 도구 실행 시점에 도구가 직접 함
- 단일 연결 객체를 직접 넘기면 동시 요청 시 race condition 위험 → 실무에서는 거의 항상 **커넥션 풀**(`psycopg2.pool.ThreadedConnectionPool` 등)을 Context에 담아 `getconn()`/`putconn()`으로 빌리고 반납

#### Context 활용 패턴 ② — 요청별 실행 설정값

```python
@dataclass
class RequestContext:
    user_id: str
    verbose: bool
    timeout: int
    max_tokens: int

@tool
def process_request(query: str, runtime: ToolRuntime[RequestContext]) -> str:
    """Process request with custom settings."""
    ctx = runtime.context
    if ctx.verbose:
        print(f"Processing with timeout: {ctx.timeout}s")
    return f"Processed: {query}"
```
- 사용자 등급별로 timeout·max_tokens 등을 호출마다 다르게 넘겨 실행 정책을 제어할 때 씀

#### Context 활용 패턴 ③ — 권한/인증 검사 (3.3절 `jump_to`와 결합)

가드레일 구현 시 Context에 `permissions` 같은 인증 정보를 담아 `before_agent`에서 검사하는 패턴 (3.3절 참고)

### 4.4 Store — 장기 메모리

```python
from langgraph.store.memory import InMemoryStore

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email writing style preferences from the store."""
    user_id = runtime.context.user_id
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            return memory.value["preferences"]
    return "기본 설정"

@tool
def save_user_preference(preference: str, runtime: ToolRuntime[Context]) -> str:
    """Save user's preference settings to the store."""
    if runtime.store:
        runtime.store.put(("users",), runtime.context.user_id, {"preferences": preference})
        return "저장됨"
    return "Store를 사용할 수 없습니다."

store = InMemoryStore()
agent = create_agent(model=model, tools=[...], context_schema=Context, store=store)
```
- 키 구조: `(네임스페이스,) + 키` — 예: `("users",), "user_123"`, 파일 시스템의 `/users/user_123` 경로와 비슷한 느낌
- `get()`으로 조회, `put()`으로 저장 — **대화·세션이 끝나도 값이 남음** (Context와의 결정적 차이)
- `InMemoryStore`는 가장 단순한 구현체(프로세스 종료 시 소멸), 실무에선 DB 기반 Store로 교체

**실무 활용**: 사용자 선호도, 대화 요약본, 장기 기억이 필요한 개인화 기능에 표준적으로 사용.

### 4.5 Stream Writer — 실시간 진행 상황 송출

```python
@tool
def process_large_dataset(num_items: int, runtime: ToolRuntime) -> str:
    """Process a large dataset and report progress."""
    writer = runtime.stream_writer
    for i in range(0, num_items, 10):
        writer({"stage": "processing", "progress": min(i + 10, num_items), "total": num_items})
    writer({"stage": "completed", "total": num_items})
    return f"Successfully processed {num_items} items!"

for chunk in agent.stream({"messages": [...]}, stream_mode="custom"):
    if "progress" in chunk:
        print(f"Progress: {chunk['progress']/chunk['total']*100:.0f}%")
```
- 도구의 **반환값**(최종 결과)과 별개로, 실행 도중 임의 시점에 데이터를 즉시 내보낼 수 있는 채널
- `stream_mode="custom"`으로 받는 쪽에서 수신

**실무 활용**: 대용량 처리, 여러 단계 파이프라인처럼 오래 걸리는 도구에서 프론트엔드에 진행률 표시줄을 그릴 때 사용.

---

## 5. 전체 계층 요약

```
create_agent
│
├─ Model ─────────────────── 실제 LLM
│   └─ wrap_model_call 미들웨어로 동적 선택 가능
│
├─ Prompt ────────────────── 시스템 프롬프트
│   ├─ 정적: system_prompt=""
│   └─ 동적: dynamic_prompt 미들웨어 (State/Context 기반)
│
├─ Middleware ────────────── 실행 흐름 개입 지점
│   ├─ Node 스타일 (특정 시점 1회)
│   │   ├─ before_agent / after_agent (실행당 1회)
│   │   ├─ before_model / after_model (모델 호출마다)
│   │   └─ jump_to로 그래프 노드 간 라우팅 제어 가능 (can_jump_to로 사전 선언)
│   └─ Wrap 스타일 (호출을 감쌈, handler로 실제 호출 여부/횟수 제어)
│       ├─ wrap_model_call
│       └─ wrap_tool_call
│
└─ Runtime ───────────────── 위 모든 곳에서 참조하는 실행 정보
    ├─ Context   : 정적 설정 (인증, DB 연결/풀, 요청별 옵션)
    ├─ Store     : 장기 메모리 (get/put, 세션을 넘어 영속)
    └─ Stream Writer : 실시간 진행 상황 송출
    (접근 경로: 도구=ToolRuntime, wrap 미들웨어=request.runtime, node 미들웨어=runtime 매개변수)
```

## 6. 실무 적용 흐름

1. 기본 에이전트는 `model` + `tools`만으로 구성, 정적 `system_prompt`로 시작
2. 사용자별 개인화가 필요해지면 `context_schema` 정의 → `dynamic_prompt`로 프롬프트 개인화
3. DB, 외부 API 같은 자원 접근이 필요하면 Context에 **연결이 아니라 커넥션 풀**을 담아 도구에서 빌려 쓰기
4. 사용자 선호도·기억을 남겨야 하면 Store 연결
5. 비용 절감이나 보안이 필요하면:
   - 모델 비용 절감 → `wrap_model_call`로 조건별 모델 전환
   - 권한/금칙어 차단 → `before_agent` + `jump_to`로 LLM 호출 전 가드레일
   - 재시도/캐싱 → `wrap_model_call`, `wrap_tool_call`
6. 오래 걸리는 작업은 Stream Writer로 진행률 노출
7. 전 과정에서 로깅이 필요하면 `before_model`/`after_model`에 가볍게 붙임 (State/Context 값만 읽고 `return None`)

## 참고
- 공식 문서: https://docs.langchain.com/oss/python/langchain/middleware/custom , https://docs.langchain.com/oss/python/langchain/runtime
- v1 마이그레이션 노트: `create_react_agent` → `create_agent`, `prompt` → `system_prompt`, pre/post-model hook → 미들웨어로 통합, Runtime context는 `config["configurable"]` 대신 `context` 인자로 전달하는 방식으로 변경됨 (검색 시점 2026년 9월 기준 최신)
