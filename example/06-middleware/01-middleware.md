# LangChain/LangGraph 미들웨어 정리

> 이 문서 전반의 "실무 활용"은 **제조 데이터 기반 AX 에이전트**(설비 모니터링/이상 감지/현장 대응) 시나리오를 기준으로 적는다. 해커톤에서 만들 법한 에이전트를 하나 가정한다:
> **"공장 설비 센서 데이터를 조회하고, 이상 징후를 판단하고, 필요하면 정지/알림 조치를 취하는 에이전트"**

---

## 1. 미들웨어란

에이전트는 기본적으로 이 루프를 돈다.

```
모델 호출 → (도구 필요하면) 도구 실행 → 모델 호출 → ... → 종료
```

미들웨어는 이 루프의 **특정 지점에 코드를 끼워 넣는 장치**다. 로깅, PII 처리, 재시도, 모델 교체, 도구 필터링, 요약 같은 걸 에이전트 로직 본체를 건드리지 않고 "끼워 넣는" 방식으로 구현한다.

**제조 AX에서 왜 중요한가**: 현장 에이전트는 "그냥 답변 잘하는 챗봇"이 아니라 **설비를 실제로 멈추거나, 알림을 보내거나, MES/SCADA에 값을 쓰는** 행동을 할 수 있어야 해커톤 심사에서 임팩트가 있다. 그런데 그런 행동일수록 실수했을 때 리스크가 크다 — 미들웨어가 바로 "모델이 판단은 하되, 위험한 행동은 규칙으로 한 번 더 걸러내는" 안전판 역할을 한다.

---

## 2. 두 가지 훅 스타일

### 2.1 Node-style (before/after) — "스쳐 지나가는" 훅

```python
def before_agent(state, runtime): ...   # 에이전트 시작 전, 1회
def before_model(state, runtime): ...   # 모델 호출 전마다
def after_model(state, runtime): ...    # 모델 호출 후마다
def after_agent(state, runtime): ...    # 에이전트 종료 후, 1회
```

- 정해진 시점에 **한 번 실행되고 끝**나는 독립된 노드
- `state`, `runtime`만 받음 — **모델을 실제로 호출하는 함수 자체는 안 쥐어짐**
- 할 수 있는 것: state 미리 확인/수정, `jump_to`로 분기(모델 호출 자체를 스킵하는 것도 가능)
- **할 수 없는 것**: 모델 호출 실패 시 재시도, 응답을 받은 후 가공 — 애초에 모델을 부르는 주체가 아니라서 "호출 결과"라는 개념 자체가 성립 안 함

### 2.2 Wrap-style (wrap_model_call / wrap_tool_call) — "감싸서 직접 실행하는" 훅

```python
def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse:
    response = handler(request)   # handler = "실제로 LLM에 요청 보내는 함수" 그 자체
    return response
```

- `handler`가 곧 실제 모델 호출 행위. 이걸 **내가 직접 부르는 코드 안**에 있음
- 그래서 가능한 것: 재시도(`handler`를 여러 번), 스킵(`handler` 안 부름 = 캐싱), 응답 후처리(받은 `response` 수정), **모델 자체 교체**(`request.override(model=...)`)
- `request`(`ModelRequest`)는 이번 호출의 설정 전체(모델, 메시지, 도구, 시스템 프롬프트, 응답 포맷)를 담은 객체 — 이게 node-style엔 안 주어짐
- `wrap_tool_call`도 동일한 구조로 **도구 실행**을 감쌈 (도구 호출 재시도/스킵/결과 가공)

### 2.3 구분 기준 — 실무에서 쓰는 판단 한 줄

> **"모델(또는 도구) 호출 결과에 반응해야 하는 일"(재시도, 응답 가공, 모델/프롬프트 자체 교체)이 필요하면 wrap 계열. 그 외 단순 로깅/조건부 스킵은 before/after로 충분.**

**제조 AX 예**: "센서 API가 타임아웃 나면 재시도" → `wrap_tool_call`. "설비 ID가 등록된 목록에 없으면 아예 실행 안 함" → `before_model` + `jump_to`.

---

## 3. 실행 순서

```
1. before_model
2. wrap_model_call (그 안에서 handler() → 진짜 모델 호출)
3. after_model
```

미들웨어 여러 개(`middleware=[m1, m2, m3]`)일 때:

```
before 계열: 순서대로       m1 → m2 → m3
wrap 계열:   양파처럼 중첩    m1( m2( m3( 진짜 호출 ) ) )
after 계열:  역순           m3 → m2 → m1
```

> **실무 참고**: wrap 계열이 중첩되므로, **리스트에서 앞쪽(바깥)에 둔 미들웨어가 안쪽 전체를 통제**한다. 재시도/폴백 미들웨어를 앞에 두면 그 안의 다른 wrap 미들웨어와 진짜 호출까지 통째로 재시도 대상이 된다.

**해커톤 실전 팁**: 심사 당일 라이브 데모에서 모델 API가 순간적으로 불안정할 수 있다. 재시도(`ModelRetryMiddleware`)를 항상 맨 앞에 깔아두면 "데모 중 에러로 죽는" 최악의 시나리오를 크게 줄일 수 있다.

---

## 4. 클래스(`AgentMiddleware`) vs 데코레이터 — 언제 뭘 쓰나

| 상황 | 선택 |
|---|---|
| 훅 하나만, 상태 기억 필요 없음 | `@before_model`, `@wrap_model_call` 등 데코레이터 |
| 여러 훅을 한 세트로 묶고 싶음 | `AgentMiddleware` 상속 |
| 설정값을 받아서 재사용 가능한 컴포넌트로 만들고 싶음 (`__init__` 파라미터) | `AgentMiddleware` 상속 |

**해커톤 실무 팁**: 시간이 부족한 해커톤 특성상 **처음엔 데코레이터로 빠르게 프로토타입**을 만들고, "이 설비 라인에도 같은 로직 필요하네" 싶어지는 순간(예: 라인 A, B, C마다 같은 이상 판단 로직을 다르게 파라미터화해야 할 때) 클래스로 승격하는 순서가 현실적이다.

```python
class EquipmentLineFilterMiddleware(AgentMiddleware):
    def __init__(self, line_id: str, max_threshold: float):
        self.line_id = line_id            # 설정값 — 라인마다 다른 인스턴스 생성
        self.max_threshold = max_threshold

    def before_model(self, state, runtime):
        ...

# 라인별로 같은 클래스, 다른 설정으로 재사용
middleware_line_a = EquipmentLineFilterMiddleware(line_id="A", max_threshold=85.0)
middleware_line_b = EquipmentLineFilterMiddleware(line_id="B", max_threshold=90.0)
```

### ⚠️ 실무 함정: `self` vs `state`

```python
class CallLimitMiddleware(AgentMiddleware):
    def __init__(self, limit=10):
        self.count = 0   # ❌ 위험한 패턴
        self.limit = limit

    def before_model(self, state, runtime):
        self.count += 1
        if self.count > self.limit:
            return {"jump_to": "end"}
```

`self.count`는 **agent 객체가 살아있는 동안** 유지된다. 여러 세션(여러 작업자가 각자 다른 thread로 접속)이 **같은 agent 인스턴스를 공유**하면, 작업자 A가 8번 조회하고 작업자 B가 이어서 쓰면 카운트가 섞여버리는 버그가 생긴다. **해커톤 데모에서 "왜 갑자기 막히지?" 같은 당황스러운 버그의 흔한 원인**이니 조심.

- `self.xxx` → 라인 ID, 임계값 같은 **설정값**
- `state["xxx"]` → 조회 횟수, 이번 세션 누적 알림 수 같은 **세션별 누적값**

> 참고: 아래 §5의 `ModelCallLimitMiddleware`/`ToolCallLimitMiddleware`가 이 문제를 미리 해결해서 제공한다.

---

## 5. 내장(prebuilt) 미들웨어

### 5.1 에러 대응 계열 — 실무 필수

| 실패 유형 | 누가 처리하나 | 대응 | 미들웨어 |
|---|---|---|---|
| 일시적 오류 (네트워크, 레이트리밋) | 시스템(자동) | 지수 백오프로 재시도 | `ModelRetryMiddleware`, `ToolRetryMiddleware` |
| LLM이 복구 가능한 오류 (도구 실패, 파싱 오류) | LLM | 에러를 `ToolMessage`로 변환해서 모델이 스스로 조정하게 함 | `ToolErrorMiddleware` |
| 과도한 호출 (무한 루프) | 시스템(자동) | 런/스레드당 호출 수 상한 | `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` |
| 예상 못 한 오류 | 개발자 | 그냥 예외 전파시킴 | 미들웨어 없이 그대로 두기 |

#### `ModelRetryMiddleware` / `ToolRetryMiddleware`

```python
ToolRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    tools=["read_sensor_data"],   # MES/SCADA 조회 도구처럼 네트워크 의존적인 것만 골라서
    retry_on=(ConnectionError, TimeoutError),
    on_failure="continue",        # 재시도 다 실패해도 에이전트가 죽지 않고 계속 진행
)
```

**제조 AX 실무**: 현장 설비의 PLC/SCADA 연동은 사내망 상태에 따라 순간적으로 끊기는 일이 흔하다. `read_sensor_data`, `query_mes` 같은 조회성 도구에 재시도를 기본으로 걸어두는 게 안정성의 8할을 차지한다.

#### `ModelFallbackMiddleware`

```python
ModelFallbackMiddleware("gpt-5.4-mini", "claude-3-5-sonnet-20241022")
```

**해커톤 실무**: 심사 당일 특정 모델 API가 장애나면 에이전트 전체가 멈춘다. 폴백을 걸어두면 "발표 도중 API 하나가 막혀도 데모가 계속 돈다" — 임팩트 대비 설정 비용이 가장 낮은 안전장치.

#### `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware`

```python
ToolCallLimitMiddleware(tool_name="stop_equipment", run_limit=1, exit_behavior="error")
```

**제조 AX 실무**: 설비 정지처럼 **한 번의 판단 사이클에 절대 여러 번 실행되면 안 되는 도구**에 `run_limit=1`을 걸어두면, 모델이 루프 중 실수로 같은 정지 명령을 반복 호출하는 사고를 원천 차단할 수 있다. 이건 "잘 만들었다"를 보여주는 디테일이라 심사에서도 잘 드러난다.

### 5.2 컨텍스트 관리 계열

#### `SummarizationMiddleware`

```python
SummarizationMiddleware(
    model="openai:gpt-5.4-mini",
    trigger=('tokens', 4000),
    keep=('messages', 20),
)
```

**제조 AX 실무**: 설비를 하루 종일 모니터링하는 시나리오라면, 교대 근무 내내 센서 조회 로그가 쌓여 대화가 매우 길어진다. 다만 요약 과정에서 **정확한 수치(온도 82.3℃, 압력 3.1bar 같은 임계값 근처 데이터)가 뭉개질 위험**이 있으므로, 안전 관련 수치는 요약 대상에서 빼고 `state`의 별도 필드(예: `critical_readings`)에 원본 그대로 보관하는 설계를 권장.

#### `ContextEditingMiddleware`

```python
ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=100000, keep=3)])
```

**제조 AX 실무**: 센서 데이터 조회 도구가 매번 수백 행짜리 시계열을 반환하는 구조라면, 요약보다 이쪽이 적합 — 오래된 조회 결과는 통째로 비우고 최근 3건만 유지.

### 5.3 보안/거버넌스 계열

#### `PIIMiddleware`

```python
middleware=[
    PIIMiddleware("employee_id", detector=r"EMP-\d{6}", strategy="mask", apply_to_output=True),
    PIIMiddleware("phone_number", detector=r"010-\d{4}-\d{4}", strategy="redact", apply_to_input=True),
]
```

**제조 AX 실무**: 현장 데이터에는 작업자 사번, 연락처 같은 개인정보가 로그나 보고서에 섞여 들어가기 쉽다. 특히 **에이전트가 만든 보고서를 외부(협력사, 심사위원 데모 화면)에 노출할 가능성**이 있다면 `apply_to_output=True`로 출력 단계도 반드시 걸어야 한다.

#### `HumanInTheLoopMiddleware`

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "stop_equipment": True,      # 설비 정지 — 반드시 사람 승인
        "send_alert": False,         # 알림 발송 — 자동 허용
        "read_sensor_data": False,   # 조회 — 자동 허용
    }
)
```

**제조 AX 실무 — 이게 해커톤에서 가장 임팩트 있는 포인트 중 하나**: "AI가 알아서 설비를 껐다"는 위험하고 신뢰가 안 가는 데모지만, "AI가 이상을 감지하고 정지를 제안하면 관리자가 승인 버튼 하나로 확정"하는 구조는 **실제 현장에서 채택 가능한 설계**로 보인다. 심사에서 "왜 이렇게 설계했나"를 물으면 이 지점을 근거로 답변하기 좋다.

### 5.4 그 외

| 미들웨어 | 용도 |
|---|---|
| `TodoListMiddleware` | 여러 설비를 순회 점검하는 멀티스텝 작업 추적 |
| `LLMToolEmulator` | 실제 MES 연동 전, 가짜 응답으로 에이전트 로직만 먼저 테스트 — 해커톤 초반 API 연동 안 됐을 때 유용 |
| `FilesystemMiddleware` (Deep Agents) | 긴 점검 리포트, 매뉴얼 PDF 등을 파일로 다루면서 컨텍스트 절약 |

---

## 6. 실무 활용 — 제조 AX 에이전트에 조합해보기

해커톤에서 설비 이상 감지 에이전트를 만든다면, 미들웨어를 이렇게 계층으로 쌓는 게 현실적인 설계다.

```python
agent = create_agent(
    model=model,
    tools=[read_sensor_data, query_mes, stop_equipment, send_alert],
    middleware=[
        # 1. 안정성 — 현장 네트워크 불안정 대응
        ToolRetryMiddleware(tools=["read_sensor_data", "query_mes"], max_retries=3),
        ModelFallbackMiddleware("gpt-5.4-mini"),

        # 2. 안전장치 — 위험한 행동 재실행 방지
        ToolCallLimitMiddleware(tool_name="stop_equipment", run_limit=1, exit_behavior="error"),

        # 3. 개인정보 보호 — 작업자 정보 노출 방지
        PIIMiddleware("employee_id", detector=r"EMP-\d{6}", strategy="mask", apply_to_output=True),

        # 4. 사람 승인 — 설비 정지는 반드시 확인
        HumanInTheLoopMiddleware(interrupt_on={"stop_equipment": True, "send_alert": False}),

        # 5. 컨텍스트 관리 — 하루치 모니터링 로그 압축
        SummarizationMiddleware(model="gpt-5.4-mini", trigger=("tokens", 4000), keep=("messages", 20)),
    ],
    checkpointer=InMemorySaver(),
)
```

**심사에서 설명하기 좋은 포인트**: "모델이 똑똑해서 잘 판단한다"보다 "**모델이 틀려도 시스템 레벨에서 안전하게 동작하도록 설계했다**"는 게 실무형 AX 엔지니어의 관점으로 보인다. 위 5단 구조가 그 근거를 코드로 보여준다.
