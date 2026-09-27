# LangChain/LangGraph 컨텍스트 엔지니어링 정리

> 이 문서의 "실무 활용"은 `01-middleware.md`와 같은 시나리오를 기준으로 한다: **공장 설비 센서 데이터를 조회하고 이상 징후를 판단하는 에이전트.** 현장 작업자와 관리자가 같은 에이전트를 역할별로 다르게 쓴다고 가정한다.

## 0. 왜 배우는가

> 에이전트가 실패하는 두 가지 이유: ① 모델 능력 부족, ② **"올바른" 컨텍스트가 전달되지 않음**. 대부분은 ②가 원인이다.

컨텍스트 엔지니어링 = **매 순간 LLM/도구에게 정확히 필요한 정보만, 올바른 형식으로 전달하도록 설계하는 것.** 미들웨어는 이걸 구현하는 도구다.

**제조 AX에서 특히 중요한 이유**: 현장 에이전트는 "설비마다, 역할마다, 시간대마다" 필요한 정보와 허용 행동이 다르다. 같은 질문("이 설비 상태 어때?")도 현장 작업자에게는 "지금 조치해야 하는지"가 중요하고, 관리자에게는 "이번 주 추세"가 중요할 수 있다. 이 차이를 프롬프트에 전부 하드코딩하는 대신, 컨텍스트 엔지니어링으로 **매 호출마다 동적으로 조립**하는 게 이 챕터의 핵심 활용처다.

---

## 1. 용어가 헷갈리는 이유 — 먼저 구조부터

같은 대상이 여러 이름으로 불려서 헷갈리기 쉽다. 4개 축으로 분해하면 정리된다.

### 축 1: 데이터가 "어디" 저장되나 (데이터 소스)

| 소스 | 다른 이름 | 범위 | 예시 | 제조 AX 예 |
|---|---|---|---|---|
| **Runtime Context** | 정적 구성 | 대화(호출) 1회 한정, 안 바뀜 | user ID, API 키, DB 연결, 권한 | 작업자 role(operator/manager), 소속 라인 ID |
| **State** | 단기 메모리 | 대화 1회 한정, 계속 바뀜 | 현재 메시지, 업로드 파일, 인증 상태 | 이번 세션에서 조회한 센서 값, 판단 중인 알람 |
| **Store** | 장기 메모리 | **대화를 넘나듦** (DB) | 유저 선호도, 추출된 인사이트, 히스토리 | 설비별 과거 고장 이력, 정비 노트 |

### 축 2: State 안 메시지 종류 (동의어 정리)

| 객체 | 동의어 |
|---|---|
| `SystemMessage` | = 시스템 프롬프트 |
| `HumanMessage` | = 유저 프롬프트 = 사용자 질문 = 쿼리 |
| `AIMessage` | = 모델 응답 |
| `ToolMessage` | = 도구 실행 결과 = 툴 메시지 |

### 축 3: "언제" 개입하나 (컨텍스트 종류) — 실무 3분류

| 종류 | 지속성 | 언제 | 뭘 다루나 |
|---|---|---|---|
| **Model Context** | Transient(일시적) | 모델 호출 순간 | 이번 호출에 들어갈 지시사항·메시지·도구·모델·응답 형식 |
| **Tool Context** | Persistent(영구) | 도구 실행 순간 | 도구가 state/store/runtime을 읽고 쓰는 것 |
| **Life-cycle Context** | Persistent(영구) | 모델↔도구 호출 사이 | 요약, 가드레일, 로깅 등 교차 관심사 |

### 축 4: 코드에서 누가 손에 쥐나 (전달 객체)

| 객체 | 누가 받나 | 담긴 것 |
|---|---|---|
| `Runtime` | 미들웨어(`before_model` 등) | `.context` `.store` `.stream_writer` |
| `ToolRuntime` | 도구(`@tool`) 함수 | `.state` `.store` `.context` |
| `ModelRequest` | `wrap_model_call`류 | `.messages` `.tools` `.model` `.response_format` 등 |

`Runtime` ⊃ `Context` (런타임이라는 큰 상자 안에 Context가 한 칸으로 들어있음).

---

## 2. Model Context — 실전 문법 + 제조 AX 적용

### 2.1 System Prompt — `@dynamic_prompt`

```python
@dataclass
class PlantContext:
    user_role: str      # "operator" | "manager" | "engineer"
    line_id: str
    shift: str           # "day" | "night"

@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    role = request.runtime.context.user_role
    line = request.runtime.context.line_id

    base = f"당신은 {line} 라인의 설비 모니터링 어시스턴트입니다."

    if role == "operator":
        base += "\n현장 작업자에게 응답합니다. 즉시 취해야 할 조치 위주로, 짧고 명확하게 답하세요."
    elif role == "manager":
        base += "\n관리자에게 응답합니다. 추세와 원인 분석을 포함해 답하세요."
    elif role == "engineer":
        base += "\n엔지니어에게 응답합니다. 센서 원시값과 임계값 근거를 함께 제시하세요."

    return base
```

**실무 포인트**: 같은 에이전트, 같은 도구를 쓰지만 **역할에 따라 응답 스타일이 완전히 달라진다** — 이걸 프롬프트를 역할별로 여러 벌 만드는 대신 **하나의 동적 프롬프트로 관리**하는 게 컨텍스트 엔지니어링의 실전 활용. 해커톤 시연에서 "operator 계정으로 물으면 짧게, manager 계정으로 물으면 분석까지" 보여주면 설계 의도가 명확히 전달된다.

### 2.2 Messages — `@wrap_model_call`로 주입/가공

```python
@wrap_model_call
def inject_recent_alerts(request: ModelRequest, handler) -> ModelResponse:
    """이번 세션에서 이미 감지된 미해결 알람을 매 호출마다 상기시킴"""
    pending_alerts = request.state.get("pending_alerts", [])
    if pending_alerts:
        alert_context = "현재 미해결 알람:\n" + "\n".join(
            f"- {a['equipment_id']}: {a['message']} ({a['detected_at']})" for a in pending_alerts
        )
        messages = [*request.messages, {"role": "user", "content": alert_context}]
        request = request.override(messages=messages)
    return handler(request)
```

**실무 포인트**: 모델이 대화 초반에 감지한 이상 징후를 나중 턴에서 "까먹고" 새 질문에만 답하는 걸 막는 패턴. 멀티턴 모니터링 세션에서 자주 필요.

### 2.3 Tools — 동적 필터링 (RBAC)

```python
@wrap_model_call
def role_based_tools(request: ModelRequest, handler) -> ModelResponse:
    role = request.runtime.context.user_role
    if role == "operator":
        # 현장 작업자는 조회와 알림만, 설비 정지는 관리자 승인 흐름으로만
        tools = [t for t in request.tools if t.name in ("read_sensor_data", "send_alert")]
        request = request.override(tools=tools)
    elif role == "manager":
        pass  # 전체 도구 사용 가능
    return handler(request)
```

> 핵심: **모델에게 도구를 아예 안 보여주면**, 모델이 실수로도 그 도구를 호출할 수 없다 — 프롬프트로 "이 도구 쓰지 마"라고 지시하는 것보다 훨씬 확실한 차단 방법. `stop_equipment` 같은 위험한 도구는 이 단계에서 권한 없는 역할한테 아예 안 보이게 하는 게 `HumanInTheLoopMiddleware`(가드레일 문서 참고)와 이중으로 겹치는 안전장치가 된다.

### 2.4 Model — 동적 모델 선택

```python
@wrap_model_call
def urgency_based_model(request: ModelRequest, handler) -> ModelResponse:
    """긴급 알람 상황이면 더 신중한(느리더라도 정확한) 모델로 전환"""
    has_critical = request.state.get("has_critical_alert", False)
    model = careful_model if has_critical else fast_model
    request = request.override(model=model)
    return handler(request)
```

**실무 활용**: 평시 조회는 빠른/저렴한 모델, 이상 징후 판단처럼 오판 비용이 큰 순간엔 더 강력한 모델로 자동 전환 — 비용과 신뢰성을 동시에 관리하는 실전 패턴.

### 2.5 Response Format — 구조화 출력

```python
class AlertReport(BaseModel):
    equipment_id: str
    severity: str = Field(description="'low', 'medium', 'high', 'critical'")
    summary: str
    recommended_action: str

@wrap_model_call
def structured_alert_output(request: ModelRequest, handler) -> ModelResponse:
    if request.state.get("has_critical_alert"):
        request = request.override(response_format=AlertReport)
    return handler(request)
```

**실무 활용**: 이상 감지 시 자유 텍스트가 아니라 **구조화된 알람 리포트**로 강제하면, 이걸 그대로 알림 시스템(Slack/문자)이나 대시보드에 연결하기 쉽다 — 해커톤에서 "에이전트 출력을 실제 시스템과 연동했다"는 임팩트를 만들기 좋은 지점.

---

## 3. Tool Context — 실전 문법 + 제조 AX 적용

### 3.1 읽기 — State

```python
@tool
def check_line_status(runtime: ToolRuntime) -> str:
    """현재 세션에서 파악된 라인 상태 확인"""
    alerts = runtime.state.get("pending_alerts", [])
    return f"미해결 알람 {len(alerts)}건" if alerts else "정상"
```

### 3.2 읽기 — Store

```python
@tool
def get_equipment_history(equipment_id: str, runtime: ToolRuntime[PlantContext]) -> str:
    """설비의 과거 고장/정비 이력 조회 (여러 세션에 걸쳐 누적된 데이터)"""
    store = runtime.store
    history = store.get(("maintenance_history",), equipment_id)
    if history:
        return f"최근 정비 이력: {history.value.get('records', [])}"
    return "이력 없음"
```

**실무 활용**: "이 설비, 예전에도 이런 적 있었나?"에 답하려면 **세션을 넘나드는 Store가 필수** — State만 쓰면 지난주 고장 이력을 이번 세션에서 알 수가 없다. 이 구분이 컨텍스트 엔지니어링에서 가장 실전 임팩트가 큰 포인트.

### 3.3 쓰기 — State (`Command` 반환)

```python
@tool
def detect_anomaly(equipment_id: str, reading: float, threshold: float, runtime: ToolRuntime) -> Command:
    """센서값이 임계치를 넘으면 세션 state에 알람 등록"""
    is_critical = reading > threshold * 1.2

    update = {
        "pending_alerts": runtime.state.get("pending_alerts", []) + [
            {"equipment_id": equipment_id, "message": f"{reading} > {threshold}", "detected_at": "now"}
        ],
        "has_critical_alert": is_critical,
        "messages": [ToolMessage(content=f"이상 감지: {equipment_id}", tool_call_id=runtime.tool_call_id)],
    }
    return Command(update=update)
```

> `tool_call_id`를 맞춰야 모델이 "내가 호출한 도구의 응답"으로 정상 인식한다.

### 3.4 쓰기 — Store

```python
@tool
def log_maintenance_event(equipment_id: str, note: str, runtime: ToolRuntime[PlantContext]) -> str:
    """정비 이력을 장기 저장 — 다음 세션에서도 참조 가능"""
    store = runtime.store
    existing = store.get(("maintenance_history",), equipment_id)
    records = existing.value.get("records", []) if existing else []
    records.append(note)
    store.put(("maintenance_history",), equipment_id, {"records": records[-20:]})
    return f"Logged: {note}"
```

---

## 4. Life-cycle Context — 실전 문법

### 4.1 대표 패턴: Summarization

```python
SummarizationMiddleware(
    model="openai:gpt-5.4-mini",
    trigger=('tokens', 4000),
    keep=('messages', 20),
)
```

- 요약은 **state를 영구적으로 업데이트**한다 — 오래된 메시지를 요약으로 영구 대체.
- 상세 리스크는 `01-middleware.md` §5.2 참고. **제조 AX에서는 임계값 근처 수치가 요약으로 뭉개지지 않도록 §2.2의 `pending_alerts`처럼 안전 관련 데이터는 메시지가 아니라 전용 state 필드에 따로 보관하는 걸 권장.**

### 4.2 그 외 라이프사이클 패턴

- 가드레일 → `03-guardrails.md` 참고 (설비 정지 승인, PII 보호 등)
- 로깅/모니터링 → 알람 발생 시점 기록, 규정 준수 감사 추적에 활용

---

## 5. 실무 활용 — 제조 AX 에이전트 설계 순서

해커톤에서 이 챕터를 실제로 적용한다면 아래 순서로 설계하는 걸 권장한다.

**1단계 — Runtime Context부터 정의**: 누가 쓰는 에이전트인가?
```python
@dataclass
class PlantContext:
    user_id: str
    user_role: str    # operator / manager / engineer
    line_id: str
```

**2단계 — State 스키마 설계**: 이번 세션 동안 뭘 기억해야 하는가?
```python
class PlantAgentState(AgentState):
    pending_alerts: list[dict]
    has_critical_alert: bool
```

**3단계 — Store 연동 설계**: 세션을 넘어 뭘 기억해야 하는가?
- 설비별 고장/정비 이력 → `("maintenance_history",)` 네임스페이스
- 작업자별 알림 선호(문자로 받을지, 대시보드로만 볼지) → `("preferences",)` 네임스페이스

**4단계 — Model Context로 역할/상황별 조정**
- 역할별 동적 프롬프트(§2.1)
- 역할별 도구 필터링(§2.3)
- 긴급도별 모델/응답 포맷 전환(§2.4, §2.5)

**5단계 — Life-cycle로 장기 운영 안정성 확보**
- 요약으로 장시간 모니터링 세션 관리
- 가드레일로 위험 행동 통제 (`03-guardrails.md`)

**심사에서 설명하기 좋은 포인트**: "Runtime Context/State/Store를 역할에 맞게 분리했다"는 설명은 단순히 "동작한다"를 넘어 **"여러 현장 인원이 동시에 써도 안전하게 격리된다"**는 신뢰를 준다 — 실제 공장에 배포 가능한 수준을 고려했다는 인상을 남기기 좋다.
