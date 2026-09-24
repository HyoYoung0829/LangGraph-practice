# LangChain/LangGraph 가드레일 정리

> 이 문서의 "실무 활용"도 동일 시나리오 기준: **공장 설비 센서 데이터를 조회하고, 이상 징후를 판단하고, 필요하면 정지/알림 조치를 취하는 에이전트.** 제조 현장에서는 가드레일이 "부가 기능"이 아니라 **에이전트를 실제로 현장에 투입할 수 있느냐를 가르는 핵심 설계**라, 이 챕터가 해커톤에서 가장 설득력 있게 보여줄 수 있는 부분이다.

## 1. 가드레일이란

에이전트 실행 중 주요 지점에서 콘텐츠를 검증·필터링하는 것. 목적은 3가지:
- 부적절한 입력 차단
- 민감 정보(PII) 보호
- 출력 품질 보장

**가드레일은 별도 시스템이 아니라 미들웨어로 구현된다** — `before_agent`/`after_agent`/`before_model`/`wrap_model_call` 등이 그대로 가드레일의 구현 수단이다.

**제조 AX에서의 의미 확장**: 위 3가지 목적에 제조 현장 특유의 4번째가 추가된다고 보면 된다 — **"물리 세계에 영향을 주는 행동(설비 정지, 라인 중단)을 함부로 실행하지 않기."** 챗봇의 가드레일이 "이상한 말을 하지 않게"라면, 현장 에이전트의 가드레일은 "이상한 행동을 하지 않게"에 더 가깝다.

---

## 2. 가드레일의 두 가지 접근 방식

| 방식 | 방법 | 속도/비용 | 강점 | 약점 |
|---|---|---|---|---|
| **결정론적(Deterministic)** | 정규식, 키워드 매칭, 명시적 규칙 | 빠름, 저비용 | 예측 가능, 재현성 100% | 미묘한 위반(문맥적 우회)은 놓칠 수 있음 |
| **모델 기반(Model-based)** | LLM/분류기로 의미론적 평가 | 느림, 고비용 | 규칙이 못 잡는 문맥적 위반 포착 | 매번 판단이 미묘하게 다를 수 있음(비결정적) |

**제조 AX 매핑**: "센서값이 안전 임계치를 넘었는가"는 **결정론적**으로 판단해야 한다(`if reading > threshold` — 이게 LLM 판단에 맡길 영역이 아님, 숫자 비교는 코드가 100% 정확). 반면 "이 이상 현상의 원인이 뭘로 추정되는가", "이 알람 문구가 현장 작업자에게 이해하기 쉬운가" 같은 건 **모델 기반**이 적합하다. **안전 임계값 판단을 LLM에게 맡기지 않는 것 자체가 하나의 설계 원칙**이라고 봐도 된다.

### ⚠️ 용어 정정: "결정론적" ≠ "휴리스틱"

- **결정론적(deterministic)**: 같은 입력 → 항상 같은 출력이 보장되는지 (**재현성**의 축)
- **휴리스틱(heuristic)**: 정확한 해법이 아니라 경험적 근사 규칙을 쓰는지 (**정확성/근사**의 축)

정규식 매칭이나 Luhn 알고리즘 검증은 **"패턴에 맞으면 100% 매치"하는 명확한 if-then 규칙**이지, "대충 맞을 확률이 높은" 근사적 추측이 아니다. 대비축은 "휴리스틱 vs 정확한 알고리즘"이 아니라 **"규칙(rule) vs 모델(model)"**.

---

## 3. `hook_config` — `jump_to`를 쓰려면 필요한 선언

```python
@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime) -> dict[str, Any] | None:
    ...
```

```python
from langchain.agents.middleware import AgentMiddleware, hook_config

class ContentFilterMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        ...
```

LangGraph는 실행 전에 노드 간 연결을 그래프로 미리 컴파일하므로, `jump_to`를 쓰려면 **어디로 갈 수 있는지 미리 선언**해야 한다. `can_jump_to`는 `"tools"`/`"model"`/`"end"` 셋 중에서 고르며, 가드레일에서는 거의 항상 `["end"]`(위반 시 즉시 종료).

---

## 4. 내장 가드레일 미들웨어

### 4.1 `PIIMiddleware`

| strategy | 동작 | 제조 AX 예 |
|---|---|---|
| `redact` | `[REDACTED_TYPE]`으로 완전 치환 | 작업자 이메일 |
| `mask` | 부분적으로 가림 | 사번 뒷자리만 노출 |
| `hash` | 결정론적 해시(추적성 유지) | 로그 분석용 IP — 누구인지는 감추되 동일인 추적은 유지 |
| `block` | 탐지 시 예외, 처리 중단 | 외부 반출 금지 정보(장비 시리얼, 도면 번호 등) |

```python
middleware=[
    PIIMiddleware("employee_id", detector=r"EMP-\d{6}", strategy="mask", apply_to_output=True),
    PIIMiddleware("equipment_serial", detector=r"SN-[A-Z0-9]{10}", strategy="block", apply_to_output=True),
]
```

**제조 AX 실무**: 개인정보(PII)뿐 아니라 **영업비밀(설비 시리얼, 공정 파라미터, 협력사 정보)**도 같은 메커니즘(`detector` 커스텀 정규식)으로 보호할 수 있다는 게 실전 확장 포인트. `apply_to_output=True`가 특히 중요 — 에이전트가 **생성한 리포트**에 이런 정보가 그대로 노출되는 걸 막아야 심사/외부 시연에서도 안전하다.

### 4.2 `HumanInTheLoopMiddleware` — 제조 AX의 핵심 안전장치

```python
agent = create_agent(
    model=model,
    tools=[read_sensor_data, stop_equipment, adjust_setpoint, send_alert],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "stop_equipment": True,     # 설비 정지 — 반드시 승인
                "adjust_setpoint": True,    # 공정 파라미터 변경 — 반드시 승인
                "send_alert": False,        # 알림 발송 — 자동 허용 (되돌리기 쉬움)
                "read_sensor_data": False,  # 조회 — 자동 허용
            }
        ),
    ],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "line_a_shift1"}}
invoke_graph(agent, {"messages": [...]}, config=config)
# → stop_equipment 호출 직전 일시 중지
invoke_graph(agent, Command(resume={"decisions": [{"type": "approve"}]}), config=config)
```

**판단 기준 — 어떤 도구에 승인을 걸지**: "**되돌리기 쉬운가(reversible)**"를 기준으로 나누는 게 실전에서 잘 통한다.
- 되돌리기 쉬움(조회, 알림) → 자동 허용
- 되돌리기 어렵거나 물리적 영향이 있음(설비 정지, 파라미터 변경) → 사람 승인 필수

**해커톤 심사 포인트**: "AI가 알아서 설비를 제어한다"는 위험하다는 인상을 주기 쉽지만, "AI는 판단과 제안까지, 실행은 사람이 최종 확인"이라는 구조는 **실제 제조 현장이 AI 도입을 주저하는 이유(안전, 책임소재)를 정확히 짚은 설계**로 보인다. 발표에서 이 부분을 "왜 이렇게 설계했는가"의 핵심 답변으로 쓰기 좋다.

---

## 5. 커스텀 가드레일

### 5.1 Before Agent — 결정론적, 입력 사전 차단

```python
@before_agent(can_jump_to=["end"])
def safety_range_filter(state: AgentState, runtime) -> dict[str, Any] | None:
    """센서값이 물리적으로 불가능한 범위면 처리 자체를 차단 (센서 오류/노이즈 방어)"""
    if not state["messages"]:
        return None
    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    # 예: 메시지에 포함된 온도값이 -50~500도 범위를 벗어나면 센서 오류로 간주
    import re
    match = re.search(r"(-?\d+\.?\d*)\s*(도|℃|C)", first_message.content)
    if match:
        temp = float(match.group(1))
        if not (-50 <= temp <= 500):
            return {
                "messages": [{"role": "assistant", "content": f"센서값({temp}℃)이 물리적으로 비정상 범위입니다. 센서 점검이 필요합니다."}],
                "jump_to": "end",
            }
    return None
```

**제조 AX 실무**: 이상 감지 로직에 들어가기 전에 **"애초에 데이터 자체가 신뢰할 수 있는가"**부터 결정론적으로 걸러내는 패턴. LLM에게 "이 온도 이상해?"를 매번 판단시키는 것보다 훨씬 빠르고 확실하다.

### 5.2 클래스 기반 — 라인별 재사용 가능한 임계값 필터

```python
class ThresholdGuardrailMiddleware(AgentMiddleware):
    """설비 라인마다 다른 안전 임계값을 파라미터로 받는 재사용형 가드레일"""

    def __init__(self, line_id: str, max_temp: float, max_pressure: float):
        super().__init__()
        self.line_id = line_id
        self.max_temp = max_temp
        self.max_pressure = max_pressure

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        # 라인별 설정값을 기준으로 사전 검증 로직 수행
        ...
        return None

# 라인마다 다른 임계값으로 재사용
middleware=[
    ThresholdGuardrailMiddleware(line_id="A", max_temp=85.0, max_pressure=3.5),
]
```

> `line_id`, `max_temp`는 라인마다 고정된 **설정값**이라 `self`에 둬도 안전 — `01-middleware.md`의 "`self`는 설정값, `state`는 세션별 누적값" 원칙 그대로.

### 5.3 After Agent — 모델 기반, 최종 출력 검증

```python
safety_model = init_chat_model("gpt-5.4-mini")

@after_agent(can_jump_to=["end"])
def report_quality_check(state: AgentState, runtime) -> dict[str, Any] | None:
    """에이전트가 생성한 리포트가 현장에서 오해를 살 만한 표현을 쓰지 않았는지 검증"""
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    check_prompt = f"""아래는 설비 이상 감지 리포트입니다. 현장 작업자가 오해하거나 과소평가할 소지가 있는 모호한 표현이 있으면 'UNCLEAR', 명확하면 'CLEAR'로만 답하세요.

    리포트: {last_message.content}"""

    result = safety_model.invoke([{"role": "user", "content": check_prompt}])

    if "UNCLEAR" in result.content:
        return {
            "messages": [{"role": "assistant", "content": "리포트 표현이 모호할 수 있습니다. 담당자 확인이 필요합니다."}],
            "jump_to": "end",
        }
    return None
```

**제조 AX 실무**: 안전 관련 리포트는 "정확성"뿐 아니라 "명확성"도 중요하다 — 애매한 표현("주의가 필요할 수도 있음") 때문에 현장에서 조치가 늦어지는 게 실제 사고 원인 중 하나. 이런 걸 모델 기반 가드레일로 한 번 더 점검하는 건 실무적으로 설득력 있는 활용이다.

---

## 6. 여러 가드레일 결합 — Defense in Depth

빠른 결정론적 검사를 먼저, 느린 모델 기반 검사를 나중에.

```python
agent = create_agent(
    model=model,
    tools=[read_sensor_data, stop_equipment, send_alert],
    middleware=[
        # 계층 1: 결정론적 — 센서 데이터 자체의 유효성 (밀리초)
        safety_range_filter,
        # 계층 2: PII/영업비밀 보호 (밀리초)
        PIIMiddleware("employee_id", detector=r"EMP-\d{6}", strategy="mask", apply_to_output=True),
        # 계층 3: 물리적 행동에 대한 사람 승인
        HumanInTheLoopMiddleware(interrupt_on={"stop_equipment": True, "send_alert": False}),
        # 계층 4: 모델 기반 — 리포트 품질/명확성 (초 단위, 비용 발생)
        report_quality_check,
    ],
    checkpointer=InMemorySaver(),
)
```

**잘못된 순서의 실제 비용**: 모델 기반 검사를 앞에 두면 명백히 비정상적인 센서값(예: 통신 오류로 -9999가 들어온 경우)까지 매번 LLM 호출을 거치게 되어 **비용과 지연이 둘 다 늘어난다**. 결정론적 필터를 항상 먼저 배치하는 게 원칙.

---

## 7. 실무 활용 — 제조 AX 가드레일 설계 순서

해커톤에서 가드레일 챕터를 실제로 적용한다면 아래 순서를 권장한다.

**1단계 — "무엇이 되돌릴 수 없는 행동인가"부터 정의**
설비 정지, 파라미터 변경처럼 물리적/비가역적 영향을 주는 도구 목록을 먼저 만든다. 이게 `HumanInTheLoopMiddleware`의 `interrupt_on` 설계 기준이 된다.

**2단계 — 결정론적으로 걸러낼 수 있는 것부터 규칙화**
센서값 물리 범위, 금지 키워드, PII/영업비밀 패턴 — LLM 판단 없이 코드로 100% 확실하게 처리 가능한 것들.

**3단계 — 모델 기반 검증이 진짜 필요한 지점만 선별**
"이 리포트가 이해하기 쉬운가", "이 원인 분석이 그럴듯한가"처럼 **규칙으로 표현하기 어려운 것만** LLM에 맡긴다. 안전 임계값 비교 같은 걸 LLM에게 맡기는 건 오히려 신뢰성을 떨어뜨린다는 걸 명심.

**4단계 — 순서 배치 및 로깅**
빠른 것부터, 느린 것 나중에. 모든 가드레일 트리거는 로깅해서 나중에 "이 에이전트가 왜 이런 판단을 했는지" 추적 가능하게 만든다 — 제조 현장은 규정 준수/감사 요구가 많아서 이 부분이 실제 도입 시 반드시 요구되는 기능이다.

**심사에서 설명하기 좋은 요약**: "판단은 LLM에게, 안전은 규칙에게, 최종 실행은 사람에게"라는 3단 원칙으로 가드레일을 설계했다고 말할 수 있으면, 단순히 동작하는 데모를 넘어 **왜 이렇게 만들었는지에 대한 근거 있는 답변**이 된다.
