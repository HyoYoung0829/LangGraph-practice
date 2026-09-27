# Human-in-the-Loop (HITL) 정리

> `langchain` v1 기준. HITL은 새로운 실행 지점을 만드는 게 아니라, 이미 배운 **미들웨어의 `after_model` 훅 위에 지어진 고수준 컴포넌트**다 — 이 관계를 이해하는 게 이번 챕터의 핵심.

## 1. HITL의 포지션 — 왜 미들웨어인가

- 예전(raw 랭그래프, 구버전 `create_react_agent`) 방식: 개발자가 직접 조건 함수를 짜고, 별도 노드를 만들고, `add_conditional_edges`로 라우팅을 손수 연결해야 했음
- 지금(`create_agent` v1) 방식: `HumanInTheLoopMiddleware`를 `middleware` 리스트에 등록하고, **"어느 도구가 멈출지"만 선언적으로 지정**
- 이건 새로운 패턴이 아니라, 지금까지 계속 봐온 흐름의 연장선:

| | 직접 구현 | 미들웨어로 통합 |
|---|---|---|
| 동적 프롬프트 | `add_conditional_edges` + 별도 함수 | `@dynamic_prompt` |
| 모델 선택 | 조건문으로 분기 | `wrap_model_call` |
| 권한 체크/차단 | 조건문 + `Command(goto=...)` | `before_agent` + `jump_to` |
| **HITL** | 조건문 + 별도 노드 + interrupt 수동 호출 | `HumanInTheLoopMiddleware` |

## 2. 핵심 개념 ①: 실행 라이프사이클 — "언제" 멈추는가

### 2.1 전체 순서

```
1. 쿼리 입력
2. before_model 훅 (필요하면 DB 연결 준비 등)
3. 모델 호출 → LLM이 "이 도구를, 이 인자로 호출한다"고 판단 완료
              → AIMessage(tool_calls 포함) 생성
              ★ 여기서 LLM의 "판단"은 이미 끝남 — 도구 호출을 결정하는 것 자체가 LLM 역할의 끝
4. after_model 훅 실행
   → HumanInTheLoopMiddleware가 방금 나온 AIMessage의 tool_calls를 검사
   → interrupt_on에 걸린 도구가 있으면 → interrupt() 호출 → 실행 멈춤
5. (사람이 approve/edit/reject/respond 결정)
6. Command(resume=...) → 멈춘 지점에서 재개
   → 결정에 따라: 도구를 실제로 실행(approve/edit)하거나, 거부 메시지를 ToolMessage처럼 합성(reject)
7. 도구 실행 결과가 ToolMessage로 messages에 추가됨
8. 그래프 루프가 다시 "model" 노드로 감 → 이건 진짜 새로운 LLM 호출
   → messages 전체(ToolMessage 포함)를 프롬프트로 받아 다음 행동/최종 답변을 판단
```

### 2.2 헷갈리기 쉬운 지점 정리

- **"interrupt가 걸릴 때 LLM 판단이 아직 안 끝난 거 아닌가?"** → 아니다. `after_model`이 실행된다는 것 자체가 "모델 호출이 이미 끝나고 응답이 생성된 뒤"라는 뜻. LLM은 "이 도구를 이 인자로 부르겠다"는 판단을 **이미 완료**한 상태이고, HITL은 그 판단의 **실행 여부만 검문**함
- **"재개(resume)와 재호출(다시 모델 부르기)은 같은 건가?"** → 아니다, 순차적으로 일어나는 별개의 일. `resume`은 멈춘 그래프를 다시 흐르게 하는 것(그 자체는 LLM 호출이 아님) → 재개된 흐름 안에서 도구가 실행되고 → 그 결과를 갖고 그래프가 다시 model 노드로 가면서 **비로소 새로운 LLM 호출**이 일어남
- **도구 결과가 다음 판단에 반영되는 이유**: `ToolMessage`는 단순 기록이 아니라, 다음 모델 호출 때 **`messages` 전체와 함께 실제로 프롬프트에 실려 전달되는 입력 데이터**이기 때문. 모델은 이 `ToolMessage`를 읽고 "성공했으니 다음은 이렇게 하자"를 판단함

### 2.3 HITL이 `after_model` 기반이라는 것의 의미

- `HumanInTheLoopMiddleware`는 새 실행 지점이 아니라, **`after_model`이라는 이미 있는 훅 위에 "도구 호출 검사 → interrupt → 재개 처리" 로직을 얹은 고수준 컴포넌트**
- 그래서 커스텀 `after_model` 함수와 같은 `middleware` 리스트에 있으면, **적은 순서대로** 차례로 실행됨 (여러 before 훅이 순서대로 실행되는 것과 같은 원리)
```python
middleware=[
    log_after_model,               # ① 먼저 실행
    HumanInTheLoopMiddleware(...), # ② 그다음 실행 (역시 after_model 훅)
]
```
- 비유: `useEffect`를 직접 쓰는 게 아니라, `useEffect` 기반으로 이미 완성되어 배포된 커스텀 훅(라이브러리 컴포넌트)에 가까움 — 내부 구현은 몰라도 되고, 옵션(`interrupt_on`)만 넘기면 됨

## 3. 핵심 개념 ②: `interrupt_on` — 도구별 정책 선언

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[write_file, read_file, delete_file],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,     # 모든 결정(approve/edit/reject/respond) 허용
                "delete_file": True,
                "read_file": False,     # 안전한 작업, 승인 불필요 — 자동 실행
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),  # 필수
)
```

- 키: 도구 함수 이름 / 값: `True`(모든 결정 허용) / `False`(개입 없이 바로 실행) / `{"allowed_decisions": [...]}`(세밀 제어)
- `description_prefix`: interrupt 안내 메시지의 접두어 — 지난번 배운 `tool_message_content`와 같은 결의 "사람이 읽을 문구" 커스터마이징

### 3.1 위험도별 정책 설계 원칙

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "delete_database": {"allowed_decisions": ["approve", "reject"]},  # 위험: 편집 불가 (수정으로 몰래 바꿔치기 방지)
        "send_email": True,                                               # 보통 위험: 모든 결정 허용
        "read_data": False,                                               # 안전: 승인 불필요
    }
)
```

## 4. 핵심 개념 ③: Checkpointer — 멈춘 지점을 기억하는 방법

### 4.1 왜 필요한가

- interrupt로 멈추면, 그 순간의 State(메시지, 멈춘 위치)를 어딘가 저장해뒀다가 **정확히 그 지점부터 재개**해야 함
- 애플리케이션 코드(개발자 스크립트)는 State를 직접 들고 있지 않음 — **오직 `thread_id`라는 열쇠만** 들고 있으면 됨

```python
config = {"configurable": {"thread_id": "thread_001"}}

# 최초 invoke — 그래프 실행 중 매 시점의 State가 thread_id로 체크포인터에 저장됨
result = agent.invoke({"messages": [...]}, config=config)

# 재개 invoke — 같은 thread_id로 체크포인터에서 멈춘 지점을 찾아와 이어감
result = agent.invoke(Command(resume={...}), config=config)
```

### 4.2 Store와의 차이

| | Store | Checkpointer |
|---|---|---|
| 저장하는 것 | 사용자 선호도 같은 임의의 장기 기억 | **그래프 실행 상태 그 자체** (어디까지 왔는지) |
| 목적 | 다음 대화에서도 참고할 정보 | 일시정지했다가 정확히 그 지점부터 재개하기 위함 |
| 찾는 키 | `(네임스페이스,) + 키` | `thread_id` |

- 둘 다 "키로 저장하고 키로 찾아온다"는 구조는 동일, 저장 대상만 다름
- `InMemorySaver()`: 가장 단순한 구현체(메모리에만 저장, 프로세스 종료 시 소멸) — 프로덕션은 `AsyncPostgresSaver` 등으로 교체

## 5. `Command`의 두 번째 얼굴 — 그래프를 "깨우는" 용도

지금까지 `Command`는 **노드 함수 반환값**(`update`/`goto`)으로만 봤는데, HITL에서는 **`invoke()`의 인자로 직접 전달**하는 새로운 쓰임새가 나옴.

| 사용 위치 | 필드 | 의미 |
|---|---|---|
| 노드 함수 안에서 `return` | `update`, `goto` | "State를 이렇게 바꾸고, 다음엔 이 노드로 가라" |
| `agent.invoke()`의 인자 | `resume` | "멈춰있는 그래프에게, 이 값을 갖고 다시 실행을 재개해라" |

- `interrupt()`가 호출되면 해당 노드 실행이 그 자리에서 일시정지됨 (`input()`으로 입력을 기다리는 것과 비슷한 상태)
- 다음 `invoke()`를 `Command(resume=값)`으로 호출하면, 랭그래프는 "이건 새 요청이 아니라 멈췄던 그 `interrupt()` 지점에 값을 넣어주는 거구나"라고 인식해서 그 지점부터 이어감
- 둘 다 애플리케이션 코드(그래프 바깥)에서 호출됨 — 노드 안의 코드가 아님. 어느 쪽 invoke인지는 개발자가 미리 "예측"해서 다른 코드를 짜는 게 아니라, **첫 번째 invoke의 결과(`"__interrupt__" in result`)를 보고 조건부로 반응**하는 것

```python
result = agent.invoke({"messages": [...]}, config=config)

if "__interrupt__" in result:          # 결과를 보고 판단 — 미리 예측 X
    result = agent.invoke(Command(resume={...}), config=config)
```

## 6. Interrupt 데이터 확인하기

```python
if "__interrupt__" in result:
    interrupt_data = result["__interrupt__"][0].value

    for action in interrupt_data["action_requests"]:   # "이런 작업을 하려는데 괜찮아?"
        print(action["name"], action["args"], action["description"])

    for cfg in interrupt_data["review_configs"]:        # "너는 이 작업에 대해 뭘 선택할 수 있어"
        print(cfg["action_name"], cfg["allowed_decisions"])
```

- `__interrupt__`: `structured_response`, `jump_to`와 같은 종류의 **예약된 키**
- `action_requests`: 대기 중인 작업 목록 (아직 실행 안 됨)
- `review_configs`: 각 작업에 대해 허용된 결정 타입

## 7. 결정 타입 3가지

```python
from langgraph.types import Command
```

### 7.1 Approve — 그대로 실행

```python
Command(resume={"decisions": [{"type": "approve"}]})
```
원래 제안된 인자 그대로 도구를 실행.

### 7.2 Edit — 인자를 고쳐서 실행

```python
Command(resume={
    "decisions": [{
        "type": "edit",
        "edited_action": {
            "name": "write_file",
            "args": {"filename": "modified.txt", "content": "Modified content"}
        }
    }]
})
```
- 모델이 제안한 게 아니라 **사람이 수정한 버전**이 실행됨
- ⚠️ 큰 폭으로 수정하면, 모델이 그 차이를 인지하고 "제가 실수했군요, 다시 고치겠습니다" 식으로 접근 방식을 재평가해 도구를 여러 번 실행하거나 예기치 않은 행동을 할 수 있음 → **최소한의 필드만 변경**하는 게 원칙

### 7.3 Reject — 거부하고 이유 전달

```python
Command(resume={
    "decisions": [{
        "type": "reject",
        "message": "I cannot delete this file because it contains important data. Please back it up first."
    }]
})
```
- 도구가 **실행되지 않음** (approve/edit와 결정적 차이)
- `message`가 마치 `ToolMessage.content`처럼 모델에게 전달됨 → 모델이 그 맥락을 이해하고 스스로 대안을 제시함

### 7.4 (참고) Respond

- "ask_user" 스타일 도구에서, 사람의 답변을 도구 실행 없이 그대로 결과로 사용
- 부작용 있는 도구를 막는 용도로는 쓰면 안 됨 — `respond`의 메시지는 "성공한 도구 결과"로 취급되기 때문 (거부하고 싶으면 `reject`를 써야 함)

## 8. 동시(병렬) 도구 호출과 HITL이 만나는 지점

### 8.1 동시 도구 호출이란

- 모델이 하나의 `AIMessage` 안에 `tool_calls`를 **여러 개** 담아 응답할 수 있음 (서로 독립적인 작업일 때)
- 랭그래프의 `ToolNode`가 이를 **병렬로 실행** → 순차 왕복보다 빠르고 효율적
- 지난번 배운 `Send`(map-reduce)와 비슷한 결: "모델이 스스로 몇 개를 병렬 실행할지 결정"하고 "결과가 ToolMessage들로 순서대로 messages에 쌓인다"는 점이 다름

### 8.2 여러 도구가 동시에 승인이 필요하면 — Interrupt는 한 번만

> 공식 문서: "When the agent calls multiple tools that require approval, all interrupts are batched together in a single interrupt."

- **interrupt 자체는 1번만 발생**, 그 안의 `action_requests`에 승인 필요한 작업이 **모두 묶여서(batched)** 들어옴
- `decisions`도 `action_requests`와 **같은 개수, 같은 순서**로 채워서 **한 번의 resume**으로 전부 처리

```python
# action_requests가 2개(send_email, schedule_meeting)일 때
Command(resume={
    "decisions": [
        {"type": "approve"},                     # action_requests[0]에 대응
        {"type": "edit", "edited_action": {...}} # action_requests[1]에 대응
    ]
})
```
- ⚠️ 순서가 어긋나면 엉뚱한 작업에 결정이 적용됨 — 반드시 `action_requests`와 동일한 순서로 배열

## 9. 조건부 승인 정책 — 사람 대신 코드가 판단

사람이 매번 직접 판단하지 않고, 명확한 기준(금액, 범위 등)이 있으면 코드로 자동 판단 가능.

```python
if amount > 1000:
    decision = {"type": "reject", "message": f"Transfer amount ${amount} exceeds the $1,000 limit."}
else:
    decision = {"type": "approve"}

result = finance_agent.invoke(Command(resume={"decisions": [decision]}), config=config)
```
- 이때 `interrupt_on`에서 `transfer_money`는 `{"allowed_decisions": ["approve", "reject"]}`로 **edit을 아예 제외** — 금액/수신자를 몰래 바꿔치기할 여지 자체를 차단

## 10. 모범 사례 요약

| 사례 | 핵심 |
|---|---|
| 체크포인터 필수 | 없으면 interrupt 후 상태 복원 불가 → 재개 자체가 안 됨. 개발/테스트는 `InMemorySaver`, 프로덕션은 `AsyncPostgresSaver` 등 |
| Thread ID 관리 | 초기 호출과 재개 호출은 반드시 **동일한 `thread_id`** 사용. 세션마다 고유 ID(UUID 등) 부여 |
| 결정 순서 | `decisions`는 `action_requests`와 정확히 같은 순서로 |
| 편집 시 주의사항 | 최소한의 필드만 변경 — 대폭 수정은 모델의 예기치 않은 재평가를 유발 |
| 적절한 승인 정책 | 위험한 작업(`delete_database`)은 승인/거부만, 보통 위험(`send_email`)은 모든 결정 허용, 안전한 작업(`read_data`)은 승인 불필요 |

## 11. 전체 계층 요약

```
HumanInTheLoopMiddleware
├─ 기반: after_model 훅 (LLM의 도구 호출 판단이 끝난 직후 개입)
│
├─ 설정
│   ├─ interrupt_on: 도구별 정책 (True / False / {"allowed_decisions": [...]})
│   ├─ description_prefix: 안내 문구
│   └─ checkpointer: 필수, thread_id로 멈춘 지점 조회/저장
│
├─ 실행 흐름
│   모델 호출(판단 완료) → after_model에서 tool_calls 검사
│   → interrupt_on 대상이면 interrupt() → 실행 정지, __interrupt__ 반환
│   → (동시 호출이면 action_requests에 여러 건 batched)
│   → Command(resume={"decisions": [...]}) 로 재개
│   → approve/edit면 도구 실제 실행, reject면 message를 ToolMessage처럼 합성
│   → 결과가 messages에 쌓이고 그래프가 다시 model 노드로 → 새로운 LLM 호출
│
└─ Command의 두 얼굴
    ├─ 노드 안에서 return: update + goto (그래프 정상 라우팅)
    └─ invoke 인자로 전달: resume (멈춘 그래프 재개)
```

## 참고
- 공식 문서: https://docs.langchain.com/oss/python/langchain/human-in-the-loop
- 검색 시점(2026년 9월) 기준 `HumanInTheLoopMiddleware`, `interrupt_on`, `Command(resume=...)` API는 langchain v1 기준 최신 문법과 일치함
