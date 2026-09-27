from dataclasses import dataclass
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import dynamic_prompt, ModelRequest, before_model
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime
from dotenv import load_dotenv

load_dotenv()


# 에이전트 실행 시 전달할 사용자 컨텍스트의 구조
@dataclass
class UserContext:
    user_id: str
    user_name: str
    user_tier: str  # free, premium, enterprise
    language: str


# 툴 정의
# 각 도구의 반환값은 ToolMessage로 대화 상태에 추가되고,
# 다음 LLM 호출에서 프롬프트 컨텍스트의 일부로 전달됨


# 사용자 티어에 따라 다른 검색 결과 문자열을 반환하는 도구
# 반환된 문자열은 다음 LLM 호출의 프롬프트 컨텍스트에 추가됨
@tool
def search_database(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Search the database. Access level depends on user tier."""
    print(f"[Tool Call] search_database(query={query!r})")
    user_tier = runtime.context.user_tier

    # 사용자 티어별 검색 결과를 문자열로 구성하여 반환
    if user_tier == "enterprise":
        return f"Full database search results for: {query} (Enterprise access)"
    elif user_tier == "premium":
        return f"Premium search results for: {query}"
    else:
        return f"Basic search results for: {query} (Limited to 10 results)"


# 런타임 스토어에서 검색 기록을 조회하여 LLM에 전달하는 도구
@tool
def get_user_history(runtime: ToolRuntime[UserContext]) -> str:
    """Get user's search history from store."""
    user_id = runtime.context.user_id
    print(f"[Tool Call] get_user_history(user_id={user_id!r})")

    if runtime.store:
        if history := runtime.store.get(("history",), user_id):
            return f"Recent searches: {history.value['searches']}"

    return "해당 유저의 히스토리를 찾을 수 없습니다."


# 검색어를 스토어에 저장하고 처리 결과를 LLM에 전달하는 도구
@tool
def save_search(query: str, runtime: ToolRuntime[UserContext]) -> str:
    """Save search query to user history."""
    user_id = runtime.context.user_id
    print(f"[Tool Call] save_search(query={query!r}, user_id={user_id!r})")

    if runtime.store:
        # 기존 히스토리 가져오기
        existing = runtime.store.get(("history",), user_id)
        searches = existing.value["searches"] if existing else []

        # 새 검색어 추가 (최근 5개만 유지)
        searches.append(query)
        runtime.store.put(("history",), user_id, {"searches": searches[-5:]})

        return f"Saved search: {query}"

    return "Store not available"


# 사용자 컨텍스트에 맞는 시스템 프롬프트를 LLM 호출마다 동적으로 생성
@dynamic_prompt
def multilingual_prompt(request: ModelRequest) -> str:
    # 프롬프트 생성에 필요한 사용자 정보를 런타임 컨텍스트에서 조회
    user_name = request.runtime.context.user_name
    language = request.runtime.context.language
    user_tier = request.runtime.context.user_tier

    # 한국어 사용자에게는 한국어 응답 지침을 적용
    if language == "ko":
        prompt = (
            f"당신은 도움이 되는 어시스턴트입니다. "
            f"사용자의 이름은 {user_name}입니다. "
            f"모든 최종 답변에서 반드시 사용자를 '{user_name}님'이라고 한 번 이상 직접 호칭하세요. "
            f"한국어로 답변하세요."
        )
        # 엔터프라이즈 사용자에게 전체 기능 접근 안내를 추가
        if user_tier == "enterprise":
            prompt += (
                " 이 사용자는 엔터프라이즈 회원이므로 모든 기능에 액세스할 수 있습니다."
            )
    else:
        prompt = f"You are a helpful assistant. Address the user as {user_name}."
        if user_tier == "enterprise":
            prompt += "This is an enterprise user with full access."

    return prompt


# LLM이 호출되기 직전에 실행되는 미들웨어
@before_model
def track_usage(state: AgentState, runtime: Runtime[UserContext]) -> dict | None:
    """Log model calls and increment the free-tier call count."""
    user_id = runtime.context.user_id
    user_tier = runtime.context.user_tier

    # 현재 모델 호출의 사용자와 티어를 로그로 출력
    print(f"[Usage Tracker] User: {user_id}, Tier: {user_tier}")

    # 무료 사용자의 누적 모델 호출 횟수를 확인
    if user_tier == "free":
        if runtime.store:
            usage = runtime.store.get(("usage",), user_id)
            count = usage.value["count"] if usage else 0

            if count >= 10:
                print("[Usage Tracker] Free tier limit reached!")
                # 현재는 경고만 출력하며 호출을 실제로 중단하지는 않음

            # 이번 모델 호출을 사용량에 반영
            runtime.store.put(("usage",), user_id, {"count": count + 1})

    return None


# 런타임 스토어 초기화 및 에이전트 생성

# 휘발성 메모리에 스토어를 생성
store = InMemoryStore()

# 테스트에서 사용할 검색 기록과 사용량을 미리 저장
store.put(
    ("history",), "user_001", {"searches": ["Python tutorial", "LangChain guide"]}
)
store.put(("usage",), "user_002", {"count": 5})


def create_runtime_agent():
    return create_agent(
        model="gpt-4o-mini",
        # LLM이 호출할 수 있는 도구 등록
        tools=[search_database, get_user_history, save_search],
        # 사용되는 미들웨어 등록
        middleware=[multilingual_prompt, track_usage],
        # invoke 호출 시 전달할 런타임 컨텍스트의 타입
        context_schema=UserContext,
        # 도구와 미들웨어가 공유할 스토어
        store=store,
    )


if __name__ == "__main__":
    agent = create_runtime_agent()

    # 테스트 1: 엔터프라이즈 사용자 (한국어)
    print("=== Test 1: Enterprise User (Korean) ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Search for 'machine learning'"}]},
        context=UserContext(
            user_id="user_001",
            user_name="김철수",
            user_tier="enterprise",
            language="ko",
        ),
    )
    print(result["messages"][-1].content)

    # 테스트 2: 무료 사용자 (영어)
    print("\n=== Test 2: Free User (English) ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Search for 'data science'"}]},
        context=UserContext(
            user_id="user_002",
            user_name="John Doe",
            user_tier="free",
            language="en",
        ),
    )
    print(result["messages"][-1].content)

    # 테스트 3: 검색 기록 조회
    print("\n=== Test 3: Check Search History ===")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my search history?"}]},
        context=UserContext(
            user_id="user_001",
            user_name="김철수",
            user_tier="enterprise",
            language="ko",
        ),
    )
    print(result["messages"][-1].content)
