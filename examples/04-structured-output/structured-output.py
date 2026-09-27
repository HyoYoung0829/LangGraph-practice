from pydantic import BaseModel, Field
from typing import Literal, Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()


# ── 스키마 1: 책 추천용 ────────────────────────────────────
class BookRecommendation(BaseModel):
    """책 추천 정보를 나타내는 스키마

    제목, 저자, 장르, 평점, 요약을 구조화합니다.
    """

    # 클래스 docstring은 모델이 "이 스키마를 언제 골라야 하는지" 판단하는 근거로 쓰임

    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    genre: Literal["fiction", "non-fiction", "science", "history", "biography"] = Field(
        description="Book genre"
    )  # Literal로 값 후보를 제한 — 모델이 정해진 카테고리 밖으로 못 나가게 함
    rating: int = Field(
        description="Rating from 1-5", ge=1, le=5
    )  # ge/le로 1~5 범위 강제
    summary: str = Field(description="Brief summary of the book")


# ── 스키마 2: 영화 추천용 ───────────────────────────────────
class MovieRecommendation(BaseModel):
    """영화 추천 정보를 나타내는 스키마

    제목, 감독, 개봉년도, 장르, 평점을 구조화합니다.
    """

    title: str = Field(description="Movie title")
    director: str = Field(description="Director name")
    year: int = Field(description="Release year")
    genre: Literal["action", "comedy", "drama", "horror", "sci-fi"] = Field(
        description="Movie genre"
    )
    rating: int = Field(description="Rating from 1-5", ge=1, le=5)


# ── 모델 준비 ────────────────────────────────────────────
model = init_chat_model("gpt-5.4")  # 실제 사용할 모델로 교체


# ── 에이전트 생성 ────────────────────────────────────────
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(
        # Union: 모델이 입력 문맥을 보고 두 스키마 중 알맞은 걸 스스로 선택
        schema=Union[BookRecommendation, MovieRecommendation],
        # handle_errors=True(기본값): 스키마 검증 실패 시 모델에게 피드백 후 자동 재시도
        handle_errors=True,
    ),
    system_prompt="You are a helpful entertainment recommendation assistant. 답변은 한국어로 하세요.",
)


# ── 실행 1: 책 추천 요청 ──────────────────────────────────
# "science fiction book"이라는 문맥 → 모델이 BookRecommendation 스키마를 선택할 것으로 기대
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Recommend a good science fiction book"}]}
)
print("Book recommendation:")
print(result1["structured_response"])  # BookRecommendation 인스턴스로 반환됨


# ── 실행 2: 영화 추천 요청 ─────────────────────────────────
# "comedy movie"라는 문맥 → 모델이 MovieRecommendation 스키마를 선택할 것으로 기대
result2 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Recommend a comedy movie from the 2000s"}
        ]
    }
)
print("\nMovie recommendation:")
print(result2["structured_response"])  # MovieRecommendation 인스턴스로 반환됨


# ── 참고: 실전에서는 결과 타입에 따라 분기 처리 ────────────────
# response = result1["structured_response"]
# if isinstance(response, BookRecommendation):
#     handle_book(response)
# elif isinstance(response, MovieRecommendation):
#     handle_movie(response)
