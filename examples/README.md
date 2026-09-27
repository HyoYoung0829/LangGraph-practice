# LangGraph Practice Examples

파트별 실습 파일을 주제 기준으로 나눈 예제 모음입니다.

## 01-langgraph-core

- `README.md`: LangGraph 기본 개념, State, reducer, conditional edge, Send 정리
- `main.py`: StateGraph, 조건부 분기, Send 기반 map-reduce 실습

## 02-agent-runtime-middleware

- `README.md`: LangChain agent의 모델, 프롬프트, 미들웨어, Runtime 정리
- `runtime.py`: dynamic prompt, before_model middleware, ToolRuntime, store 실습
- `middleware-auth.py`: 권한 미들웨어 실습 자리

## 03-human-in-the-loop

- `README.md`: Human-in-the-Loop, interrupt, checkpointer, Command resume 정리

## 04-structured-output

- `structured-output.py`: ToolStrategy와 Pydantic schema 기반 구조화 출력 실습

## 05-database-connection-pool

- `db-connectionpool.py`: ToolRuntime context에 DB connection pool을 담아 쓰는 실습
