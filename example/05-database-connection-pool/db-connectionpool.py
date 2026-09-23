from dataclasses import dataclass
from typing import Any

# 파이썬에서 PostgreSQL 데이터베이스에 접속하기 위한 드라이버(라이브러리)
from psycopg2 import pool
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

# ── 1. 커넥션 풀 준비 (앱 시작 시 한 번만 생성) ──────────────────
db_pool = pool.ThreadedConnectionPool(
    minconn=5,  # 미리 만들어둘 최소 연결 수
    maxconn=20,  # 동시에 허용할 최대 연결 수
    host="localhost",
    dbname="mydb",
    user="myuser",
    password="mypassword",
)


# ── 2. Context에는 연결 하나가 아니라 "풀"을 담음 ─────────────
@dataclass
class DatabaseContext:
    db_pool: Any  # 실제로는 psycopg2.pool.ThreadedConnectionPool
    user_id: str


# ── 3. 도구: 실행될 때마다 풀에서 연결을 빌리고, 끝나면 반납 ──
@tool
def query_database(sql: str, runtime: ToolRuntime[DatabaseContext]) -> str:
    """Execute SQL query on the database."""
    pool_ref = runtime.context.db_pool
    user_id = runtime.context.user_id

    conn = pool_ref.getconn()  # 풀에서 연결 하나 빌림
    try:
        cursor = conn.cursor()
        cursor.execute(sql)

        # SELECT문이면 결과를 가져오고, 아니면 커밋
        if cursor.description:
            rows = cursor.fetchall()
            result = str(rows)
        else:
            conn.commit()
            result = "쿼리 실행 완료"

        cursor.close()
        return f"Query executed for user {user_id}: {result}"

    except Exception as e:
        conn.rollback()  # 에러 나면 롤백
        return f"쿼리 실행 중 오류 발생: {e}"

    finally:
        pool_ref.putconn(conn)  # 성공하든 실패하든 반드시 반납


# ── 4. 에이전트 생성 ────────────────────────────────────────
agent = create_agent(
    model=model,
    tools=[query_database],
    context_schema=DatabaseContext,
)


# ── 5. 실행 ─────────────────────────────────────────────────
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Query user data"}]},
    context=DatabaseContext(db_pool=db_pool, user_id="user_123"),
)
print(result["messages"][-1].content)


# ── 6. 앱 종료 시 풀 전체를 닫아줌 (자원 정리) ────────────────
db_pool.closeall()
