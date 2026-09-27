# LangGraph Practice

Next.js UI와 FastAPI + LangGraph API를 분리한 실습용 저장소입니다.

```text
apps/
  web/    Next.js UI
  api/    FastAPI + LangGraph API
data/     실습 데이터
examples/ LangGraph 예제
```

## 실행

API:

```powershell
cd apps/api
py -m uv sync
py -m uv run fastapi dev app/main.py
```

Web:

```powershell
cd apps/web
pnpm dev
```

- Web: http://localhost:3000
- API 문서: http://localhost:8000/docs
