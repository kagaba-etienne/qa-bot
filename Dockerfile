FROM node:24-alpine AS frontend

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pymongo \
    openai \
    python-dotenv \
    requests

COPY src/main.py src/main.py
COPY src/config src/config
COPY src/routes src/routes
COPY src/controllers src/controllers
COPY src/lib src/lib
COPY src/services/chat.py src/services/chat.py
COPY --from=frontend /app/frontend/build/client frontend/build/client

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
