FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --no-dev --extra serve

# The trained bundle is mounted: docker run -v $PWD/artifacts:/app/artifacts -e UPLIFT_DATASET=x5 ...
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "uplift_optimizer.api:app", "--host", "0.0.0.0", "--port", "8000"]
