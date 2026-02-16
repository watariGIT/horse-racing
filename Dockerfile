FROM python:3.10-slim

# Install system dependencies required by LightGBM (libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY config/ config/
COPY src/ src/

# Install the project itself
RUN uv sync --no-dev

EXPOSE 8080

CMD ["uv", "run", "python", "-m", "src.pipeline", "--stage", "full"]
