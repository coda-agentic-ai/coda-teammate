FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy liaison_brain package and workspace packages
COPY apps/liaison_brain/ ./
COPY packages/ /packages/

# Install workspace packages in editable mode so they can be imported
RUN uv pip install -e /packages/sentry_foundation && \
    uv pip install -e /packages/skillhive

EXPOSE 8000

# Use uvicorn with reload for development
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
