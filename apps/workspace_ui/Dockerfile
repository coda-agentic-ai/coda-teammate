FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy only the workspace_ui app
COPY apps/workspace_ui/ ./

EXPOSE 5001

# Use uvicorn with reload for development
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5001", "--reload"]
