FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy only the workspace_ui app
COPY apps/workspace_ui/ ./

EXPOSE 5001

CMD ["uv", "run", "python", "app.py"]
