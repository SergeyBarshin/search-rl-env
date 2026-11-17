FROM python:3.12-slim

ARG SOURCE_COMMIT=unknown
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    SEARCH_RL_SOURCE_COMMIT=${SOURCE_COMMIT}

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN python -m pip install --no-cache-dir '.[dev,environments]'
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY scripts/check-container.sh ./scripts/check-container.sh

CMD ["sh", "scripts/check-container.sh"]
