FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml ./
COPY qsplit ./qsplit

RUN pip install --upgrade pip setuptools wheel \
    && pip install .

ENV PYTHONPATH=/workspace

CMD ["python3"]