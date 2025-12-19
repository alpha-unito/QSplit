FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip setuptools wheel \
    && pip install ".[ibm-cpu,ibm-quantum,dwave]"

ENV PYTHONPATH=/workspace

CMD ["python3"]