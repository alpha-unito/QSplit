FROM qsplit-base:latest

WORKDIR /workspace

RUN pip install ".[dwave]"

CMD ["python3"]