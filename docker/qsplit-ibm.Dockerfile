FROM qsplit-base:latest

WORKDIR /workspace

RUN pip install ".[ibm-quantum]"

CMD ["python3"]