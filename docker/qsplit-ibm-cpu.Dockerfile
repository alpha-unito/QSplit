FROM qsplit-base:latest

WORKDIR /workspace

RUN pip install ".[ibm-cpu]"

CMD ["python3"]