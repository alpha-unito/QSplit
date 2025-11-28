FROM python:3.14-slim

RUN apt-get update && apt-get install -y \
    gcc g++ make \
    && apt-get clean

WORKDIR /workspace

COPY . /workspace

RUN pip install --upgrade pip \
    && pip install .

ENV PYTHONPATH=/workspace

CMD ["python3"]