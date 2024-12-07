ARG REGISTRY=''
FROM ${REGISTRY}python:3.11

RUN pip install pip==23.1.2 pip-tools==6.13.0

ENV PYTHONPATH=/src

COPY applications/insight_crawler/requirements/*.txt /requirements/
RUN pip install -r /requirements/requirements.txt
COPY applications/insight_crawler/pyproject.toml /

COPY applications/insight_crawler/src /src
WORKDIR /src
