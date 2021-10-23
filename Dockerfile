FROM python:3.9.7-slim-buster

RUN set -eux; \
	apt-get update; \
    apt-get install -y gcc

ADD app /app
WORKDIR /app

RUN pip install --no-cache-dir -r /app/requirements.txt