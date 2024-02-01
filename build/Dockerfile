FROM python:3.9-bullseye

ENV PYTHONUNBUFFERED 1
ENV REDIS_URL "redis://127.0.0.1:6379/2"
ENV REDIS_PORT 6379
ENV REDIS_BROKER_DB_NUM 0
ENV REDIS_RESULT_BACKEND_DB_NUM 1

WORKDIR /app/

# Copied separately to allow for caching of the `pip install` build step
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app/
ENTRYPOINT celery -A celery_tasks worker

