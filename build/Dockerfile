FROM python:3.9-bullseye

ENV PYTHONUNBUFFERED 1
ENV CELERY_BROKER_URL "redis://127.0.0.1:6379/2"
ENV CELERY_BACKEND_URL "redis://127.0.0.1:6379/3"

WORKDIR /app/

# Copied separately to allow for caching of the `pip install` build step
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app/
ENTRYPOINT celery -A celery_tasks worker -l info -c 5

