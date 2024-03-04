FROM python:3.9-bullseye

ENV PYTHONUNBUFFERED 1
ENV REDIS_URL "redis://127.0.0.1"
ENV REDIS_PORT 6379
ENV REDIS_BROKER_DB_NUM 0
ENV REDIS_RESULT_BACKEND_DB_NUM 1
ENV QUEUE_NAME "llm"
ENV PYTHONPATH "${PYTHONPATH}:/app/"

WORKDIR /app/

# Copied separately to allow for caching of the `pip install` build step
COPY app/requirements.txt /app/requirements.txt
COPY app/llm_interface /app/llm_interface
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/

COPY build/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
