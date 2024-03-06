import os
import json
from celery_setup.generate_config import generate_config, SentinelConfig, RedisConfig

redis_port = os.getenv('REDIS_PORT')
broker_db_num = os.getenv('CELERY_REDIS_BROKER_DB_NUM')
result_backend_db_num = os.getenv('CELERY_REDIS_RESULT_BACKEND_DB_NUM')

# Either define SENTINEL_HEADLESS_URL if using sentinel or REDIS_URL for a simple redis instance
sentinel_url = os.getenv('SENTINEL_HEADLESS_URL')
sentinel_transport_opts = json.loads(os.getenv('SENTINEL_TRANSPORT_OPTS', '{}'))
sentinel_password = os.getenv('SENTINEL_PASSWORD')
redis_url = os.getenv('REDIS_URL')
redis_password = os.getenv('REDIS_PASSWORD')

def generate_config_from_env():
    return generate_config(
        RedisConfig(redis_url, redis_password, redis_port, broker_db_num, result_backend_db_num),
        SentinelConfig(sentinel_url, sentinel_password, redis_port, sentinel_transport_opts)
    )
