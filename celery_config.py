import os

broker_url = os.getenv('CELERY_BROKER_URL')  # 'redis://127.0.0.1:6379/2'
broker_transport_options = {}
result_backend = os.getenv('CELERY_BACKEND_URL')  # 'redis://127.0.0.1:6379/3'
