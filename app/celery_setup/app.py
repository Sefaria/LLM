from celery import Celery
from celery_setup.config import generate_config_from_env

app = Celery('llm')
app.conf.update(**generate_config_from_env())
app.autodiscover_tasks(packages=['topic_prompt'])
