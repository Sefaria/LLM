from celery import Celery

app = Celery('llm')
app.config_from_object('app.celery_setup.config')
app.autodiscover_tasks(packages=['app.topic_prompt'])
