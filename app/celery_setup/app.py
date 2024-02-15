from celery import Celery

app = Celery('llm')
app.config_from_object('celery_setup.config')
app.autodiscover_tasks(packages=['topic_prompt'])
