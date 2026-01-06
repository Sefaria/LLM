#!/bin/bash

celery -A celery_setup.app worker -Q ${QUEUE_NAME} -l INFO --concurrency 4