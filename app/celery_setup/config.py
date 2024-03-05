import os
import dns.resolver
import json
import re

redis_port = os.getenv('REDIS_PORT')
broker_db_num = os.getenv('REDIS_BROKER_DB_NUM')
result_backend_db_num = os.getenv('REDIS_RESULT_BACKEND_DB_NUM')

# Either define SENTINEL_HEADLESS_URL if using sentinel or REDIS_URL for a simple redis instance
sentinel_url = os.getenv('SENTINEL_HEADLESS_URL')
sentinel_transport_opts = json.loads(os.getenv('SENTINEL_TRANSPORT_OPTS', '{}'))
redis_url = os.getenv('REDIS_URL')
redis_password = os.getenv('REDIS_PASSWORD')


def add_db_num_to_url(url, db_num):
    return url.replace(f':{redis_port}', f':{redis_port}/{db_num}')


def add_password_to_url(url, password):
    if len(password) == 0:
        return url
    return re.sub(r'((?:redis|sentinel)://)', fr'\1:{password}@', url)


if sentinel_url:
    redisdns = dns.resolver.resolve(sentinel_url, 'A')
    addressstring = []
    for res in redisdns.response.answer:
        for item in res.items:
            addressstring.append(add_password_to_url(f"sentinel://{item.to_text()}:{redis_port}", redis_password))
    joined_address = ";".join(addressstring)

    # celery config vars
    broker_url = add_db_num_to_url(joined_address, broker_db_num)
    result_backend = add_db_num_to_url(joined_address, result_backend_db_num)
    result_backend_transport_options = sentinel_transport_opts
    broker_transport_options = sentinel_transport_opts
else:
    redis_url = add_password_to_url(f"{redis_url}:{redis_port}", redis_password)
    broker_url = add_db_num_to_url(redis_url, broker_db_num)
    result_backend = add_db_num_to_url(redis_url, result_backend_db_num)
