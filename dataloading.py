import logging
import traceback
from datetime import datetime, timedelta

import elasticsearch
from elasticsearch import Elasticsearch
from perceval.backends.core.github import GitHub, CATEGORY_PULL_REQUEST

from config import GITHUB_API_TOKEN, ELASTICSEARCH_HOST


def create_index(elastic_search: Elasticsearch, pull_request_index: str):
    try:
        elastic_search.indices.create(index=pull_request_index)
        elastic_search.indices.put_settings(index=pull_request_index, body={
            "index.mapping.total_fields.limit": 5000
        })
        print("Index %s created" % pull_request_index)
        return True
    except Exception:
        print("Index already exists, remove before relaunching the script")
        logging.error(traceback.format_exc())
        return False


def index_pull_request(elastic_search: Elasticsearch, pull_request_data: dict, pull_request_index: str):
    pull_request_id: str = str(pull_request_data['number'])
    try:
        elastic_search.index(index=pull_request_index, id=pull_request_id, document=pull_request_data)
    except Exception:
        print("Could not store pull request %s " % pull_request_id)
        logging.error(traceback.format_exc())


def get_and_store(owner: str, repository: str, factor: int = 0, new_index: bool = True):
    pull_request_index: str = owner.lower() + "-" + repository.lower()

    days_in_year: int = 365
    reference: datetime = datetime(year=2021, month=11, day=1)

    from_date: datetime = reference - timedelta(days=(factor + 1) * days_in_year)
    to_date: datetime = reference - timedelta(days=factor * days_in_year)

    pull_requests: GitHub = GitHub(owner=owner, repository=repository,
                                   api_token=GITHUB_API_TOKEN, sleep_for_rate=True)
    elastic_search: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    if new_index and not create_index(elastic_search, pull_request_index):
        return

    counter: int = 0
    print("Loading pull request data from %s/%s. From %s to %s" % (owner, repository, str(from_date), str(to_date)))
    for pull_request_data in pull_requests.fetch(category=CATEGORY_PULL_REQUEST, from_date=from_date,
                                                 to_date=to_date):
        print('.', end='')
        index_pull_request(elastic_search, pull_request_data['data'], pull_request_index)
        counter += 1

    print("\n%d issues stored" % counter)
