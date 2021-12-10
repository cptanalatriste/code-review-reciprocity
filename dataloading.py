import logging
import traceback
from datetime import datetime, timedelta

import elasticsearch
from elasticsearch import Elasticsearch
from perceval.backends.core.github import GitHub, CATEGORY_PULL_REQUEST

from config import GITHUB_API_TOKEN, ELASTICSEARCH_HOST

OWNER: str = "microsoft"
REPOSITORY: str = "TypeScript"
PULL_REQUEST_INDEX: str = OWNER.lower() + "-" + REPOSITORY.lower()


def create_index(elastic_search: Elasticsearch):
    try:
        elastic_search.indices.create(index=PULL_REQUEST_INDEX)
        elastic_search.indices.put_settings(index=PULL_REQUEST_INDEX, body={
            "index.mapping.total_fields.limit": 5000
        })
        print("Index %s created" % PULL_REQUEST_INDEX)
        return True
    except Exception:
        print("Index already exists, remove before relaunching the script")
        logging.error(traceback.format_exc())
        return False


def index_pull_request(elastic_search: Elasticsearch, pull_request_data: dict):
    pull_request_id: str = str(pull_request_data['number'])
    try:
        elastic_search.index(index=PULL_REQUEST_INDEX, id=pull_request_id, document=pull_request_data)
    except Exception:
        print("Could not store pull request %s " % pull_request_id)
        logging.error(traceback.format_exc())


def get_and_store(factor: int = 0, new_index: bool = True):
    days_in_year: int = 365
    reference: datetime = datetime(year=2021, month=11, day=1)

    from_date: datetime = reference - timedelta(days=(factor + 1) * days_in_year)
    to_date: datetime = reference - timedelta(days=factor * days_in_year)

    pull_requests: GitHub = GitHub(owner=OWNER, repository=REPOSITORY,
                                   api_token=GITHUB_API_TOKEN, sleep_for_rate=True)
    elastic_search: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    if new_index and not create_index(elastic_search):
        return

    counter: int = 0
    print("Loading pull request data from %s/%s. From %s to %s" % (OWNER, REPOSITORY, str(from_date), str(to_date)))
    for pull_request_data in pull_requests.fetch(category=CATEGORY_PULL_REQUEST, from_date=from_date,
                                                 to_date=to_date):
        print('.', end='')
        index_pull_request(elastic_search, pull_request_data['data'])
        counter += 1

    print("\n%d issues stored" % counter)


if __name__ == "__main__":
    get_and_store(factor=0, new_index=True)
    for year_factor in range(1, 10):
        get_and_store(factor=year_factor, new_index=False)
