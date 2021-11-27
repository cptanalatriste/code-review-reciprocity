from datetime import datetime, timedelta

import elasticsearch
from elasticsearch import Elasticsearch
from perceval.backends.core.github import GitHub, CATEGORY_PULL_REQUEST

from config import GITHUB_API_TOKEN, ELASTICSEARCH_HOST

PULL_REQUEST_INDEX: str = "pull-requests-v2"


def create_index(elastic_search: Elasticsearch):
    try:
        elastic_search.indices.create(index=PULL_REQUEST_INDEX)
        elastic_search.indices.put_settings(index=PULL_REQUEST_INDEX, body={
            "index.mapping.total_fields.limit": 5000
        })
        return True
    except elasticsearch.exceptions.RequestError:
        print("Index already exists, remove before relaunching the script")
        return False


def index_pull_request(elastic_search: Elasticsearch, pull_request_data: dict):
    try:
        elastic_search.index(index=PULL_REQUEST_INDEX, document=pull_request_data)
    except elasticsearch.exceptions.RequestError:
        print("Could not store pull request %d " % pull_request_data['number'])


def main():
    owner: str = "microsoft"
    repository: str = "vscode"
    days_in_year: int = 365
    factor: int = 1
    new_index: bool = True

    from_date: datetime = datetime.now() - timedelta(days=(factor + 1) * days_in_year)
    to_date: datetime = datetime.now() - timedelta(days=factor * days_in_year)

    pull_requests: GitHub = GitHub(owner=owner, repository=repository,
                                   api_token=GITHUB_API_TOKEN, sleep_for_rate=True)
    elastic_search: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    if new_index and not create_index(elastic_search):
        return

    counter: int = 0
    for pull_request_data in pull_requests.fetch(category=CATEGORY_PULL_REQUEST, from_date=from_date,
                                                 to_date=to_date):
        print('.', end='')
        index_pull_request(elastic_search, pull_request_data['data'])
        counter += 1

    print("%d issues stored" % counter)


if __name__ == "__main__":
    main()
