from datetime import datetime
from typing import List, Any

import pandas as pd
from elasticsearch import Elasticsearch

from dataloading import PULL_REQUEST_INDEX

MERGES_PERFORMED_COLUMN: str = "merges_performed"
MERGES_SUCCESSFUL_COLUMN: str = "requests_merged"
MERGES_REQUESTED_COLUMN: str = "merge_requests"


def do_query_with_aggregation(elastic_search: Elasticsearch, aggregation_name: str, query: dict[str, Any],
                              date_histogram: dict[str, Any]) -> pd.DataFrame:
    search_results: dict[str, dict] = elastic_search.search(index=PULL_REQUEST_INDEX,
                                                            size=0,
                                                            query=query,
                                                            aggs={
                                                                aggregation_name: {
                                                                    "date_histogram": date_histogram
                                                                }
                                                            })

    event_date_field: str = "event_date"
    results: List[dict[str, Any]] = [
        {event_date_field: datetime.strptime(data["key_as_string"], "%Y-%m-%dT%H:%M:%S.%fZ"),
         aggregation_name: data["doc_count"]}
        for data in
        search_results['aggregations'][aggregation_name]['buckets']]

    result_dataframe: pd.DataFrame = pd.DataFrame(results)
    if not result_dataframe.empty:
        result_dataframe = result_dataframe.set_index(event_date_field)
    return result_dataframe


def get_merges_performed(es: Elasticsearch, user_login: str,
                         calendar_interval: str = "month") -> pd.DataFrame:
    result_dataframe: pd.DataFrame = do_query_with_aggregation(es, MERGES_PERFORMED_COLUMN, query={
        "bool": {
            "must": {
                "match": {"merged_by.login": user_login}
            },
            "must_not": {
                "match": {"user.login": user_login}
            }
        }
    }, date_histogram={
        "field": "merged_at",
        "calendar_interval": calendar_interval
    })

    return result_dataframe


def get_merge_requests(elastic_search: Elasticsearch, user_login: str,
                       calendar_interval: str = "month") -> pd.DataFrame:
    result_dataframe: pd.DataFrame = do_query_with_aggregation(elastic_search, MERGES_REQUESTED_COLUMN, query={
        "bool": {
            "must": {
                "match": {"user.login": user_login}
            },
            "must_not": {
                "match": {"merged_by.login": user_login}
            }
        }
    }, date_histogram={
        "field": "created_at",
        "calendar_interval": calendar_interval
    })

    return result_dataframe


def get_requests_merged(es: Elasticsearch, user_login: str,
                        calendar_interval: str = "month"):
    result_dataframe: pd.DataFrame = do_query_with_aggregation(es, MERGES_SUCCESSFUL_COLUMN, query={
        "bool": {
            "must": [
                {
                    "match": {"user.login": user_login}
                },
                {
                    "match": {"merged": "true"}
                }],
            "must_not": {
                "match": {"merged_by.login": user_login}
            }
        }
    }, date_histogram={
        "field": "merged_at",
        "calendar_interval": calendar_interval
    })

    return result_dataframe


def get_all_mergers(es: Elasticsearch) -> List[str]:
    aggregation_name: str = "frequent_mergers"

    es.indices.refresh(index=PULL_REQUEST_INDEX)
    document_count: List[dict] = es.cat.count(index=PULL_REQUEST_INDEX, params={"format": "json"})
    print("Documents on index %s: %s" % (PULL_REQUEST_INDEX, document_count[0]['count']))

    search_results: dict[str, dict] = es.search(index=PULL_REQUEST_INDEX,
                                                size=0,
                                                aggs={
                                                    aggregation_name: {
                                                        "terms": {
                                                            "field": "merged_by.login.keyword"
                                                        }
                                                    }
                                                })
    mergers: List[str] = [merger_data['key'] for merger_data in
                          search_results['aggregations'][aggregation_name]['buckets']]
    print("mergers", mergers)
    return mergers
