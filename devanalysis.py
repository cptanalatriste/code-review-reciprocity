from datetime import datetime
from typing import List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import elasticsearch
from elasticsearch import Elasticsearch

from config import ELASTICSEARCH_HOST
from dataloading import PULL_REQUEST_INDEX

MERGES_PERFORMED_COLUMN: str = "merges_performed"
MERGES_SUCCESSFUL_COLUMN: str = "requests_merged"


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


def get_merges_performed(elastic_search: Elasticsearch, user_login: str,
                         calendar_interval: str = "month") -> pd.DataFrame:
    result_dataframe: pd.DataFrame = do_query_with_aggregation(elastic_search, MERGES_PERFORMED_COLUMN, query={
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
    aggregation_name: str = "merge_requests"

    result_dataframe: pd.DataFrame = do_query_with_aggregation(elastic_search, aggregation_name, query={
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


def get_requests_merged(elastic_search: Elasticsearch, user_login: str,
                        calendar_interval: str = "month"):
    result_dataframe: pd.DataFrame = do_query_with_aggregation(elastic_search, MERGES_SUCCESSFUL_COLUMN, query={
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


def plot_dataframe(consolidated_dataframe: pd.DataFrame, user_login: str):
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.style.use('fivethirtyeight')
    ax = consolidated_dataframe.plot(linewidth=2, fontsize=12, title="User %s" % user_login)
    ax.set_xlabel('Date')
    ax.legend(fontsize=12)
    plt.show()


def plot_offered_vs_given(consolidated_dataframe: pd.DataFrame, user_login: str):
    plt.figure(figsize=(12, 6))
    sns.jointplot(
        x=MERGES_PERFORMED_COLUMN, y=MERGES_SUCCESSFUL_COLUMN,
        edgecolor="white",
        data=consolidated_dataframe,
    )
    plt.xlabel("Performed")
    plt.ylabel("Requested")
    plt.title("User %s" % user_login)
    plt.show()


def analyse_user(elastic_search: Elasticsearch, user_login: str):
    merges_performed_dataframe: pd.DataFrame = get_merges_performed(elastic_search, user_login)
    merge_requests_dataframe: pd.DataFrame = get_merge_requests(elastic_search, user_login)
    requests_merged_dataframe: pd.DataFrame = get_requests_merged(elastic_search, user_login)

    consolidated_dataframe: pd.DataFrame = pd.concat([merges_performed_dataframe, merge_requests_dataframe,
                                                      requests_merged_dataframe], axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)

    if MERGES_PERFORMED_COLUMN in consolidated_dataframe and MERGES_SUCCESSFUL_COLUMN in consolidated_dataframe:
        correlation: float = consolidated_dataframe[MERGES_PERFORMED_COLUMN].corr(
            consolidated_dataframe[MERGES_SUCCESSFUL_COLUMN])
        print(user_login, "Correlation: Merges done and succesful PRs ", correlation)
        # plot_offered_vs_given(consolidated_dataframe, user_login)

    plot_dataframe(consolidated_dataframe, user_login)


def main():
    aggregation_name: str = "frequent_mergers"
    elastic_search: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)
    search_results: dict[str, dict] = elastic_search.search(index=PULL_REQUEST_INDEX,
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
    # user_login: str = mergers[0]
    for user_login in mergers:
        analyse_user(elastic_search, user_login)


if __name__ == "__main__":
    main()
