from datetime import datetime
from typing import List, Any

import elasticsearch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

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


def plot_dataframe(consolidated_dataframe: pd.DataFrame, user_login: str):
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.style.use('fivethirtyeight')
    ax = consolidated_dataframe.plot(linewidth=2, fontsize=12, title="User %s" % user_login)
    ax.set_xlabel('Date')
    ax.legend(fontsize=12)
    plt.show()


def plot_offered_vs_given(consolidated_dataframe: pd.DataFrame, user_login: str):
    plt.figure(figsize=(12, 6))
    merges_performed: np.ndarray = consolidated_dataframe[MERGES_PERFORMED_COLUMN].values.reshape(-1, 1)
    merges_successful: np.ndarray = consolidated_dataframe[MERGES_SUCCESSFUL_COLUMN].values.reshape(-1, 1)
    linear_regression: LinearRegression = LinearRegression()
    linear_regression.fit(merges_performed, merges_successful)
    predicted_merges_successful: np.ndarray = linear_regression.predict(merges_performed)

    plt.scatter(merges_performed, merges_successful)
    plt.plot(merges_performed, predicted_merges_successful, color="red")

    plt.xlabel("Performed")
    plt.ylabel("Successful")
    plt.title("User %s" % user_login)
    plt.show()


def check_stationarity(consolidated_dataframe: pd.DataFrame, user_login: str, data_column: str,
                       threshold: float = 0.05):
    test_result: list[float] = adfuller(consolidated_dataframe[data_column])
    adf_statistic: float = test_result[0]
    p_value: float = test_result[1]
    if p_value <= threshold:
        print("%s is stationary for user %s. ADF statistic: %f, p-value: %f" % (
            data_column, user_login, adf_statistic, p_value))
        return True

    print("%s is NOT stationary for user %s. ADF statistic: %f, p-value: %f" % (
        data_column, user_login, adf_statistic, p_value))
    return True


def analyse_user(es: Elasticsearch, user_login: str):
    merges_performed_dataframe: pd.DataFrame = get_merges_performed(es, user_login)
    merge_requests_dataframe: pd.DataFrame = get_merge_requests(es, user_login)
    requests_merged_dataframe: pd.DataFrame = get_requests_merged(es, user_login)

    consolidated_dataframe: pd.DataFrame = pd.concat([merges_performed_dataframe, merge_requests_dataframe,
                                                      requests_merged_dataframe], axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)
    print("Data points for user %s: %d" % (user_login, len(consolidated_dataframe)))
    consolidated_dataframe = consolidated_dataframe.diff()
    consolidated_dataframe = consolidated_dataframe.dropna()
    print("Applying 1st order differencing to the data")

    if MERGES_PERFORMED_COLUMN in consolidated_dataframe and MERGES_SUCCESSFUL_COLUMN in consolidated_dataframe:
        correlation: float = consolidated_dataframe[MERGES_PERFORMED_COLUMN].corr(
            consolidated_dataframe[MERGES_SUCCESSFUL_COLUMN])
        print(user_login, "Correlation: Merges done and successful PRs ", correlation)
        plot_offered_vs_given(consolidated_dataframe, user_login)
        check_stationarity(consolidated_dataframe, user_login, MERGES_SUCCESSFUL_COLUMN)
        check_stationarity(consolidated_dataframe, user_login, MERGES_PERFORMED_COLUMN)

    plot_dataframe(consolidated_dataframe, user_login)


def main():
    aggregation_name: str = "frequent_mergers"
    es: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)

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
    for user_login in mergers:
        analyse_user(es, user_login)


if __name__ == "__main__":
    main()
