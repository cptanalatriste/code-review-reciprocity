from datetime import datetime
from typing import List, Any, Tuple

import elasticsearch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VARResults

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
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.style.use('fivethirtyeight')
    _ = consolidated_dataframe.plot(subplots=True, linewidth=2, fontsize=12, title="User %s" % user_login)
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


def check_causality(consolidated_dataframe: pd.DataFrame, user_login: str, cause_data_column: str,
                    effect_data_column: str,
                    threshold: float = 0.05,
                    max_lag: int = 6,
                    test_name: str = "ssr_chi2test") -> bool:
    granger_causality_results: dict[int, Tuple] = grangercausalitytests(
        consolidated_dataframe[[effect_data_column, cause_data_column]],
        maxlag=max_lag, verbose=False)

    consolidated_results: List[Tuple[int, float, float]] = []
    for lag, results in granger_causality_results.items():
        test_results: dict[str, Tuple]
        test_results, _ = results
        values = test_results[test_name]

        test_statistic: float = values[0]
        p_value: float = values[1]
        consolidated_results.append((lag, p_value, test_statistic))

    min_results: Tuple[int, float, float] = min(consolidated_results, key=lambda lag_results: lag_results[1])
    min_lag: int = min_results[0]
    min_p_value: float = min_results[1]
    min_test_statistic: float = min_results[2]

    if min_p_value < threshold:
        print("%s Rejecting NULL hypothesis! %s Granger CAUSES %s (p-value=%f, test=%s statistic=%f, lag=%d)" % (
            user_login, cause_data_column, effect_data_column, min_p_value, test_name, min_test_statistic, min_lag))
        return True

    print("%s: NULL hypothesis (%s DOES NOT Granger cause %s) cannot be rejected" % (
        user_login, cause_data_column, effect_data_column))
    return False


def check_stationarity(consolidated_dataframe: pd.DataFrame, user_login: str, data_column: str,
                       threshold: float = 0.05) -> bool:
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


def train_var_model(consolidated_dataframe: pd.DataFrame, user_login: str, max_order: int = 6,
                    information_criterion='aic'):
    test_observations: int = 6
    train_dataset: pd.DataFrame = consolidated_dataframe[:-test_observations]
    test_dataset: pd.DataFrame = consolidated_dataframe[-test_observations:]
    print("%s Train data: %d Test data: %d" % (user_login, len(train_dataset), len(test_dataset)))

    candidate_results: List[Tuple[int, float, float]] = []
    for candidate_order in range(1, max_order):
        var_model: VAR = VAR(train_dataset)
        training_result = var_model.fit(candidate_order)
        candidate_result: Tuple[int, float, float] = (candidate_order, training_result.aic, training_result.bic)
        candidate_results.append(candidate_result)

    selected_order: Tuple[int, float, float] = min(candidate_results, key=lambda result: result[1])
    print("%s order for VAR model: %d" % (user_login, selected_order[0]))

    var_model: VAR = VAR(train_dataset)
    training_result: VARResults = var_model.fit(maxlags=selected_order[0], ic=information_criterion)
    print(training_result.summary())

    impulse_response = training_result.irf(periods=12)
    impulse_response.plot(orth=False)


def analyse_user(es: Elasticsearch, user_login: str):
    merges_performed_dataframe: pd.DataFrame = get_merges_performed(es, user_login)
    # merge_requests_dataframe: pd.DataFrame = get_merge_requests(es, user_login)
    requests_merged_dataframe: pd.DataFrame = get_requests_merged(es, user_login)

    consolidated_dataframe: pd.DataFrame = pd.concat([merges_performed_dataframe,
                                                      requests_merged_dataframe], axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)
    consolidated_dataframe = consolidated_dataframe.rename_axis('metric', axis=1)
    print("Data points for user %s: %d" % (user_login, len(consolidated_dataframe)))
    consolidated_dataframe = consolidated_dataframe.diff()
    consolidated_dataframe = consolidated_dataframe.dropna()
    print("Applying 1st order differencing to the data")

    if MERGES_PERFORMED_COLUMN in consolidated_dataframe and MERGES_SUCCESSFUL_COLUMN in consolidated_dataframe:

        merges_successful_stationary: bool = check_stationarity(consolidated_dataframe, user_login,
                                                                MERGES_SUCCESSFUL_COLUMN)
        merges_performed_stationary: bool = check_stationarity(consolidated_dataframe, user_login,
                                                               MERGES_PERFORMED_COLUMN)
        if merges_successful_stationary and merges_performed_stationary:
            merges_cause_success: bool = check_causality(consolidated_dataframe, user_login,
                                                         cause_data_column=MERGES_PERFORMED_COLUMN,
                                                         effect_data_column=MERGES_SUCCESSFUL_COLUMN)

            _: bool = check_causality(consolidated_dataframe, user_login,
                                      cause_data_column=MERGES_SUCCESSFUL_COLUMN,
                                      effect_data_column=MERGES_PERFORMED_COLUMN)

            if merges_cause_success:
                correlation: float = consolidated_dataframe[MERGES_PERFORMED_COLUMN].corr(
                    consolidated_dataframe[MERGES_SUCCESSFUL_COLUMN])
                print(user_login, "Correlation: Merges done and successful PRs ", correlation)
                plot_offered_vs_given(consolidated_dataframe, user_login)
                train_var_model(consolidated_dataframe, user_login)
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
