import itertools
from datetime import datetime
from typing import List, Any, Tuple

import elasticsearch
import matplotlib.pyplot as plt
import pandas as pd
from elasticsearch import Elasticsearch
from matplotlib.figure import Figure
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.var_model import VARResults, LagOrderResults

from config import ELASTICSEARCH_HOST
from dataloading import PULL_REQUEST_INDEX

MERGES_PERFORMED_COLUMN: str = "merges_performed"
MERGES_SUCCESSFUL_COLUMN: str = "requests_merged"
MERGES_REQUESTED_COLUMN: str = "merge_requests"

VARIABLES: Tuple[str, str, str] = (MERGES_PERFORMED_COLUMN, MERGES_SUCCESSFUL_COLUMN)


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


def plot_dataframe(consolidated_dataframe: pd.DataFrame, plot_title: str) -> None:
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.style.use('fivethirtyeight')
    _ = consolidated_dataframe.plot(subplots=True, linewidth=2, fontsize=12, title=plot_title)
    plt.savefig(plot_title + ".png")
    plt.show()


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


def check_residual_correlation(residuals, min_threshold=1.5, max_threshold=2.5):
    statistics: list[float] = durbin_watson(residuals)

    for statistic in statistics:
        if statistic < min_threshold or statistic > max_threshold:
            print("There might be a correlation here . Statistic: %f" % statistic)
            return False

    print("No significant serial correlation. Result: " + str(statistics))
    return True


def check_causality(training_result: VARResults, causality_threshold=0.05
                    ):
    for cause_data_column in VARIABLES:
        for effect_data_column in VARIABLES:
            if cause_data_column != effect_data_column:
                causality_results: CausalityTestResults = training_result.test_causality(causing=cause_data_column,
                                                                                         caused=effect_data_column,
                                                                                         kind='wald',
                                                                                         signif=causality_threshold)
                print(causality_results.summary())


def train_var_model(consolidated_dataframe: pd.DataFrame, user_login: str, max_order: int = 12,
                    information_criterion='aic',
                    periods=24):
    test_observations: int = 6
    train_dataset: pd.DataFrame = consolidated_dataframe[:-test_observations]
    test_dataset: pd.DataFrame = consolidated_dataframe[-test_observations:]
    print("%s Train data: %d Test data: %d" % (user_login, len(train_dataset), len(test_dataset)))

    var_model: VAR = VAR(train_dataset)
    order_results: LagOrderResults = var_model.select_order(maxlags=max_order)
    print(order_results.summary())

    for permutation_index, permutation in enumerate(itertools.permutations(VARIABLES)):
        print("Permutation %d : %s" % (permutation_index, str(permutation)))
        train_dataset: pd.DataFrame = train_dataset[list(permutation)]

        training_result: VARResults = var_model.fit(maxlags=max_order, ic=information_criterion)
        print(training_result.summary())

        check_causality(training_result)

        if not check_residual_correlation(training_result.resid):
            print("ALERT! Serial correlation found in the residuals")
        impulse_response: IRAnalysis = training_result.irf(periods=periods)
        impulse_response.plot(figsize=(15, 15))
        plt.savefig("%d_impulse_response_%s.png" % (permutation_index, user_login))

        variance_decomposition = training_result.fevd(periods=periods)
        variance_decomposition.plot(figsize=(15, 15))
        plt.savefig("%d_variance_decomposition_%s.png" % (permutation_index, user_login))


def plot_seasonal_decomposition(consolidated_dataframe: pd.DataFrame, user_login: str,
                                column: str = MERGES_PERFORMED_COLUMN) -> None:
    merges_performed_decomposition: DecomposeResult = seasonal_decompose(
        consolidated_dataframe[column],
        model='additive')

    _: Figure = merges_performed_decomposition.plot()
    plt.savefig("seasonal_decomposition_%s_%s.png" % (column, user_login))


def analyse_user(es: Elasticsearch, user_login: str):
    merges_performed_dataframe: pd.DataFrame = get_merges_performed(es, user_login)
    merge_requests_dataframe: pd.DataFrame = get_merge_requests(es, user_login)
    requests_merged_dataframe: pd.DataFrame = get_requests_merged(es, user_login)

    consolidated_dataframe: pd.DataFrame = pd.concat([merges_performed_dataframe,
                                                      merge_requests_dataframe,
                                                      requests_merged_dataframe], axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)
    consolidated_dataframe = consolidated_dataframe.rename_axis('metric', axis=1)
    print("Data points for user %s: %d" % (user_login, len(consolidated_dataframe)))
    plot_dataframe(consolidated_dataframe, "%s: before differencing" % user_login)

    after_differencing_data = consolidated_dataframe.diff()
    after_differencing_data = after_differencing_data.dropna()
    print("Applying 1st order differencing to the data")

    for variable in VARIABLES:
        is_stationary: bool = check_stationarity(after_differencing_data, user_login, variable)
        if not is_stationary:
            print("ALERT! %s is not stationary" % variable)
            return

    plot_seasonal_decomposition(consolidated_dataframe, user_login)
    train_var_model(after_differencing_data[list(VARIABLES)], user_login)
    plot_dataframe(after_differencing_data, "%s: after differencing" % user_login)


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
    print("mergers", mergers)
    analyse_user(es, "roblourens")
    # for user_login in mergers:
    #     try:
    #         analyse_user(es, user_login)
    #     except ValueError:
    #         print("Error! Cannot analyse user %s" % user_login)


if __name__ == "__main__":
    main()
