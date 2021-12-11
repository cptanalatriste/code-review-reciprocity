import itertools
import logging
import traceback
from typing import Tuple, Any, Optional

import elasticsearch
import matplotlib.pyplot as plt
import pandas as pd
from elasticsearch import Elasticsearch
from matplotlib.figure import Figure
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults, WhitenessTestResults
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.var_model import VARResults, LagOrderResults

from aggregation import MERGES_PERFORMED_COLUMN, MERGES_SUCCESSFUL_COLUMN, get_merges_performed, get_merge_requests, \
    get_requests_merged, get_all_mergers
from config import ELASTICSEARCH_HOST

VARIABLES: Tuple[str, str] = (MERGES_PERFORMED_COLUMN, MERGES_SUCCESSFUL_COLUMN)
IMAGE_DIRECTORY: str = "img/"


def plot_dataframe(consolidated_dataframe: pd.DataFrame, plot_title: str) -> None:
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.style.use('fivethirtyeight')
    _ = consolidated_dataframe.plot(subplots=True, linewidth=2, fontsize=12, title=plot_title)
    plt.savefig(IMAGE_DIRECTORY + plot_title + ".png")
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


def check_causality(training_result: VARResults, causality_threshold=0.05) -> dict[str, bool]:
    test_results: dict[str, Any] = {}
    for cause_data_column in VARIABLES:
        for effect_data_column in VARIABLES:
            if cause_data_column != effect_data_column:
                causality_results: CausalityTestResults = training_result.test_causality(causing=cause_data_column,
                                                                                         caused=effect_data_column,
                                                                                         kind='wald',
                                                                                         signif=causality_threshold)
                granger_causality: bool = causality_results.conclusion == "reject"
                test_results[cause_data_column + "->" + effect_data_column] = granger_causality
                print(causality_results.summary())

    return test_results


def train_var_model(consolidated_dataframe: pd.DataFrame, user_login: str, max_order: int = 12,
                    information_criterion='bic',
                    periods=24) -> dict[str, set]:
    test_observations: int = 6
    train_dataset: pd.DataFrame = consolidated_dataframe[:-test_observations]
    test_dataset: pd.DataFrame = consolidated_dataframe[-test_observations:]
    print("%s Train data: %d Test data: %d" % (user_login, len(train_dataset), len(test_dataset)))

    var_model: VAR = VAR(train_dataset)
    order_results: LagOrderResults = var_model.select_order(maxlags=max_order)
    print(order_results.summary())

    result_analysis: dict[str, Any] = {
        "var_order": set(),
        "serial_correlation": set()
    }

    for permutation_index, permutation in enumerate(itertools.permutations(VARIABLES)):
        print("Permutation %d : %s" % (permutation_index, str(permutation)))
        train_dataset: pd.DataFrame = train_dataset[list(permutation)]

        training_result: VARResults = var_model.fit(maxlags=max_order, ic=information_criterion)
        print(training_result.summary())

        var_order: int = training_result.k_ar
        result_analysis["var_order"].add(training_result.k_ar)
        if not var_order:
            logging.error("Model with 0 lags for user %s. Cannot test Granger Causality" % user_login)
            continue

        whiteness_result: WhitenessTestResults = training_result.test_whiteness()
        print(whiteness_result.summary())
        if whiteness_result.conclusion == "reject":
            logging.error("ALERT! Serial correlation found in the residuals for user %s" % user_login)
            result_analysis["serial_correlation"].add(True)
            return result_analysis

        result_analysis["serial_correlation"].add(False)

        causality_results: dict[str, bool] = check_causality(training_result)
        for test, result in causality_results.items():
            if test in result_analysis:
                result_analysis[test].add(result)
            else:
                result_analysis[test] = set()
                result_analysis[test].add(result)

        impulse_response: IRAnalysis = training_result.irf(periods=periods)
        impulse_response.plot(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_impulse_response_%i.png" % (user_login, permutation_index))

        variance_decomposition = training_result.fevd(periods=periods)
        variance_decomposition.plot(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_variance_decomposition_%i.png" % (user_login, permutation_index))

    return result_analysis


def plot_seasonal_decomposition(consolidated_dataframe: pd.DataFrame, user_login: str,
                                column: str = MERGES_PERFORMED_COLUMN) -> None:
    merges_performed_decomposition: DecomposeResult = seasonal_decompose(
        consolidated_dataframe[column],
        model='additive')

    _: Figure = merges_performed_decomposition.plot()
    plt.savefig(IMAGE_DIRECTORY + "%s_seasonal_decomposition_%s.png" % (user_login, column))


def consolidate_dataframe(es: Elasticsearch, pull_request_index: str, user_login: str,
                          calendar_interval: str) -> pd.DataFrame:
    merges_performed_dataframe: pd.DataFrame = get_merges_performed(es, pull_request_index, user_login,
                                                                    calendar_interval)
    if not len(merges_performed_dataframe):
        logging.error("User %s does not merge PRs for other developers" % user_login)
        return pd.DataFrame()

    merge_requests_dataframe: pd.DataFrame = get_merge_requests(es, pull_request_index, user_login, calendar_interval)

    requests_merged_dataframe: pd.DataFrame = get_requests_merged(es, pull_request_index, user_login, calendar_interval)
    if not len(requests_merged_dataframe):
        logging.error("User %s does not have PRs merged by other developers" % user_login)
        return pd.DataFrame()

    consolidated_dataframe: pd.DataFrame = pd.concat([merges_performed_dataframe,
                                                      merge_requests_dataframe,
                                                      requests_merged_dataframe], axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)
    consolidated_dataframe = consolidated_dataframe.rename_axis('metric', axis=1)
    return consolidated_dataframe


def analyse_user(es: Elasticsearch, pull_request_index: str, user_login: str, calendar_interval: str,
                 information_criterion: str) -> Optional[
    dict[str, Any]]:
    consolidated_dataframe = consolidate_dataframe(es, pull_request_index, user_login, calendar_interval)
    data_points: int = len(consolidated_dataframe)
    if not len(consolidated_dataframe):
        print("No data points for user %s" % user_login)
        return None

    print("Data points for user %s: %d. Calendar interval: %s" % (user_login, data_points, calendar_interval))
    plot_dataframe(consolidated_dataframe, "%s: before differencing" % user_login)

    analysis_result: dict[str, Any] = {
        "user_login": user_login,
        "data_points": data_points,
        MERGES_PERFORMED_COLUMN: consolidated_dataframe[MERGES_PERFORMED_COLUMN].sum(),
        MERGES_SUCCESSFUL_COLUMN: consolidated_dataframe[MERGES_SUCCESSFUL_COLUMN].sum()
    }

    after_differencing_data = consolidated_dataframe.diff()
    after_differencing_data = after_differencing_data.dropna()
    print("Applying 1st order differencing to the data")

    for variable in VARIABLES:
        is_stationary: bool = check_stationarity(after_differencing_data, user_login, variable)
        if not is_stationary:
            logging.error("ALERT! %s is not stationary" % variable)
            return analysis_result

    plot_seasonal_decomposition(consolidated_dataframe, user_login)
    var_results: dict[str, Any] = train_var_model(after_differencing_data[list(VARIABLES)], user_login,
                                                  information_criterion=information_criterion)
    plot_dataframe(after_differencing_data, "%s: after differencing" % user_login)

    for key, values in var_results.items():
        analysis_result[key] = " ".join([str(value) for value in values])

    return analysis_result


def start_analysis(pull_request_index: str, calendar_interval: str, information_criterion: str):
    es: Elasticsearch = elasticsearch.Elasticsearch(ELASTICSEARCH_HOST)

    all_mergers: list[str] = get_all_mergers(es, pull_request_index)
    merger_data: list[dict[str, Any]] = []
    for merger in all_mergers:
        try:
            user_analysis: dict[str, Any] = analyse_user(es, pull_request_index, merger, calendar_interval,
                                                         information_criterion)
            if user_analysis:
                merger_data.append(user_analysis)
        except Exception:
            logging.error(traceback.format_exc())
            logging.error("Cannot analyse user %s" % merger)

    consolidated_analysis: pd.DataFrame = pd.DataFrame(merger_data)
    csv_file: str = "%s_consolidated_analysis_%s.csv" % (pull_request_index, calendar_interval)
    consolidated_analysis.to_csv(csv_file)
    print("Results written in %s" % csv_file)


if __name__ == "__main__":
    start_analysis("facebook-react", "week", "aic")
