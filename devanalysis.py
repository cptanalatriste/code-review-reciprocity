import itertools
import logging
import traceback
from math import ceil
from typing import Tuple, Any, Optional, List

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
    get_requests_merged, get_all_mergers, MERGES_REQUESTED_COLUMN

VARIABLES: Tuple[str, str] = (MERGES_PERFORMED_COLUMN, MERGES_SUCCESSFUL_COLUMN)
IMAGE_DIRECTORY: str = "img/"


def plot_dataframe(consolidated_dataframe: pd.DataFrame, plot_title: str) -> None:
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.style.use('fivethirtyeight')
    _ = consolidated_dataframe.plot(subplots=True, linewidth=2, fontsize=12, title=plot_title)
    plt.savefig(IMAGE_DIRECTORY + plot_title + ".png")


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


def fit_var_model(var_model: VAR, information_criterion: str, user_login: str) -> Tuple[
    VARResults, WhitenessTestResults]:
    order_results: LagOrderResults = var_model.select_order()
    candidate_order: int = order_results.selected_orders[information_criterion]

    training_result: VARResults = var_model.fit(maxlags=candidate_order)
    whiteness_result: WhitenessTestResults = training_result.test_whiteness(nlags=ceil(candidate_order * 1.5))

    while whiteness_result.conclusion == "reject" and candidate_order <= 12:
        candidate_order += 1
        logging.warning("ALERT! Serial correlation in residuals for user %s. Increasing lag order to %d" % (
            user_login, candidate_order))
        training_result: VARResults = var_model.fit(maxlags=candidate_order)
        whiteness_result: WhitenessTestResults = training_result.test_whiteness(nlags=ceil(candidate_order * 1.5))

    return training_result, whiteness_result


def train_var_model(consolidated_dataframe: pd.DataFrame, user_login: str,
                    project: str,
                    calendar_interval: str,
                    information_criterion='bic',
                    periods=24) -> dict[str, set]:
    test_observations: int = 6
    train_dataset: pd.DataFrame = consolidated_dataframe[:-test_observations]
    test_dataset: pd.DataFrame = consolidated_dataframe[-test_observations:]
    print("%s Train data: %d Test data: %d" % (user_login, len(train_dataset), len(test_dataset)))

    result_analysis: dict[str, Any] = {
        "var_order": set(),
        "serial_correlation": set()
    }

    for permutation_index, permutation in enumerate(itertools.permutations(VARIABLES)):
        print("Permutation %d : %s" % (permutation_index, str(permutation)))
        train_dataset: pd.DataFrame = train_dataset[list(permutation)]

        var_model: VAR = VAR(train_dataset)
        training_result, whiteness_result = fit_var_model(var_model, information_criterion, user_login)

        print(training_result.summary())
        print(whiteness_result.summary())

        var_order: int = training_result.k_ar
        result_analysis["var_order"].add(training_result.k_ar)

        if whiteness_result.conclusion == "reject":
            logging.error("ALERT! Serial correlation found in the residuals for user %s" % user_login)
            result_analysis["serial_correlation"].add(True)
        else:
            result_analysis["serial_correlation"].add(False)

        if not var_order:
            logging.error("Model with 0 lags for user %s. Cannot test Granger Causality" % user_login)
            continue

        causality_results: dict[str, bool] = check_causality(training_result)
        for test, result in causality_results.items():
            if test in result_analysis:
                result_analysis[test].add(result)
            else:
                result_analysis[test] = set()
                result_analysis[test].add(result)

        impulse_response: IRAnalysis = training_result.irf(periods=periods)
        impulse_response.plot(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_%s_impulse_response_%s_%i.png" % (
            user_login, project, calendar_interval, permutation_index))

        variance_decomposition = training_result.fevd(periods=periods)
        variance_decomposition.plot(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_%s_variance_decomposition_%s_%i.png" % (
            user_login, project, calendar_interval, permutation_index))

    return result_analysis


def plot_seasonal_decomposition(consolidated_dataframe: pd.DataFrame, user_login: str,
                                project: str,
                                column: str = MERGES_PERFORMED_COLUMN) -> None:
    merges_performed_decomposition: DecomposeResult = seasonal_decompose(
        consolidated_dataframe[column],
        model='additive')

    _: Figure = merges_performed_decomposition.plot()
    plt.savefig(IMAGE_DIRECTORY + "%s_%s_seasonal_decomposition_%s.png" % (user_login, project, column))


def consolidate_dataframe(es: Elasticsearch, pull_request_index: str, user_login: str,
                          calendar_interval: str) -> pd.DataFrame:
    data: list[pd.DataFrame] = []
    if MERGES_PERFORMED_COLUMN in VARIABLES:
        merges_performed_dataframe: pd.DataFrame = get_merges_performed(es, pull_request_index, user_login,
                                                                        calendar_interval)
        if not len(merges_performed_dataframe):
            logging.error("User %s does not merge PRs for other developers" % user_login)
            return pd.DataFrame()

        data.append(merges_performed_dataframe)

    if MERGES_REQUESTED_COLUMN in VARIABLES:
        merge_requests_dataframe: pd.DataFrame = get_merge_requests(es, pull_request_index, user_login,
                                                                    calendar_interval)
        data.append(merge_requests_dataframe)

    if MERGES_SUCCESSFUL_COLUMN in VARIABLES:
        requests_merged_dataframe: pd.DataFrame = get_requests_merged(es, pull_request_index, user_login,
                                                                      calendar_interval)
        if not len(requests_merged_dataframe):
            logging.error("User %s does not have PRs merged by other developers" % user_login)
            return pd.DataFrame()

        data.append(requests_merged_dataframe)

    consolidated_dataframe: pd.DataFrame = pd.concat(data, axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)
    consolidated_dataframe = consolidated_dataframe.rename_axis('metric', axis=1)
    return consolidated_dataframe


def analyse_user(es: Elasticsearch, pull_request_index: str, user_login: str, calendar_interval: str,
                 information_criterion: str,
                 project: str) -> Optional[dict[str, Any]]:
    consolidated_dataframe = consolidate_dataframe(es, pull_request_index, user_login, calendar_interval)
    data_points: int = len(consolidated_dataframe)
    if not len(consolidated_dataframe):
        print("No data points for user %s" % user_login)
        return None

    print("Data points for user %s: %d. Calendar interval: %s" % (user_login, data_points, calendar_interval))

    analysis_result: dict[str, Any] = {
        "user_login": user_login,
        "data_points": data_points,
        "index": pull_request_index
    }

    for column in VARIABLES:
        analysis_result[column] = consolidated_dataframe[column].sum()

    after_differencing_data = consolidated_dataframe.diff()
    after_differencing_data = after_differencing_data.dropna()
    print("Applying 1st order differencing to the data")

    for variable in VARIABLES:
        is_stationary: bool = check_stationarity(after_differencing_data, user_login, variable)
        if not is_stationary:
            logging.error("ALERT! %s is not stationary" % variable)
            analysis_result[variable + "_stationary"] = False
        else:
            analysis_result[variable + "_stationary"] = True

    var_results: dict[str, Any] = train_var_model(after_differencing_data[list(VARIABLES)], user_login,
                                                  project, calendar_interval,
                                                  information_criterion=information_criterion)

    try:
        plot_dataframe(consolidated_dataframe, "%s_%s_before_differencing" % (user_login, project))
        plot_dataframe(after_differencing_data, "%s_%s_after_differencing" % (user_login, project))
        plot_seasonal_decomposition(consolidated_dataframe, user_login, project)
    except ValueError:
        logging.error(traceback.format_exc())
        logging.error("Error while building diagnosis plots for user %s" % user_login)

    for key, values in var_results.items():
        analysis_result[key] = " ".join([str(value) for value in values])

    return analysis_result


def analyse_project(es: Elasticsearch, pull_request_index: str, calendar_interval: str,
                    information_criterion: str) -> pd.DataFrame:
    es.indices.refresh(index=pull_request_index)
    document_count: List[dict] = es.cat.count(index=pull_request_index, params={"format": "json"})
    documents: int = int(document_count[0]['count'])
    print("Documents on index %s: %s" % (pull_request_index, documents))

    all_mergers: list[str] = get_all_mergers(es, pull_request_index)
    merger_data: list[dict[str, Any]] = []
    for merger in all_mergers:
        try:
            user_analysis: dict[str, Any] = analyse_user(es, pull_request_index, merger, calendar_interval,
                                                         information_criterion, pull_request_index)
            if user_analysis:
                merger_data.append(user_analysis)
        except Exception:
            logging.error(traceback.format_exc())
            logging.error("Cannot analyse user %s" % merger)

    consolidated_analysis: pd.DataFrame = pd.DataFrame(merger_data)
    return consolidated_analysis
