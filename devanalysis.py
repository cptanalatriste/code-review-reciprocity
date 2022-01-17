import itertools
import logging
import traceback
from typing import Tuple, Any, Optional, List

import matplotlib.pyplot as plt
import pandas as pd
from elasticsearch import Elasticsearch
from matplotlib.figure import Figure
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult

from aggregation import PRS_REVIEWED_AND_MERGED, PRS_AUTHORED_AND_MERGED, get_prs_reviewed_and_merged, get_prs_authored, \
    get_prs_authored_and_merged, get_all_mergers, PRS_AUTHORED
from config import IMAGE_DIRECTORY, TEXT_DIRECTORY
from structuralanalysis import do_structural_analysis
from varmodelfit import check_stationarity, fit_var_model


def plot_dataframe(consolidated_dataframe: pd.DataFrame, plot_title: str) -> None:
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.style.use('fivethirtyeight')
    _ = consolidated_dataframe.plot(subplots=True, linewidth=2, fontsize=12, title=plot_title)
    plt.savefig(IMAGE_DIRECTORY + plot_title + ".png")


def train_var_model(consolidated_dataframe: pd.DataFrame, user_login: str, variables: Tuple, project: str,
                    calendar_interval: str, information_criterion='bic', periods=24) -> dict[str, set]:
    test_observations: int = 6
    train_dataset: pd.DataFrame = consolidated_dataframe[:-test_observations]
    test_dataset: pd.DataFrame = consolidated_dataframe[-test_observations:]
    train_sample_size: int = len(train_dataset)
    print("%s Train data: %d Test data: %d" % (user_login, train_sample_size, len(test_dataset)))

    var_order_key: str = "var_order"
    serial_correlation_key: str = "serial_correlation"
    residual_white_noise_key: str = "residual_white_noise"

    result_analysis: dict[str, Any] = {
        "train_sample_size": [train_sample_size],
        var_order_key: set(),
        serial_correlation_key: set(),
        residual_white_noise_key: set()
    }

    for permutation_index, permutation in enumerate(itertools.permutations(variables)):
        print("Permutation %d : %s" % (permutation_index, str(permutation)))
        train_dataset: pd.DataFrame = train_dataset[list(permutation)]

        var_model: VAR = VAR(train_dataset)
        training_result, whiteness_result, normality_result, var_order_result = fit_var_model(var_model,
                                                                                              information_criterion,
                                                                                              user_login,
                                                                                              train_sample_size)

        user_report_file: str = TEXT_DIRECTORY + "user_{}_permutation_{}_analysis_results.txt".format(user_login,
                                                                                                      permutation_index)
        with open(user_report_file, "a") as file:
            file.truncate()
            file.write(str(var_order_result.summary()) + "\n")
            file.write(str(training_result.summary()) + "\n")
            file.write(str(whiteness_result.summary()) + "\n")
            file.write(str(normality_result.summary()) + "\n")

        result_analysis[var_order_key].add(training_result.k_ar)

        if whiteness_result.conclusion == "reject":
            logging.error("ALERT! Serial correlation found in the residuals for user %s" % user_login)
            result_analysis[serial_correlation_key].add(True)
        else:
            result_analysis[serial_correlation_key].add(False)

        if normality_result.conclusion == "reject":
            logging.error("ALERT! Residuals are NOT Gaussian white noise for user %s" % user_login)
            result_analysis[residual_white_noise_key].add(False)
        else:
            result_analysis[residual_white_noise_key].add(True)

        causality_results: dict[str, bool] = do_structural_analysis(variables, training_result, periods, user_login,
                                                                    project, calendar_interval, permutation_index)

        for test, result in causality_results.items():
            if test in result_analysis:
                result_analysis[test].add(result)
            else:
                result_analysis[test] = set()
                result_analysis[test].add(result)

    return result_analysis


def plot_seasonal_decomposition(consolidated_dataframe: pd.DataFrame, user_login: str,
                                project: str,
                                column: str = PRS_REVIEWED_AND_MERGED) -> None:
    merges_performed_decomposition: DecomposeResult = seasonal_decompose(
        consolidated_dataframe[column],
        model='additive')

    _: Figure = merges_performed_decomposition.plot()
    plt.savefig(IMAGE_DIRECTORY + "%s_%s_seasonal_decomposition_%s.png" % (user_login, project, column))


def consolidate_dataframe(es: Elasticsearch, pull_request_index: str, user_login: str, variables: Tuple,
                          calendar_interval: str) -> pd.DataFrame:
    data: list[pd.DataFrame] = []
    if PRS_REVIEWED_AND_MERGED in variables:
        merges_performed_dataframe: pd.DataFrame = get_prs_reviewed_and_merged(es, pull_request_index, user_login,
                                                                               calendar_interval)
        if not len(merges_performed_dataframe):
            logging.error("User %s does not merge PRs for other developers" % user_login)
            return pd.DataFrame()

        data.append(merges_performed_dataframe)

    if PRS_AUTHORED in variables:
        merge_requests_dataframe: pd.DataFrame = get_prs_authored(es, pull_request_index, user_login,
                                                                  calendar_interval)
        data.append(merge_requests_dataframe)

    if PRS_AUTHORED_AND_MERGED in variables:
        requests_merged_dataframe: pd.DataFrame = get_prs_authored_and_merged(es, pull_request_index, user_login,
                                                                              calendar_interval)
        if not len(requests_merged_dataframe):
            logging.error("User %s does not have PRs merged by other developers" % user_login)
            return pd.DataFrame()

        data.append(requests_merged_dataframe)

    consolidated_dataframe: pd.DataFrame = pd.concat(data, axis=1)
    consolidated_dataframe = consolidated_dataframe.fillna(0)
    consolidated_dataframe = consolidated_dataframe.rename_axis('metric', axis=1)
    return consolidated_dataframe


def analyse_user(es: Elasticsearch, pull_request_index: str, user_login: str, variables: Tuple, calendar_interval: str,
                 information_criterion: str,
                 project: str) -> Optional[dict[str, Any]]:
    consolidated_dataframe: pd.DataFrame = consolidate_dataframe(es, pull_request_index, user_login, variables,
                                                                 calendar_interval)
    data_points: int = len(consolidated_dataframe)
    if not len(consolidated_dataframe):
        print("No data points for user %s on index %s" % (user_login, pull_request_index))
        return None

    print("Data points for user %s: %d. Calendar interval: %s" % (user_login, data_points, calendar_interval))

    analysis_result: dict[str, Any] = {
        "user_login": user_login,
        "data_points": data_points,
        "index": pull_request_index
    }

    for column in variables:
        metric_series: pd.Series = consolidated_dataframe[column]
        analysis_result[column] = metric_series.sum()
        analysis_result[column + "_active"] = len(metric_series[metric_series > 0])

    after_differencing_data = consolidated_dataframe.diff()
    after_differencing_data = after_differencing_data.dropna()
    print("Applying 1st order differencing to the data")

    for variable in variables:
        is_stationary: bool = check_stationarity(after_differencing_data, user_login, variable)
        if not is_stationary:
            logging.error("ALERT! %s is not stationary" % variable)
            analysis_result[variable + "_stationary"] = False
        else:
            analysis_result[variable + "_stationary"] = True

    var_results: dict[str, Any] = train_var_model(after_differencing_data[list(variables)], user_login, variables,
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


def analyse_project(es: Elasticsearch, pull_request_index: str, calendar_interval: str, variables: Tuple,
                    information_criterion: str) -> Tuple[int, pd.DataFrame]:
    es.indices.refresh(index=pull_request_index)
    # noinspection PyTypeChecker
    document_count: List[dict] = es.cat.count(index=pull_request_index, params={"format": "json"})
    documents: int = int(document_count[0]['count'])
    print("Documents on index %s: %s" % (pull_request_index, documents))

    all_mergers: list[str] = get_all_mergers(es, pull_request_index)
    merger_data: list[dict[str, Any]] = []
    for user_login in all_mergers:
        try:
            user_analysis: dict[str, Any] = analyse_user(es, pull_request_index, user_login, variables,
                                                         calendar_interval,
                                                         information_criterion, pull_request_index)
            if user_analysis:
                user_analysis['index_documents'] = documents
                merger_data.append(user_analysis)
        except Exception:
            logging.error(traceback.format_exc())
            logging.error("Cannot analyse user %s" % user_login)

    consolidated_analysis: pd.DataFrame = pd.DataFrame(merger_data)
    return documents, consolidated_analysis
