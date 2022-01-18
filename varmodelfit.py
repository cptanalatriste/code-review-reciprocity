import logging
from math import sqrt
from typing import Tuple

import pandas as pd
from statsmodels.sandbox.tsa.varma import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.hypothesis_test_results import WhitenessTestResults, NormalityTestResults
from statsmodels.tsa.vector_ar.var_model import VARResults, LagOrderResults


def check_stationarity(consolidated_dataframe: pd.DataFrame, user_login: str, data_column: str,
                       threshold: float = 0.05) -> bool:
    # noinspection PyTypeChecker
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


def get_lags_for_whiteness_test(user_login: str, sample_size: int, candidate_order) -> int:
    lags_for_whiteness: int = max(round(sqrt(sample_size)), candidate_order + 1)
    logging.info(
        "User {}: Portmanteau test using lags {} for VAR({}) and {} samples".format(user_login, lags_for_whiteness,
                                                                                    candidate_order, sample_size))
    return lags_for_whiteness


def fit_var_model(var_model: VAR, information_criterion: str, user_login: str, sample_size: int) -> Tuple[
    VARResults, WhitenessTestResults, NormalityTestResults, LagOrderResults]:
    order_results: LagOrderResults = var_model.select_order()
    candidate_order: int = order_results.selected_orders[information_criterion]

    training_result: VARResults = var_model.fit(maxlags=candidate_order)
    whiteness_result: WhitenessTestResults = training_result.test_whiteness(
        nlags=get_lags_for_whiteness_test(user_login, sample_size, candidate_order))
    normality_result: NormalityTestResults = training_result.test_normality()

    while whiteness_result.conclusion == "reject" and candidate_order < 12:
        candidate_order += 1
        logging.warning("ALERT! Serial correlation in residuals for user %s. Increasing lag order to %d" % (
            user_login, candidate_order))
        training_result: VARResults = var_model.fit(maxlags=candidate_order)
        whiteness_result: WhitenessTestResults = training_result.test_whiteness(
            nlags=get_lags_for_whiteness_test(user_login, sample_size, candidate_order))
        normality_result: NormalityTestResults = training_result.test_normality()

    print(training_result.summary())
    print(whiteness_result.summary())
    print(normality_result.summary())

    return training_result, whiteness_result, normality_result, order_results
