import logging
import traceback
from typing import Tuple, Any

from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.hypothesis_test_results import CausalityTestResults
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.var_model import VARResults

from config import TEXT_DIRECTORY, IMAGE_DIRECTORY


def check_causality(variables: Tuple, training_result: VARResults, user_login: str, permutation_index: int,
                    causality_threshold=0.05) -> dict[str, bool]:
    test_results: dict[str, Any] = {}
    for cause_data_column in variables:
        for effect_data_column in variables:
            if cause_data_column != effect_data_column:
                causality_results: CausalityTestResults = training_result.test_causality(causing=cause_data_column,
                                                                                         caused=effect_data_column,
                                                                                         kind='wald',
                                                                                         signif=causality_threshold)

                with open(TEXT_DIRECTORY + "user_{}_permutation_{}_analysis_results.txt".format(user_login,
                                                                                                permutation_index),
                          "a") as file:
                    file.write(str(causality_results.summary()) + "\n")

                granger_causality: bool = causality_results.conclusion == "reject"
                test_results[cause_data_column + "->" + effect_data_column] = granger_causality
                print(causality_results.summary())

    return test_results


def do_structural_analysis(variables: Tuple, training_result: VARResults, periods: int,
                           user_login: str, project: str, calendar_interval: str, permutation_index: int) -> dict[
    str, bool]:
    causality_results: dict[str, bool] = {}
    try:
        causality_results = check_causality(variables, training_result, user_login, permutation_index)

        impulse_response: IRAnalysis = training_result.irf(periods=periods)
        impulse_response.plot(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_%s_impulse_response_%s_%i.png" % (
            user_login, project, calendar_interval, permutation_index))
        impulse_response.plot_cum_effects(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_%s_cumulative_response_%s_%i.png" % (
            user_login, project, calendar_interval, permutation_index))

        variance_decomposition = training_result.fevd(periods=periods)
        variance_decomposition.plot(figsize=(15, 15))
        plt.savefig(IMAGE_DIRECTORY + "%s_%s_variance_decomposition_%s_%i.png" % (
            user_login, project, calendar_interval, permutation_index))

    except Exception:
        logging.error(traceback.format_exc())
        logging.error("Cannot do structural analysis for user %s" % user_login)
    finally:
        return causality_results
