import os
import sys

from scipy.stats import percentileofscore

sys.path.append(
    os.path.abspath("/ocean/projects/asc170022p/shg121/PhD/ICLR-2022/codebase")
)
import time
import numpy as np
import warnings
from sklearn.metrics import roc_auc_score
import utils

warnings.filterwarnings("ignore")


def score_stat_ci(
        y_true,
        y_preds,
        score_fun,
        stat_fun=np.mean,
        sample_weight=None,
        n_bootstraps=2000,
        confidence_level=0.95,
        seed=None,
        reject_one_class_samples=True,
):
    """
    Compute confidence interval for given statistic of a score function based on labels and predictions using
    bootstrapping.
    :param y_true: 1D list or array of labels.
    :param y_preds: A list of lists or 2D array of predictions corresponding to elements in y_true.
    :param score_fun: Score function for which confidence interval is computed. (e.g. sklearn.metrics.accuracy_score)
    :param stat_fun: Statistic for which confidence interval is computed. (e.g. np.mean)
    :param sample_weight: 1D list or array of sample weights to pass to score_fun, see e.g. sklearn.metrics.roc_auc_score.
    :param n_bootstraps: The number of bootstraps. (default: 2000)
    :param confidence_level: Confidence level for computing confidence interval. (default: 0.95)
    :param seed: Random seed for reproducibility. (default: None)
    :param reject_one_class_samples: Whether to reject bootstrapped samples with only one label. For scores like AUC we
    need at least one positive and one negative sample. (default: True)
    :return: Mean score statistic evaluated on labels and predictions, lower confidence interval, upper confidence
    interval, array of bootstrapped scores.
    """

    y_true = np.array(y_true)
    y_preds = np.atleast_2d(y_preds)
    assert all(len(y_true) == len(y) for y in y_preds)

    np.random.seed(seed)
    scores = []
    for i in range(n_bootstraps):
        readers = np.random.randint(0, len(y_preds), len(y_preds))
        indices = np.random.randint(0, len(y_true), len(y_true))
        if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
            continue
        reader_scores = []
        for r in readers:
            if sample_weight is not None:
                reader_scores.append(
                    score_fun(
                        y_true[indices],
                        y_preds[r][indices],
                        sample_weight=sample_weight[indices],
                    )
                )
            else:
                reader_scores.append(score_fun(y_true[indices], y_preds[r][indices]))
        scores.append(stat_fun(reader_scores))

    mean_score = np.mean(scores)
    sorted_scores = np.array(sorted(scores))
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = sorted_scores[int(round(alpha * len(sorted_scores)))]
    ci_upper = sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))]
    return mean_score, ci_lower, ci_upper, scores


def get_dict_CI(path, concepts, data_GT, data_PRED):
    score_fun = roc_auc_score
    n_bootstraps = 2000
    confidence_level = 0.95
    reject_one_class_samples = True
    _dict_scores = {}
    start = time.time()
    for i in range(len(concepts)):
        score = score_fun(data_GT[:, i], data_PRED[:, i])
        mean_score, ci_lower, ci_upper, scores = score_stat_ci(
            y_true=data_GT[:, i],
            y_preds=data_PRED[:, i],
            score_fun=score_fun,
            sample_weight=None,
            n_bootstraps=n_bootstraps,
            confidence_level=confidence_level,
            seed=0,
            reject_one_class_samples=reject_one_class_samples,
        )
        _dict_scores[i] = {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "score": score,
            "mean_score": mean_score,
        }
        print("==================================")
        print(concepts[i])
        print(
            f"ci_lower: {ci_lower}, ci_upper: {ci_upper}, "f"score: {score}, mean_score: {mean_score}"
        )
        print("==================================")

    done = time.time()
    elapsed = done - start
    print("Total time: " + str(elapsed) + " secs")
    utils.dump_in_pickle(output_path=path, file_name="CI_concepts.pkl", stats_to_dump=_dict_scores)
    return _dict_scores


def cal_p_value(path, concepts, data_GT, data_PRED, _dict_scores, key="ci_lower"):
    p_list = []
    for ii in range(len(concepts)):
        y_preds1 = np.atleast_2d(data_PRED[:, ii])
        mean_score = _dict_scores[ii][key]
        y_true = data_GT[:, ii]
        n_bootstraps = 2000
        sample_weight = None
        two_tailed = True
        reject_one_class_samples = True
        z = []
        start = time.time()
        for i in range(n_bootstraps):
            readers1 = np.random.randint(0, len(y_preds1), len(y_preds1))
            indices = np.random.randint(0, len(y_true), len(y_true))
            if reject_one_class_samples and len(np.unique(y_true[indices])) < 2:
                continue
            reader1_scores = []
            for r in readers1:
                if sample_weight is not None:
                    reader1_scores.append(
                        roc_auc_score(
                            y_true[indices],
                            y_preds1[r][indices],
                            sample_weight=sample_weight[indices],
                        )
                    )
                else:
                    reader1_scores.append(
                        roc_auc_score(y_true[indices], y_preds1[r][indices])
                    )
            score1 = np.mean(reader1_scores)
            reader2_scores = []
            score2 = mean_score
            if key == "ci_lower":
                z.append(np.subtract(score1, score2))
            else:
                z.append(np.subtract(score2, score1))
        p = percentileofscore(z, 0.0, kind="weak") / 100.0
        if two_tailed:
            p *= 2.0

        print(f"i: {ii}, {concepts[ii]}, p-value: {p}")
        done = time.time()
        elapsed = done - start
        print("Total time: " + str(elapsed) + " secs")
        p_list.append(p)

    utils.dump_in_pickle(output_path=path, file_name=f"p_value_using_{key}_concepts.pkl", stats_to_dump=p_list)
    return p_list
