from scipy.stats import pearsonr, spearmanr
import numpy as np


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


def spearmanr_pval(x, y):
    return spearmanr(x, y)[1]


def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)


def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)


def corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))


def generate_confidence_interval_data(repetitions, x, y, w, correlations_data, correlation_func):
    ret = []
    for i in range(repetitions):
        samples_df = correlations_data.sample(len(correlations_data[x]), replace=True, random_state=i)
        ret.append(correlation_func(samples_df[x], samples_df[y], samples_df[w]))
    return ret


def calc_ci(data, alpha, repetitions):
    data.sort()
    trim_val = int((1 - alpha) * repetitions / 2)
    del data[len(data) - trim_val:]
    del data[:trim_val]
    return min(data), max(data)


def calc_correlations(x, y, w):
    corr_data = bubble_table[[x, y, w]]
    alpha = 0.95
    repetitions = 10000

    # # Calculate weighted correlation - 1st method
    # weighted_correlation_1st_method = wpcc.wpearson(corr_data[x], corr_data[y], corr_data[w])
    # ci_data = generate_confidence_interval_data(repetitions, x, y, w, corr_data, wpcc.wpearson)
    # ci_1st_method = calc_ci(ci_data, alpha, repetitions)

    # Calculate weighted correlation - 2nd method
    weighted_correlation_2nd_method = corr(corr_data[x], corr_data[y], corr_data[w])
    ci_data = generate_confidence_interval_data(repetitions, x, y, w, corr_data, corr)
    ci_2nd_method = calc_ci(ci_data, alpha, repetitions)

    # Calculate the pearson correlation
    pearson_correlation = corr_data[y].corr(corr_data[x], method='pearson')
    pearson_p_value = corr_data[y].corr(corr_data[x], method=pearsonr_pval)

    # Calculate the spearman correlation
    spearman_correlation = corr_data[y].corr(corr_data[x], method='spearman')
    spearman_p_value = corr_data[y].corr(corr_data[x], method=spearmanr_pval)

    # print("Weighted Correlation - 1st method: {}".format(weighted_correlation_1st_method) +
    #       "; Confidence Interval (alpha={}) : {}".format(alpha, ci_1st_method))
    print("Weighted Correlation - 2nd method: {}".format(weighted_correlation_2nd_method) +
          "; Confidence Interval: (alpha={}) : {}".format(alpha, ci_2nd_method))
    print("Pearson Correlation: {} +- {}".format(pearson_correlation, pearson_p_value))
    print("Spearman Correlation: {} +- {}".format(spearman_correlation, spearman_p_value))
    print()