import numpy as np


def digitize_times(values: np.ndarray, time_step: float = 1.) -> np.ndarray:
    """Generate unique time bin values that cover the input times
    and are rounded to the next time_step multiple

    :param values: an array times
    :param time_step: a step value for which the final bins will be based
    :return: an array of unique time bins that cover the input values
    """
    full_range = np.arange(_floor_base(np.min(values) - time_step, time_step),
                           np.max(values) + time_step,
                           step=time_step)
    times = full_range[np.digitize(values, full_range, right=True)]
    times = np.unique(times)
    return times


def stack_timepoints(X: np.ndarray, y: np.ndarray, times: np.ndarray) -> \
        tuple[np.ndarray, np.ndarray]:
    """Generate a survival stacked dataset and the accompanying binary outcome
    for a survival dataset for all given timepoints.

    :param X: training input samples
    :param y: survival observations in the format of a 2d array, where
    the first column is the time and second column is the event
    :param times: array of time points on which to create risk sets
    :return: a tuple containing the survival stacked dataset and a binary
    outcome
    """
    num_times = times.shape[0]
    stacked_events = list(zip(*[_stack_timepoint(X, y, times, t)
                                for t in np.arange(num_times)]))
    X_stacked = np.vstack(stacked_events[0])
    y_stacked = np.concatenate(stacked_events[1])
    return X_stacked, y_stacked


def _stack_timepoint(X: np.ndarray, y: np.ndarray, times: np.ndarray,
                     i: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate the predictor matrix and response vector for a survival dataset
    at a specific time-point `times[i]`.

    :param X: training input samples
    :param y: structured array with two fields. The binary event indicator
        as first field, and time of event or time of censoring as second field.
    :param times: array of time points on which to create risk sets
    :param i: index of array `times` at which to construct the dataset
    :return: a tuple containing the predictor matrix and response vector
    """
    event_field, time_field = y.dtype.names
    y_bins = np.digitize(y[time_field], times, right=True)
    X_i = X[y_bins >= i, :]
    y_i = y[y_bins >= i]
    X_risk = np.zeros((X_i.shape[0], times.shape[0]))
    X_risk[:, i] = 1
    y_outcome = (
            (np.digitize(y_i[time_field], times, right=True) == i) &
            (y_i[event_field])
    ).astype(int)
    X_new = np.hstack((X_i, X_risk))
    return X_new, y_outcome


def stack_eval(X: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Generate a predictor matrix for outcome prediction for given times. This
    is to be used for evaluation of a model, not for training.

    :param X: Survival input samples
    :param times: array of time points on which to create risk sets
    :return: a generalized predictor matrix for input X
    """
    X_cov = np.repeat(X, times.shape[0], axis=0)
    X_risk = np.tile(np.eye(times.shape[0]), (X.shape[0], 1))
    X_new = np.hstack((X_cov, X_risk))
    return X_new


def cumulative_hazard_function(estimates: np.ndarray,
                               times: np.ndarray) -> np.ndarray:
    """Calculate the cumulative hazard function from the stacked survival
    estimates.

    :param estimates: estimates as returned from a model trained on
    an evaluation set
    :param times: array of time points on which the hazard was estimated
    :return: the cumulative risk matrix for the fitted time-points
    """
    surv_curve = np.cumprod(1 - estimates.reshape(-1, times.shape[0]), axis=1)
    return 1 - surv_curve


def risk_score(estimates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate risk score from stacked survival estimates.

    :param estimates: estimates as returned from a model trained on
    an evaluation set
    :param times: array of time points on which the hazard was estimated
    :return: the risk score
    """
    chf = cumulative_hazard_function(estimates, times)
    return chf.sum(axis=1)


def _floor_base(x: float, base: float) -> float:
    """Return the floor of the input to the nearest multiple of given base

    :param x: input data
    :param base: the base of the floor calculations
    :return: the floor of the input data
    """
    return base * np.floor(x / base)
