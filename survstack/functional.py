import numpy as np

from functools import partial

ALLOWED_ENCODINGS: tuple[str, ...] = ("onehot", "continuous")

def digitize_times(values: np.ndarray, time_step: float = 1.) -> np.ndarray:
    """Generate unique time bin values that cover the input times
    and are rounded to the next time_step multiple

    :param values: an array of times
    :param time_step: a step value for which the final bins will be based
    :return: an array of unique time bins that cover the input values
    """
    min_edge = np.floor(values.min() / time_step) * time_step
    max_edge = np.nextafter(np.ceil(values.max() / time_step) * time_step + time_step, np.inf)

    full_range = np.arange(min_edge, max_edge, step=time_step)
    if time_step < 1.0:
        decimal_places = max(0, int(-np.floor(np.log10(time_step))))
        full_range = np.round(full_range, decimals=decimal_places)

    obs_time_indices = np.searchsorted(full_range, values, side='left')
    obs_time_indices = np.clip(obs_time_indices, 0, full_range.size - 1)

    times = full_range[obs_time_indices]
    times = np.unique(times)
    return times


def stack_timepoints(X: np.ndarray, y: np.ndarray, times: np.ndarray,
                     time_encoding: str = "onehot") -> tuple[np.ndarray, np.ndarray]:
    """Generate a survival stacked dataset and the accompanying binary outcome
    for a survival dataset for all given timepoints.

    :param X: training input samples
    :param y: survival observations in the format of a 2d array, where
    the first column is the time and second column is the event
    :param times: array of time points on which to create risk sets
    :param time_encoding: encoding to use for timepoints {"onehot","continuous"}
    :return: a tuple containing the survival stacked dataset and a binary
    outcome
    """
    if time_encoding not in ALLOWED_ENCODINGS:
        raise ValueError(f"time_encoding must be one of {ALLOWED_ENCODINGS!r} got {time_encoding!r}")
    stack_funcs = {
        "onehot": _stack_timepoint_onehot,
        "continuous": _stack_timepoint_continuous,
    }
    num_times = times.shape[0]
    stacked_events = list(zip(*[stack_funcs[time_encoding](X, y, times, t) for t in range(num_times)]))
    X_stacked = np.vstack(stacked_events[0])
    y_stacked = np.concatenate(stacked_events[1])
    return X_stacked, y_stacked


def _stack_timepoint_onehot(X: np.ndarray, y: np.ndarray, times: np.ndarray,
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
    y_bins = np.searchsorted(times, y[time_field], side='right') - 1
    y_bins = np.clip(y_bins, 0, times.size - 1)
    X_i = X[y_bins >= i, :]
    y_i = y[y_bins >= i]
    X_risk = np.zeros((X_i.shape[0], times.shape[0]))
    X_risk[:, i] = 1
    y_outcome = (y_bins[y_bins >= i] == i) & (y_i[event_field])
    y_outcome = y_outcome.astype(int)
    X_new = np.hstack((X_i, X_risk))
    return X_new, y_outcome


def _stack_timepoint_continuous(X: np.ndarray, y: np.ndarray, times: np.ndarray,
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
    y_bins = np.searchsorted(times, y[time_field], side='right') - 1
    y_bins = np.clip(y_bins, 0, times.size - 1)
    risk_mask = y_bins >= i
    X_i = X[risk_mask, :]
    y_i = y[risk_mask]
    y_outcome = (y_bins[risk_mask] == i) & (y_i[event_field])
    y_outcome = y_outcome.astype(int)
    time_feature = np.full((X_i.shape[0], 1), times[i])
    X_new = np.hstack((X_i, time_feature))
    return X_new, y_outcome


def stack_eval(X: np.ndarray, times: np.ndarray, time_encoding: str = "onehot", normalize: bool = False) -> np.ndarray:
    """Generate a predictor matrix for outcome prediction for given times. This
    is to be used for evaluation of a model, not for training.

    :param X: Survival input samples
    :param times: array of time points on which to create risk sets
    :param time_encoding: encoding to use for timepoints {"onehot","continuous"}
    :return: a generalized predictor matrix for input X
    """
    if time_encoding not in ALLOWED_ENCODINGS:
        raise ValueError(f"time_encoding must be one of {ALLOWED_ENCODINGS!r} got {time_encoding!r}")
    encode_funcs = {
        'onehot': _encode_onehot,
        'continuous': partial(_encode_continuous, normalize=normalize)
    }
    X_cov = np.repeat(X, times.shape[0], axis=0)
    X_risk = encode_funcs[time_encoding](times, X.shape[0])
    X_new = np.hstack((X_cov, X_risk))
    return X_new


def _encode_onehot(times: np.ndarray, n_samples: int) -> np.ndarray:
    """Tile an identity matrix for one‐hot time encoding."""
    return np.tile(np.eye(times.shape[0]), (n_samples, 1))


def _encode_continuous(times: np.ndarray, n_samples: int, normalize: bool = False) -> np.ndarray:
    """Tile a (optionally normalized) time vector for continuous time encoding."""
    if normalize:
        times = times / times[-1]
    return np.tile(times, n_samples).reshape(-1, 1)


def cumulative_hazard_function(estimates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate cumulative_hazard from hazard predictions.

    :param estimates: estimates as returned from a model trained on
    an evaluation set
    :param times: array of time points on which the hazard was estimated
    :return: array of survival probabilities for each sample over time.
    """
    hazard_preds = estimates.reshape(-1, times.shape[0])
    cum_hazard = np.cumsum(hazard_preds, axis=1)
    return cum_hazard

def survival_function(estimates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate survival probability curves from hazard predictions.

    :param estimates: estimates as returned from a model trained on
    an evaluation set
    :param times: array of time points on which the hazard was estimated
    :return: array of survival probabilities for each sample over time.
    """
    hazard_preds = estimates.reshape(-1, times.shape[0])
    surv = np.cumprod(1 - hazard_preds, axis=1)
    return surv


def risk_score(estimates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculate risk score from stacked survival estimates.

    :param estimates: estimates as returned from a model trained on
    an evaluation set
    :param times: array of time points on which the hazard was estimated
    :return: the risk score
    """
    cum_hazard = cumulative_hazard_function(estimates, times)
    risk = cum_hazard.sum(axis=1)
    return risk
