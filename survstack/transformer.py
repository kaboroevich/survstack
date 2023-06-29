from typing import Optional, Tuple
from numpy.typing import NDArray
import numpy as np

import functional as F


class SurvivalStacker:
    """Casts a survival analysis problem as a classification problem as
    proposed in Craig E., et al. 2021 (arXiv:2107.13480)
    """

    def __init__(self, times: Optional[NDArray] = None) -> None:
        """Generate a SurvivalStacker instance

        :param times: array of time points on which to create risk sets
        """
        self.times = times

    def fit(self, X: NDArray, y: NDArray,
            time_step: Optional[float] = None):
        """Generate the risk time points

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        :param time_step: a base multiple on which to bin times. If none, the
            times for all observed events are used.
        :return: self
        """
        event_field, time_field = y.dtype.names
        event_times = np.unique(y[time_field][y[event_field]])
        if time_step is None:
            self.times = event_times
        else:
            self.times = F.digitize_times(event_times, time_step)
        return self

    def transform(self, X: NDArray, y: Optional[NDArray] = None,
                  eval: bool = False) -> Tuple[NDArray, Optional[NDArray]]:
        """Convert the input survival dataset to a stacked survival dataset

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        :param eval: if set to False (default), the stacked survival dataset is
        constructed for training. If set to True, the returned dataset is
        constructed for evaluation.
        :return: a tuple containing the predictor matrix and response vector
        """
        if eval:
            X_stacked = F.stack_eval(X, self.times)
            y_stacked = None
        else:
            X_stacked, y_stacked = F.stack_timepoints(X, y, self.times)
        return X_stacked, y_stacked

    def fit_transform(self, X: NDArray, y: NDArray):
        """Fit to data, then transform it.

        :param X: survival input samples
        :param y: structured array with two fields. The binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        :return: a tuple containing the predictor matrix and response vector
        """
        self.fit(X, y)
        return self.transform(X, y, eval=False)

    def cumulative_hazard_function(self, estimates: NDArray):
        """Calculate the cumulative hazard function from the stacked survival
        estimates.

        :param estimates: estimates as returned from a model trained on
        an evaluation set
        :return: a cumulative risk matrix for the fitted time-points
        """
        return F.cumulative_hazard_function(estimates, self.times)

    def risk_score(self, estimates: NDArray):
        """Calculate risk score from stacked survival estimates.

        :param estimates: estimates as returned from a model trained on
        an evaluation set
        :return: the risk score
        """
        return F.risk_score(estimates, self.times)