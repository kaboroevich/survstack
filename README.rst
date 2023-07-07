********
survstak
********

survstack is a Python implementation of the survival stacking
method proposed in *Survival stacking: casting survival
analysis as a classification problem* by Erin Craig,
Chenyang Zhong, and Robert Tibshirani (2021) [`1`_]. The package
offers both an OO and functional interface.

=====================
SurvivalStacker Class
=====================

The recommended use is the provided SurvivalStacker class.
Survival data format follows that of the `scikit-survival`_
package - a structured array with the first field indicating
the observation of an event as a boolean value, and the second
field denoting the survival time.

.. code-block:: python

    from survstack import SurvivalStacker
    from sksurv.datasets import load_breast_cancer

    X, y = load_breast_cancer()
    X = X.loc[:, X.dtypes == np.float64].values

    event_field, time_field = y.dtype.names
    print(X.shape, y.shape, y[event_field].sum())
    # (198, 78) (198,) 51

    ss = SurvivalStacker()
    X_stacked, y_stacked = ss.fit_transform(X, y)

    print(X_stacked.shape, y_stacked.shape)
    # (8117, 129) (8117,)

In the above example code, you can see the number of columns in X
increased by the number of observed events, while y became a
single column. The number of rows increases with respect to the
number of samples still under observation at each time-point.

.. code-block:: bibtex

    @article{Craig2021-or,
      title={Survival stacking: casting survival analysis as a classification problem},
      author={Craig, Erin and Zhong, Chenyang and Tibshirani, Robert},
      journal={arXiv preprint arXiv:2107.13480},
      year={2021}
    }


..  _1: https://doi.org/10.48550/arXiv.2107.13480
..  _scikit-survival: https://scikit-survival.readthedocs.io/en/stable/
