import itertools
from collections import defaultdict
from functools import partial
from typing import Callable, Mapping, Sequence, Any, Union

import numpy as np
from numpy.ma import MaskedArray

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_random_state, check_is_fitted

from molecular_cross_validation.util import split_molecules


# copy of sklearn.model_selection._check_param_grid
def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, str) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))


def adjusted_mse_loss(
    y_pred: np.ndarray, y_true: np.ndarray, a: float, b: float
) -> float:
    y_pred = ut.convert_expectations(y_pred, a, b)

    return mean_squared_error(y_pred, y_true)


def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


def adjusted_poisson_nll_loss(
    y_pred: np.ndarray, y_true: np.ndarray, a: float, b: float
) -> float:
    return poisson_nll_loss(y_pred - np.log(a) + np.log(b), y_true)


class GridSearchMCV(BaseEstimator):
    """Exhaustive search over specified parameters values for a scRNAseq denoiser.

    This class is inspired by scikit-learn's GridSearchCV, but it doesn't implement
    all of the features of that class. Some notable differences:
      - `iid` is always True and thus not an argument
      - `cv` is always a random split of UMIs
      - no parallelization, so `n_jobs` and `pre_dispatch` args are removed

    After `fit` has been called, the following attributes will be set:
    `best_params_` : the parameters for the best performing model
    `best_score_` : the MCV score of the best performing model

    :param denoiser: A function or method that takes an input array of counts and
                     returns a denoised version. This object should not retain any
                     state between calls.
    :param param_grid: Dictionary with parameters names (string) as keys and lists of
                       parameter settings to try as values, or a list of such
                       dictionaries, in which case the grids spanned by each dictionary
                       in the list are explored. This enables searching over any
                       sequence of parameter settings.
    :param data_split: Proportion of UMIs to use for denoising.
    :param overlap: Overlap factor to adjust split, if desired.
    :param n_splits: Number of times to split UMIs for a given parameterization.
    :param scoring: either `mse` or `poisson`. if `mse`, data will be sqrt transformed.
    :param random_state: If int, random_state is the seed used by the random number generator;
                         If RandomState instance, random_state is the random number generator;
                         If None, the random number generator is the RandomState instance used
                         by `np.random`.
    """
    def __init__(
        self,
        denoiser: Callable,
        param_grid: Union[Mapping, Sequence[Mapping]],
        data_split: Union[float, np.ndarray] = 0.9,
        overlap: Union[float, np.ndarray] = None,
        n_splits: int = 1,
        scoring: str = None,
        random_state: Union[int, np.random.RandomState] = None
    ):
        self.denoiser = denoiser
        self.param_grid = param_grid
        _check_param_grid(param_grid)

        self.data_split = data_split
        self.overlap = overlap or 0.0
        self.n_splits = n_splits
        self.scoring = scoring
        if scoring == "mse":
            self.loss = adjusted_mse_loss
        elif scoring == "poisson":
            self.loss = adjusted_poisson_nll_loss
        else:
            raise ValueError("'scoring' must be one of 'mse' or 'poisson'")

        self.random_state = random_state

    def fit(self, X: np.ndarray, **fit_params: Mapping[str, object]):
        """
        :param X: raw count array of UMIs. Must not be pre-processed, except for
                  optional filtering of bad cells/genes.
        :params fit_params: Additional parameters passed to the denoiser
        """

        rng = check_random_state(self.random_state)
        param_grid = ParameterGrid(self.param_grid)

        scores = defaultdict(list)

        for i in range(self.n_splits):
            umis_X, umis_Y = split_molecules(
                X, self.data_split, self.overlap, random_state=rng
            )

            if args.loss == "mse":
                umis_X = np.sqrt(umis_X)
                umis_Y = np.sqrt(umis_Y)

            for params in param_grid:
                denoised_umis = self.denoiser(umis_X, **fit_params, **params)
                scores[i].append(
                    self.loss(
                        denoised_umis, umis_Y, data_split, 1 - data_split + self.overlap
                    )
                )

        scores = [np.mean(s) for s in zip(*scores.values())]
        results = list(zip(param_grid, scores))

        best_index_ = min(range(len(results)), key=lambda i: results[i][1])
        self.best_params_ = results[best_index_][0]
        self.best_score_ = results[best_index_][1]

        return self

    def fit_transform(self, X: np.ndarray, **fit_params: Mapping[str, object]):
        self.fit(X, **fit_params)

        return self.denoiser(X, **self.best_params_)

    def transform(self, X: np.ndarray):
        check_is_fitted(self, "best_params_")

        return self.denoiser(X, **self.best_params_)
