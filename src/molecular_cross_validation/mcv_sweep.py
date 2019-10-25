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

import molecular_cross_validation.util as ut


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


def poisson_nll_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return (y_pred - y_true * np.log(y_pred + 1e-6)).mean()


class GridSearchMCV(BaseEstimator):
    """Exhaustive search over specified parameters values for a scRNAseq denoiser.

    This class is inspired by scikit-learn's GridSearchCV, but it doesn't implement
    all of the features of that class. Some notable differences:
      - `iid` is always True and thus not an argument
      - `cv` is always a random split of UMIs
      - no parallelization, so `n_jobs` and `pre_dispatch` args are removed

    After `fit` has been called, the following attributes will be set:
    `best_params_` : the parameters for the best performing model
    `best_loss_` : the MCV score of the best performing model

    :param denoiser: A function or method that takes an input array of counts and
                     returns a denoised version. This object should not retain any
                     state between calls.
    :param param_grid: Dictionary with parameters names (string) as keys and lists of
                       parameter settings to try as values, or a list of such
                       dictionaries, in which case the grids spanned by each dictionary
                       in the list are explored. This enables searching over any
                       sequence of parameter settings.
    :param data_split: Proportion of UMIs to use for denoising.
    :param sample_ratio: Estimated ratio of counts in the sample compared to the original cells.
    :param n_splits: Number of times to split UMIs for a given parameterization.
    :param loss: either `mse` or `poisson`.
    :param transformation: Transformation to apply to count matrix before denoising.
                           Either `None`, `sqrt`, or an arbitrary function. If a
                           function is used, `data_split` must be 0.5.
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
        sample_ratio: Union[float, np.ndarray] = None,
        n_splits: int = 1,
        loss: str = None,
        transformation: Union[str, Callable] = None,
        random_state: Union[int, np.random.RandomState] = None
    ):
        self.denoiser = denoiser
        self.param_grid = param_grid
        _check_param_grid(param_grid)

        self.n_splits = n_splits

        if sample_ratio is None:
            self.data_split = data_split
            self.data_split_complement = 1.0 - data_split
            self.overlap = 0.0
        else:
            self.data_split, self.data_split_complement, self.overlap = (
                ut.overlap_correction(data_split, sample_ratio)
            )

        if loss == "mse":
            self.loss = mean_squared_error
        elif loss == "poisson":
            self.loss = poisson_nll_loss
            if transformation is not None:
                raise ValueError("Transformations only apply to 'mse' loss.")
        else:
            raise ValueError("'loss' must be one of 'mse' or 'poisson'")

        if transformation is None:
            self.transformation = lambda x: x
        elif transformation == 'sqrt':
            self.transformation = np.sqrt
        elif callable(transformation):
            self.transformation = transformation
        else:
            raise ValueError("Unknown value for 'transformation'")

        if data_split == 0.5:
            self.conversion = lambda x: x
        else:
            if transformation is None:
                self.conversion = lambda x: (
                    x * self.data_split_complement / self.data_split
                )
            elif transformation == 'sqrt':
                self.conversion = lambda x: ut.convert_expectations(
                    x, self.data_split, self.data_split_complement
                )
            else:
                raise NotImplementedError(
                    "Expectation conversion not implemented for arbitrary transformations."
                )

        self.random_state = random_state

    def fit(self, X: np.ndarray, **fit_params: Mapping[str, object]):
        """
        :param X: raw count array of UMIs. Must not be pre-processed, except for
                  optional filtering of bad cells/genes.
        :params fit_params: Additional parameters passed to the denoiser
        """

        rng = check_random_state(self.random_state)
        param_grid = ParameterGrid(self.param_grid)

        losses = defaultdict(list)

        for i in range(self.n_splits):
            umis_X, umis_Y = ut.split_molecules(
                X, self.data_split, self.overlap, random_state=rng
            )

            umis_X = self.transformation(umis_X)
            umis_Y = self.transformation(umis_Y)

            for params in param_grid:
                denoised_umis = self.denoiser(umis_X, **fit_params, **params)
                converted_denoised_umis = self.conversion(denoised_umis)
                losses[i].append(self.loss(converted_denoised_umis, umis_Y))

        losses = [np.mean(s) for s in zip(*losses.values())]

        best_index_ = np.argmin(losses)
        self.best_params_ = param_grid[best_index_]
        self.best_loss_ = losses[best_index_]

        self.cv_results_ = defaultdict(list)
        self.cv_results_["mcv_loss"] = losses

        for params in param_grid:
            for k in params:
                self.cv_results_[k].append(params[k])

        return self

    def fit_transform(self, X: np.ndarray, **fit_params: Mapping[str, object]):
        self.fit(X, **fit_params)

        return self.denoiser(X, **self.best_params_)

    def transform(self, X: np.ndarray):
        check_is_fitted(self, "best_params_")

        return self.denoiser(X, **self.best_params_)
