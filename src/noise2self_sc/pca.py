#!/usr/bin/env python

import numpy as np

from matplotlib import pyplot as plt

from scipy.sparse import issparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd


class LowRank(object):
    def __init__(
        self,
        rank: int = 10,
        sqrt: bool = True,
        normalize: bool = False,
        regression: bool = False,
        random_state: int = None,
    ):
        self.rank = rank
        self.sqrt = sqrt
        self.normalize = normalize
        self.regression = regression
        self.random_state = random_state

    @staticmethod
    def preprocess(X: np.ndarray, normalize: bool, sqrt: bool):
        if normalize:
            X = sklearn_normalize(X, norm="l1")

        if sqrt:
            X = X.sqrt() if issparse(X) else np.sqrt(X)

        return X

    @staticmethod
    def postprocess(X: np.ndarray, sqrt: bool):
        X = np.maximum(X, 0)

        if sqrt:
            X = np.square(X)

        return X

    def fit_transform(self, X1: np.ndarray, X2: np.ndarray):
        X1 = self.preprocess(X1, self.normalize, self.sqrt)
        X2 = self.preprocess(X2, self.normalize, self.sqrt)

        if issparse(X2):
            X2 = np.array(X2.todense())

        random_state = check_random_state(self.random_state)

        U, sigma, VT = randomized_svd(
            X1, n_components=self.rank, random_state=random_state
        )
        self.components_ = VT

        if self.regression:
            denoised = X1.dot(VT.T).dot(np.diag(1 / sigma)).dot(U.T).dot(X2)
        else:
            denoised = U.dot(np.diag(sigma).dot(VT))

        return self.postprocess(denoised, self.sqrt)

    def sweep(self, X1: np.ndarray, X2: np.ndarray, max_rank: int = 30):
        self.max_rank = max_rank

        X1 = self.preprocess(X1, self.normalize, self.sqrt)
        X2 = self.preprocess(X2, self.normalize, self.sqrt)

        if issparse(X2):
            X2 = np.array(X2.todense())

        if self.regression:
            X1_train, X1_test, X2_train, X2_test = train_test_split(
                X1, X2, test_size=0.3, random_state=0
            )
        else:
            X1_train, X1_test, X2_train, X2_test = X1, X1, X2, X2

        random_state = check_random_state(self.random_state)

        U, sigma, VT = randomized_svd(
            X1_train, n_components=self.max_rank, random_state=random_state
        )

        self.components_ = VT

        self.rank_range = np.arange(1, self.max_rank + 1)

        self.losses = np.zeros(self.max_rank)

        best_loss = np.square(X2).mean()
        best_rank = 0

        for i, rank in enumerate(self.rank_range):
            if self.regression:
                denoised = (
                    X1_test.dot(VT[:rank, :].T)
                    .dot(np.diag(1 / sigma[:rank]))
                    .dot(U[:, :rank].T)
                    .dot(X2_train)
                )
            else:
                denoised = U[:, :rank].dot(np.diag(sigma[:rank]).dot(VT[:rank, :]))

            loss = np.square(denoised - X2_test).mean()

            if loss < best_loss:
                best_rank = rank
                best_loss = loss

            self.losses[i] = loss

        print("Optimal rank: " + str(best_rank))

        self.rank = best_rank

        return best_rank, self.rank_range, self.losses


def n2s_low_rank(adata, max_rank: int = 30, plot: bool = True, **kwargs):
    model = LowRank(**kwargs)

    X = adata.X

    if issparse(X):
        X = np.array(X.todense())

    X = X.astype(np.int)

    X1 = np.random.binomial(X, 0.5)
    X2 = X - X1

    best_rank, rank_range, losses = model.sweep(X1, X2, max_rank)

    if plot:
        plt.plot(rank_range, losses)
        plt.xlabel("Rank")
        plt.ylabel("Loss")
        plt.title("Sweep Rank")
        plt.axvline(best_rank, color="k", linestyle="--")

    denoised = model.fit_transform(X1, X2)

    denoised_adata = adata.copy()
    denoised_adata.X = denoised

    return denoised_adata
