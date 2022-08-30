from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def nearest_neighbors(X, Y):
    distances = np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None] - 2 * np.dot(X, Y.T)
    indices = np.argmin(distances, axis=1)
    plt.plot(X[:, 0], X[:, 1], "o")
    plt.plot(Y[:, 0], Y[:, 1], "o")
    for i, j in enumerate(indices):
        x = X[i]
        y = Y[j]
        plt.plot([x[0], y[0]], [x[1], y[1]])
    plt.show()
    return Y[indices]


def rigid_point_set_registration(source: NDArray[np.float64], target: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    _, d = source.shape
    assert d == target.shape[1]
    R = np.eye(d)
    t = np.zeros((2, 1))
    X = source
    for _ in range(20):
        Y = nearest_neighbors(X, target)
        x0 = np.mean(X, axis=0)
        y0 = np.mean(Y, axis=0)

        H = Y.T @ X

        u, _, vh = np.linalg.svd(H)
        det = np.linalg.det(u) * np.linalg.det(vh)

        S = np.eye(d)
        if not np.isclose(det, 1):
            S[d - 1, d - 1] *= -1

        R = u @ S @ vh
        t = y0 - R @ x0
        print(t)
        X = X @ R.T + t
        plt.plot(X[:, 0], X[:, 1], "o")
        plt.plot(target[:, 0], target[:, 1], "o")
        plt.show()
    return R, t
