import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def hessian(X, A, lamb):
    """
    :param X: shape = (m, n). rescale X, added interception
    :param A: shape = (m, m). a diagonal matrix
    :param lamb: shape = (n, n). lambda is regularization strength
    :return H: shape = (n, n). Hessian matrix
    """
    return X.T.dot(A).dot(X) + lamb


def log_likelihood_i(x_i, y_i, w):
    return np.log(sigmoid(y_i * x_i.dot(w)))


def log_likelihood(x, y, w):
    return sum(log_likelihood_i(x_i, y_i, w) for x_i, y_i in zip(x, y))


def log_likelihood_partial_ij(x_i, y_i, w, j):
    """here i is the index of the data point,
    j the index of the derivative"""
    return (1 - sigmoid(y_i * x_i.dot(w))) * y_i * x_i[j]


def log_gradient_i(x_i, y_i, w):
    """the gradient of the log likelihood
    corresponding to the ith data point"""
    return [log_likelihood_partial_ij(x_i, y_i, w, j) for j, _ in enumerate(w)]


def log_gradient(x, y, w):
    return np.sum([log_gradient_i(x_i, y_i, w) for x_i, y_i in zip(x, y)])


class LogisticRegression:
    def __init__(self, x, y, lamb=1, tol=1e-8, iter_lim=100, fit_intercept=True):
        self.tol = tol
        self.iter_lim = iter_lim
        self.fit_intercept = fit_intercept
        reshape_x, reshape_y = self._reshape(x, y)
        self.x = self._rescale_x(reshape_x)
        self.y = self._rescale_y(reshape_y)
        self.variables = self.x.shape[1]
        self.lamb = lamb * np.diagflat(np.ones((self.variables, 1)))

    def _reshape(self, x, y):
        x = np.array(x)
        y = np.array(y)
        mx, nx = x.shape
        sy = y.shape
        if mx in sy:
            reshape_x = np.reshape(x, (mx, nx))
            reshape_y = np.reshape(y, (mx, 1))
        elif nx in sy:
            reshape_x = np.reshape(x.T, (nx, mx))
            reshape_y = np.reshape(y, (nx, 1))
        else:
            assert "dimensions not match"
        return reshape_x, reshape_y

    def _rescale_x(self, x):
        if self.fit_intercept:
            mx, nx = x.shape
            return np.concatenate((np.ones((mx, 1)), x), axis=1)
        else:
            return x

    def _rescale_y(self, y):
        max_y = y.max()
        min_y = y.min()
        return (y - (max_y + min_y) / 2.) / ((max_y - min_y) / 2.)
