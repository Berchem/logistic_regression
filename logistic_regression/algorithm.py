# -*- coding: utf-8 -*-

from .core import *


class newton(LogisticRegression):
    def __init__(self, x, y, lamb=1, tol=1e-8, iter_lim=100, fit_intercept=True):
        LogisticRegression.__init__(self, x, y, lamb, tol, iter_lim, fit_intercept)
        _, n = self.x.shape
        w_new = np.zeros((n, 1))
        ntime = 0
        while ntime < self.iter_lim:
            # update parameter
            w_old = w_new

            # calculate diagonal matrix, that a is the A_ii
            a = sigmoid_prime(self.x.dot(w_old))
            A = np.diagflat(a)

            # Hessian matrix
            H = hessian(self.x, A, self.lamb)

            # a part of gradient objective and combine previous iterated parameter
            z = self.x.dot(w_old) + (1 - sigmoid(self.y * self.x.dot(w_old))) * self.y / a
            XAz = self.x.T.dot(A).dot(z)

            # calculate current parameter
            w_new = np.linalg.inv(H).dot(XAz)

            # calculate tolerance and iteration time to break loop
            if np.sqrt((w_new - w_old).T.dot(w_new - w_old)) < tol:  # all(abs(w - w_old) < tol):
                break

            else:
                ntime += 1

        # get parameter
        self.weight = w_new


class coord(LogisticRegression):
    def __init__(self, x, y, lamb=1, tol=1e-8, iter_lim=100, fit_intercept=True):
        LogisticRegression.__init__(self, x, y, lamb, tol, iter_lim, fit_intercept)
        _, n = self.x.shape
        w_new = np.zeros((n, 1))
        ntime = 0
        while ntime < self.iter_lim:
            # updata parameter
            w_old = w_new

            # calculate diagonal matrix, that a is the A_ii
            a = sigmoid_prime(self.x.dot(w_old))

            g = []
            axx =[]
            for k, _ in enumerate(w_old):
                for i, _ in enumerate(self.x):
                    g += [1 - sigmoid(self.y[i] * np.dot(self.x[i, :], w_old)) * self.y[i] * self.x[i, k]]
                    axx += a[i] * self.x[i, k] ** 2
                g = sum(g) - lamb * w_old[k]
                w_new[k] = w_old[k] + g / (lamb + sum(axx))

            # calculate current parameter
            # w_new = w_old

            # calculate tolerance and iteration time to break loop
            if np.sqrt((w_new - w_old).T.dot(w_new - w_old)) < tol:  # all(abs(w - w_old) < tol):
                break

            else:
                ntime += 1

        # get parameter
        self.weight = w_new
