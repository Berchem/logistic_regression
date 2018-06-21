# -*- coding: utf-8 -*-

from .core import *


class Algorithm:

    def __init__(self):
        pass

    def newton(self, x, y, tol=TOL, iter_lim=LIM, fit_intercept=True):
        if fit_intercept:
            _

        p = x.shape[1]
        w = np.zeros((p, 1))
        times = 0

        while times < iter_lim:
            w_old = w
            a = sigmoid(x.dot(w_old)) * (1 - sigmoid(x.dot(w_old)))
            A = np.diagflat(a)
            H = x.T.dot(A).dot(x)
            z = x.dot(w_old) + (1 - sigmoid(y * x.dot(w_old))) * y / a
            XAz = x.T.dot(A).dot(z)
            w = np.linalg.inv(H).dot(XAz)
            if all(abs(w - w_old) < tol):
                break
            else:
                times += 1

        return w, H
