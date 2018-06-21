import numpy as np
import pandas as pd
import logistic_regression


DF = pd.read_csv("binary.csv")
data = np.array(DF)
y = (data[:, 0:1] - 0.5) * 2  # outcome 0, 1 -> -1, 1
x = np.concatenate((np.ones((data.shape[0], 1)), data[:, 1:]), axis=1)

logist_fit = logistic_regression.fit()
w, H = logist_fit.newton(x, y)

print w
