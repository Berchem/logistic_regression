import pandas as pd
import logistic_regression
from logistic_regression.algorithm import newton


# import data
DF = pd.read_csv("binary.csv")

# get data label
col = DF.columns.values

# outcome y
y = DF[["admit"]]

# variables x
x = DF.loc[:, col[col != 'admit']]

# logist_fit = logistic_regression.fit("newton", x, y)
logist_fit = newton(x, y)


print "  ".join("%.4f" % w for w in logist_fit.weight)
