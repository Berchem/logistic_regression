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

# define argument
lamb = 1.0
lim = 3
fit_intercept = True
tol = 1e-8

logist_fit = logistic_regression.fit(x, y, lamb=1./lamb, iter_lim=lim, fit_intercept=fit_intercept, tol=tol)
# logist_fit = newton(x, y)



import os
print "\njava started"
os.system("java -jar log.jar binary.csv weight.txt %s %d %s %s" % (str(lamb), lim, str(fit_intercept).lower(), str(tol)))
print "java end\n"
with open("weight.txt") as f:
    s = f.read()

w_java = [float(i) for i in s.strip().split("\t")]
print "with java  :" + "  ".join("%.4f" % wj for wj in w_java)
print "with python:" + "  ".join("%.4f" % wp for wp in logist_fit.weight)
print type(logist_fit.weight)
