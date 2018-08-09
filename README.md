# Algorithm for Logistic Regression

Building an algorithm for logistic regression in python

## Usage

**Import Module**

```
import logistic_regression
```

**Fit variables x and outcome y**

```
lr = logistic_regression.fit(x, y[, ...])
```

## Description

**class *fit*(*x*, *y*,** *****karg*)** 

 parameters
> *x: {array-like, sparse matrix}, shape (n_samples, n_features)*
>
> *y: array-like, shape (n_samples, 1)*
> 
> *tol: float, default: 1e-8*
>
>&nbsp; &nbsp; &nbsp; &nbsp; Tolerance for stopping criteria.
>
> *lamb: float, default: 1.0*
>
>&nbsp; &nbsp; &nbsp; &nbsp; regularization strength; **larger** values specify **stronger** regularization.
>
> *fit_intercept: bool, default: True*
> 
>&nbsp; &nbsp; if a constant should be added to the decision function.
>
> *iter_lim : int, default: 100*
>
>&nbsp; &nbsp; &nbsp; &nbsp; Maximum number of iterations taken for the solvers to converge.

 attributes
> *weight: array, shape (n_features, 1)*
>
> &nbsp; &nbsp; Coefficient of the features in the decision function. The features contain the constant item  if it added.

## Class Diagram


**logistic_regression**
```
    ├── init 
    │     └── fit(algorithm.newton)
    ├── algorithm
    │     └── newton(core.LogisticRegression)
    └── core
          └── LogisticRegression
```
## Algorithm
see also <a href="algorithm_doc.ipynb">algorithm document</a>

## Requirements
* Python 2.7
* numpy
* pandas
* matplotlib

