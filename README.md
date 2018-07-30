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

class *fit*(*x*, *y*, ***karg*)

> parameters
>> *x: {array-like, sparse matrix}, shape (n_samples, n_features)*
>>
>> *y: array-like, shape (n_samples, 1)*
>> 
>> *tol: float, default: 1e-8*
>>> Tolerance for stopping criteria.
>>
>> *lamb: float, default: 1.0*
>>> regularization strength; **larger** values specify **stronger** regularization.
>>
>> *fit_intercept: bool, default: True*
>>> if a constant should be added to the decision function.
>>
>> *iter_lim : int, default: 100*
>>> Maximum number of iterations taken for the solvers to converge.
>
> attributes
>> *weight: array, shape (n_features, 1)*
>>> Coefficient of the features in the decision function. The features contain the constant item  if it added.

## Class Diagram


logistic_regression
<br>&nbsp; &nbsp; |__ init 
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |__ fit(algorithm.newton)
<br>&nbsp; &nbsp; |__ algorithm
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |__ newton(core.LogisticRegression)
<br>&nbsp; &nbsp; |__ core
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |__ LogisticRegression


## Requirements
* Python 2.7
* numpy
* pandas
* matplotlib

