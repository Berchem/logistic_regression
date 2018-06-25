# -*- coding: UTF-8 -*-

"""
    Logistic regression
    ----------
    Using of Newton's Method for Python (2.7)

    :copyright: (c) 2018 by Berchem Lin
"""


# from .core import *
from .algorithm import newton


class fit(newton):
    def __init__(self, method="newton", *args):
        if method.lower() == "newton":
            newton.__init__(self, *args)
        # elif method.lower() == "coord":
        #     coord.__init__(self, *args)




__copyright__   = "Copyright 2018 by Berchem Lin"
__author__      = "Berchem Lin"
__source__      = "https://github.com/Berchem/ ..... "
