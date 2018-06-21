# -*- coding: UTF-8 -*-

"""
    Logistic regression
    ----------
    Using of Newton's Method for Python (2.7)

    :copyright: (c) 2018 by Berchem Lin
"""


from .core import *
from .algorithm import Algorithm



class fit(Algorithm):
    def __init__(self, *args):
        Algorithm.__init__(self, *args)
        # Videos.__init__(self, keywords=keywords, *args)
        # Photos.__init__(self, keywords=keywords, *args)



__copyright__   = "Copyright 2018 by Berchem Lin"
__author__      = "Berchem Lin"
__source__      = "https://github.com/Berchem/ ..... "
