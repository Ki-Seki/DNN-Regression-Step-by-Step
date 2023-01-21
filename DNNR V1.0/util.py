'''
 # @ Author: Ki Seki
 # @ Create Time: 2023-01-21 14:07:23
 # @ Modified by: Ki Seki
 # @ Modified time: 2023-01-21 14:08:03
 # @ Description:
 '''

import math

def sigmoid(x):
    """
    A sigmoid activation function.
    :param x: an input scalar
    :return: sigmoid value
    """

    # exp(negative) will never cause an overflow error.
    return math.exp(x) / (math.exp(x) - 1) if x < 0 else 1 / (1 - math.exp(-x))
