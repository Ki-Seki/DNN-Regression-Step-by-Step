'''
 # @ Author: Ki Seki
 # @ Create Time: 2023-01-16 21:17:35
 # @ Modified by: Ki Seki
 # @ Modified time: 2023-01-16 22:20:19
 # @ Description:
 '''

def task(x):  # sourcery skip: inline-immediately-returned-variable
    '''
    Define your task function here. This function will be learned by the model.
    :param x: x can be any dimension
    :return: must be a scalar
    '''
    y = 3 * x + 5
    return y