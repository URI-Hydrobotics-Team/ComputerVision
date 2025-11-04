import sys

from expttools import Experiment

class NamedObject():
    def __init__(self,name,value):
        '''
        wrap value in an object with name

        Parameters
        -----------------
        name : string
            name of object
        value : any
             object to store
        '''
        self.name = name
        self.value = value

def my_expt_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Define a dictionary with parameters and their values
param_grid = {
    "param1": [1, 2, 3],
    "param2": ["a", "b", "c"],
    "param3": [True, False],
    "param4": [NamedObject("obj1", 1), NamedObject("obj2", 2)]
}

for param, values in param_grid.items():
    for value in values:
        my_expt_function(**{param: value})
        


curr_experiment = Experiment(my_expt_function, param_grid)