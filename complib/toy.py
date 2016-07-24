import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_boston

def model(*argList, **argDict):
    classifier = RandomForestRegressor()
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(10, 20), 'max_features':np.arange(0.1, 1.0, 0.1)})
    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(10, 100, 10), 'max_depth':np.arange(50, 100, 5)})
    return searcher
    
def loadTrainSet(filepath):
    boston = load_boston()
    trainSet = np.hstack((boston.data, boston.target.reshape((-1,1))))
    return trainSet

def loadTestSet(filepath):
    raise Exception('No Test Set')

def saveSubmissor(filepath, y):
    raise Exception('No Test Set')
