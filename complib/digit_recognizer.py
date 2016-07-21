import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

def model(*argList, **argDict):
    classifier = RandomForestClassifier(verbose=2)
#    classifier = GradientBoostingClassifier(verbose=2)
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[50, 100, 150], 'learning_rate':[0.05, 0.1, 0.15]})
    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(8, 12), 'max_features':np.arange(0.5, 1.1, 0.1)})
    return searcher
    
def loadTrainSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', skiprows=1)
    trainSet = np.hstack((raw[:,1:], raw[:,[0]]))
    return trainSet

def loadTestSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', skiprows=1)
    testSet = raw
    return testSet

def saveSubmission(filepath, y):
    result = np.vstack((np.arange(1, y.size+1), y)).T
    np.savetxt(filepath, result, fmt='%d', delimiter=',', header='ImageId,Label', comments='')
