import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

def model(*argList, **argDict):
    classifier = RandomForestClassifier()
    searcher = GridSearchCV(classifier, *argList, **argDict)
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
