import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

def model(*argList, **argDict):
    classifier = RandomForestClassifier(verbose=2)
#    classifier = GradientBoostingClassifier(verbose=2)
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[50, 100, 150], 'learning_rate':[0.05, 0.1, 0.15]})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(8, 12), 'max_features':np.arange(0.5, 1.0, 0.1)})
    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(100, 1001, 300)})
    return searcher
    
def loadTrainSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', skiprows=1)
#    X, X_abandom, y, y_abandom = train_test_split(raw[:,1:], raw[:,[0]], test_size=0.9, random_state=0)
    X, y = raw[:,1:], raw[:,0]
    trainSet = np.hstack((X, y.reshape(-1,1)))
    return trainSet

def loadTestSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', skiprows=1)
    testSet = raw
    return testSet

def saveSubmission(filepath, y):
    result = np.vstack((np.arange(1, y.size+1), y)).T
    np.savetxt(filepath, result, fmt='%d', delimiter=',', header='ImageId,Label', comments='')
