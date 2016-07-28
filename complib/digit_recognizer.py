import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

def model(*argList, **argDict):
    classifier = RandomForestClassifier(verbose=2, n_jobs=-1)

#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(1, 202, 10)})
    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'max_features':np.arange(0.01, 0.11, 0.01)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'max_depth':np.arange(11, 22, 1)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'min_samples_split':np.arange(2, 10001, 1000)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'min_samples_leaf':np.arange(1, 10001, 1000)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'max_leaf_nodes':np.arange(1000, 2001, 100)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'max_leaf_nodes':np.arange(1000, 2001, 100)})

    return searcher
    
def loadTrainSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', dtype=np.str, skiprows=1)
    X, y = raw[:,1:], raw[:,0]
    trainSet = np.hstack((X, y.reshape(-1,1)))
    return trainSet

def loadTestSet(filepath):
    raw = np.loadtxt(filepath, delimiter=',', dtype=np.str, skiprows=1)
    testSet = np.hstack((np.arange(1, raw.shape[0]+1).reshape(-1,1), raw))
    return testSet

def saveSubmission(filepath, idList, y):
    result = np.vstack((idList.astype(np.int64), y.astype(np.int64))).T
    np.savetxt(filepath, result, fmt='%d', delimiter=',', header='ImageId,Label', comments='')
