import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

def model(*argList, **argDict):
#    classifier = RandomForestClassifier(verbose=2, n_jobs=-1)

#    param_grid={'n_estimators':np.arange(1, 202, 10)}
#    param_grid={'n_estimators':[200], 'criterion':['gini', 'entropy']}
#    param_grid={'n_estimators':[200], 'max_features':np.append(np.arange(28-20, 28, 1), np.arange(28, 28+20, 1))}
#    param_grid={'n_estimators':[200], 'max_depth':np.arange(40, 40+20, 1)}
#    param_grid={'n_estimators':[200], 'min_samples_split':np.arange(2, 2+10, 1)}
#    param_grid={'n_estimators':[200], 'min_samples_leaf':np.arange(1, 1+10, 1)}
#    param_grid={'n_estimators':[200], 'max_leaf_nodes':np.arange(3000, 3000+1000, 100)}

    classifier = GradientBoostingClassifier(verbose=1)
#    param_grid={'n_estimators':[1]}
#    param_grid={'subsample':[0.05], 'max_features':[500], 'n_estimators':[1]}
#    param_grid={'subsample':[0.05], 'max_features':[500], 'n_estimators':[100], 'learning_rate':np.arange(0.01, 0.01+0.2, 0.01)}
    param_grid={'subsample':[0.05], 'max_features':[500], 'n_estimators':[100], 'learning_rate':[0.05], 'max_depth':np.arange(3, 3+5, 1)}
#    param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'max_depth':np.arange(1, 1+20, 1)}
#    param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'min_samples_split':np.arange(2, 2+20, 2)}
#    param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'min_samples_leaf':np.arange(1, 1+20, 2)}
#    param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'subsample':np.arange(0.1, 0.1+1, 0.1)}
#    param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'max_leaf_nodes':np.arange(20, 20+20, 2)}
#    param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'max_features':np.arange(1, 1+7, 1)}


    searcher = GridSearchCV(classifier, n_jobs=-1, param_grid=param_grid)

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
