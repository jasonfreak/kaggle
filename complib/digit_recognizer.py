import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

def model(*argList, **argDict):
    classifier = RandomForestClassifier(verbose=2, n_jobs=-1)
#    classifier = GradientBoostingClassifier(verbose=2)
#    classifier = GradientBoostingClassifier(verbose=2)
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(10, 101, 10)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(10, 101, 10), 'max_features':np.arange(0.1, 1, 0.1)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'max_features':[0.1], 'min_samples_split':np.arange(2, 100, 10)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'max_features':np.arange(0.05, 0.15, 0.01), 'min_samples_split':[2]})
#    searcher = GridSearchCV(classifier, param_grid={'criterion':['gini', 'entropy'], 'n_estimators':[100], 'max_features':[0.1], 'min_samples_split':[2]})
#    searcher = GridSearchCV(classifier, param_grid={'criterion':['gini'], 'n_estimators':[100], 'max_features':[0.1], 'min_samples_split':np.arange(100, 1001, 100)})
    searcher = GridSearchCV(classifier, param_grid={'criterion':['gini'], 'n_estimators':[100], 'max_features':[0.1], 'min_samples_split':[5]})

#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(10, 51, 10)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[100], 'learning_rate':np.arange(0.05, 0.16, 0.02)})
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
