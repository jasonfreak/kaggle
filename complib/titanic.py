import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from ple import FeatureUnionExt, PipelineExt

_sex_map = {'female': '0', 'male': '1', '':'nan'}
_embark_map = {'C': '0', 'Q': '1', 'S':'2', '':'nan'}
_miss_value = lambda x: 'nan' if x == '' else x

def model(*argList, **argDict):
    classifier = GradientBoostingClassifier(verbose=1)
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':np.arange(50, 50+200, 10), 'learning_rate':np.arange(0.01, 0.01+0.2, 0.01)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'max_depth':np.arange(1, 1+20, 1)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'min_samples_split':np.arange(2, 2+20, 2)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'min_samples_leaf':np.arange(1, 1+20, 2)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'subsample':np.arange(0.1, 0.1+1, 0.1)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'max_leaf_nodes':np.arange(20, 20+20, 2)})
#    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'max_features':np.arange(1, 1+7, 1)})

    searcher = GridSearchCV(classifier, param_grid={'n_estimators':[181], 'learning_rate':[0.08], 'min_samples_leaf':np.arange(1, 1+20, 2), 'max_features':np.arange(1, 1+7, 1)})

    return searcher

def _transfer(X):
    imputers = [['Imputer_{i}'.format(i=i), Imputer()] for i in range(7)] 
    imputers[1][1] = imputers[6][1] = Imputer(strategy='most_frequent')
    step1 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=imputers, idx_list=[[i] for i in range(7)]))
    step2 = ('OneHotEncoder', OneHotEncoder(categorical_features=[6], sparse=False))
    step3 = ('StandardScaler', StandardScaler())
    pipeline = PipelineExt(steps=[step1, step2, step3])
    X = pipeline.fit_transform(X)
    return X

def loadTrainSet(filepath):
    #Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    converters = dict([(i, _miss_value) for i in range(12)])
    converters[4] = lambda x:_sex_map[x]
    converters[11] = lambda x:_embark_map[x]

    raw = np.loadtxt(filepath, delimiter=',', usecols=[1, 2, 4, 5, 6, 7, 9, 11], converters=converters, dtype=np.str, skiprows=1)
    raw = raw.astype(np.float64)
    X, y = raw[:,1:], raw[:,0]
    X = _transfer(X)
    print X.shape
    
    trainSet = np.hstack((X, y.reshape(-1,1)))
    return trainSet

def loadTestSet(filepath):
    #Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    converters = dict([(i, _miss_value) for i in range(11)])
    converters[3] = lambda x:_sex_map[x]
    converters[10] = lambda x:_embark_map[x]

    raw = np.loadtxt(filepath, delimiter=',', usecols=[0, 1, 3, 4, 5, 6, 8, 10], converters=converters, dtype=np.str, skiprows=1)
    idList, X = raw[:,0], raw[:,1:].astype(np.float64)
    X = _transfer(X)

    testSet = np.hstack((idList.reshape(-1,1), X))
    return testSet

def saveSubmission(filepath, idList, y):
    result = np.vstack((idList, y.astype(np.int64))).T
    np.savetxt(filepath, result, fmt='%s', delimiter=',', header='PassengerId,Survived', comments='')
