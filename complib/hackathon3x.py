from time import strptime, localtime
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from ple import FeatureUnionExt, PipelineExt

now = localtime()
_gender_map = {'Female': '0', 'Male': '1', '':'nan'}
_bool_map = {'N': '0', 'Y': '1', '':'nan'}
_var1_map = {'HAVC': 0, 'HAXA': 1, 'HAXB': 2, 'HAXC': 3, 'HAXF': 4, 'HAXM': 5, 'HAYT': 6, 'HAZD': 7, 'HBXA': 8, 'HBXB': 9, 'HBXC': 10, 'HBXD': 11, 'HBXH': 12, 'HBXX': 13, 'HCXD': 14, 'HCXF': 15, 'HCXG': 16, 'HCYS': 17, 'HVYS': 18}
_device_type_map = {'Mobile': '0', 'Web-browser': '1', '':'nan'}
_var2_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, '':'nan'}
_source_map = {'S122': 0, 'S133': 1,'':'nan'}
_miss_value = lambda x: 'nan' if x == '' else x

def model(*argList, **argDict):
    classifier = GradientBoostingClassifier(verbose=1)

#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_split': [1200], 'min_samples_leaf':[60], 'max_depth':[9], 'max_features':[7], 'subsample':[0.8]}

#    param_grid={'n_estimators':np.arange(50, 50+50, 10), 'learning_rate':np.arange(0.01, 0.01+0.2, 0.01)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'max_depth':np.arange(1, 1+10, 1)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_split':np.arange(2, 2+1000, 100)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':np.arange(2, 2+100, 10)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':np.arange(0.7, 0.7+0.29, 0.01)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'max_leaf_nodes':np.arange(2, 2+100, 10)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'max_features':np.arange(1, 1+19, 1)}

#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'min_samples_split':np.arange(2, 2+3000, 100)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'subsample':np.arange(0.7, 0.7+0.29, 0.01)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'max_depth':np.arange(1, 1+10, 1)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'max_features':np.arange(1, 1+19, 1)}

#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'max_depth':[4], 'subsample':np.arange(0.7, 0.7+0.29, 0.01)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'max_depth':[4], 'max_features':np.arange(1, 1+19, 1)}

#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_leaf':[12], 'max_depth':[4], 'subsample':[0.77], 'max_features':np.arange(1, 1+19, 1)}


#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_split':[202], 'min_samples_leaf':np.arange(2, 2+100, 10)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_split':[202], 'subsample':np.arange(0.7, 0.7+0.29, 0.01)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'min_samples_split':[202], 'max_features':np.arange(1, 1+19, 1)}

#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':[0.84], 'max_features':np.arange(1, 1+19, 1)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':[0.84], 'max_features':[11], 'min_samples_leaf':np.arange(1, 1+100, 10)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':[0.84], 'max_features':[11], 'min_samples_leaf':[51], 'min_samples_split':np.arange(2, 2+1500, 100)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':[0.84], 'max_features':[11], 'min_samples_leaf':[51], 'min_samples_split':[402], 'max_depth':np.arange(1, 1+10, 1)}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':[0.84], 'max_features':[11], 'min_samples_leaf':[51], 'min_samples_split':np.arange(2, 2+1500, 100), 'max_depth':[7]}
#    param_grid={'n_estimators':[60], 'learning_rate':[0.1], 'subsample':[0.84], 'max_features':[11], 'min_samples_leaf':np.arange(1, 1+100, 1), 'min_samples_split':[1202], 'max_depth':[7]}

    searcher = GridSearchCV(classifier, n_jobs=-1, scoring='roc_auc', param_grid=param_grid)

    return searcher

def _transfer(X):
    imputers = [['Imputer_{i}'.format(i=i), Imputer()] for i in range(19)] 
    imputers[0][1] = imputers[6][1] = imputers[8][1] = imputers[9][1] = imputers[10][1] = imputers[11][1] = imputers[12][1] = imputers[13][1] = imputers[14][1] = imputers[15][1] = imputers[16][1] = imputers[17][1] = Imputer(strategy='most_frequent')
    step1 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=imputers, idx_list=[[i] for i in range(19)]))
    step2 = ('OneHotEncoder', OneHotEncoder(categorical_features=[0, 6, 8, 14, 15, 16, 17], sparse=False))
    step3 = ('StandardScaler', StandardScaler())
    pipeline = PipelineExt(steps=[step1, step2, step3])
    X = pipeline.fit_transform(X)
    return X

def loadTrainSet(filepath):
    converters = dict([(i, _miss_value) for i in range(26)])
    converters[1] = lambda x: _gender_map[x]
    converters[4] = lambda x: now.tm_year - strptime(x, '%d-%b-%y').tm_year
    converters[11] = converters[19] = lambda x: _bool_map[x]
    converters[13] = lambda x: _var1_map[x]
    converters[14] = converters[15] = converters[16] = converters[17] = converters[18] = lambda x: 1 if len(x.strip()) == 0 else 0
    converters[20] = lambda x: _device_type_map[x]
    converters[21] = lambda x: _var2_map[x]
    converters[22] = lambda x: _source_map.get(x, '2')

    usecols = [1, 3, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25]
    raw = np.loadtxt(filepath, delimiter=',', usecols=usecols, converters=converters, dtype=np.str, skiprows=1)
    raw = raw.astype(np.float64)
    X, y = raw[:,:-1], raw[:,-1]
    X = _transfer(X)
    
    trainSet = np.hstack((X, y.reshape(-1,1)))
    return trainSet

def loadTestSet(filepath):
    raise

def saveSubmission(filepath, idList, y):
    raise
