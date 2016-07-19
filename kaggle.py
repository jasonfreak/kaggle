from os import listdir
from os.path import basename, splitext
from argparse import ArgumentParser
import json
import numpy as np
from sklearn.externals import joblib
from sklearn.grid_search import BaseSearchCV
from sklearn.preprocessing import minmax_scale
from matplotlib import pyplot as plt
import complib

def fit(competition, label, train):
    lib = getattr(complib, competition)

    with open('config/{competition}.json'.format(competition=competition)) as f:
        config = json.load(f)

    model = lib.model(*config['param_list'], **config['param_dict'])
    try:
        trainSet = joblib.load('dump/{competition}/train.dmp'.format(competition=competition))
    except IOError, e:
        trainSet = lib.loadTrainSet('data/{competition}/{train}'.format(competition=competition, train=train))
        joblib.dump(trainSet, 'dump/{competition}/train.dmp'.format(competition=competition), compress=3)

    X, y = (trainSet[:,:-1], trainSet[:,-1])
    model.fit(X, y)
    joblib.dump(model, 'dump/{competition}/{label}.dmp'.format(competition=competition, label=label), compress=3)

def predict(competition, label, test, submission):
    lib = getattr(complib, competition)

    model = joblib.load('dump/{competition}/{label}.dmp'.format(competition=competition, label=label))
    try:
        testSet = joblib.load('dump/{competition}/test.dmp'.format(competition=competition))
    except IOError, e:
        testSet = lib.loadTestSet('data/{competition}/{test}'.format(competition=competition, test=test))
        joblib.dump(testSet, 'dump/{competition}/test.dmp'.format(competition=competition), compress=3)
    X = testSet
    y = model.predict(X)

    lib.saveSubmission('data/{competition}/{submission}'.format(competition=competition, submission=submission), y)

def _sortValueAndScore(valueList, scoreList):
    valueAndScoreList = sorted(zip(valueList, scoreList), key=lambda x:x[0])
    valueList = [x[0] for x in valueAndScoreList]
    scoreList = [x[1] for x in valueAndScoreList]
    return valueList, scoreList

def singleParamAnalyze(competition, label, param):
    model = joblib.load('dump/{competition}/{label}.dmp'.format(competition=competition, label=label))
    assert(isinstance(model, BaseSearchCV))

    n_grid_scores = len(model.grid_scores_)
    assert(len(model.grid_scores_[0][0]) == 1 and param in model.grid_scores_[0][0])
    paraType = type(model.grid_scores_[0][0][param])
    valueList = np.array([])
    scoreList = np.array([])

    for grid_score in model.grid_scores_:
        valueList = np.append(valueList, grid_score[0][param])
        scoreList = np.append(scoreList, grid_score[1])
    scoreList = minmax_scale(scoreList)

    if paraType is unicode:
        x_ticks = np.unique(valueList)
        x_tickDict = dict([(x_ticks[i], i) for i in range(len(x_ticks))])
        x_pos = [x_tickDict[value] for value in valueList]
        plt.xticks(x_tickDict.values(), x_tickDict.keys())
    else:
        valueList, scoreList = _sortValueAndScore(valueList, scoreList)
        x_pos = valueList
    plt.plot(x_pos, scoreList, '-')

    plt.title('How {param} Affects Accuracy'.format(param=param))
    plt.x_label(param)
    plt.y_label('Accuracy')
    plt.show()
    print 'Best Score:{best_score}, Best Params:{best_params}'.format(best_score=model.best_score_, best_params=model.best_params_)

def coupleParamAnalyze(competition, label, param1, param2):
    model = joblib.load('dump/{competition}/{label}.dmp'.format(competition=competition, label=label))
    assert(isinstance(model, BaseSearchCV))

    n_grid_scores = len(model.grid_scores_)
    assert(len(model.grid_scores_[0][0]) == 2 and param1 in model.grid_scores_[0][0] and param2 in model.grid_scores_[0][0])
    paraType1 = type(model.grid_scores_[0][0][param1])
    paraType2 = type(model.grid_scores_[0][0][param2])
    valueList1 = np.array([])
    valueList2 = np.array([])
    scoreList = np.array([])

    for grid_score in model.grid_scores_:
        valueList1 = np.append(valueList1, grid_score[0][param1])
        valueList2 = np.append(valueList2, grid_score[0][param2])
        scoreList = np.append(scoreList, grid_score[1])

    scoreList = minmax_scale(scoreList)

    if paraType1 is unicode:
        x_ticks = np.unique(valueList1)
        x_tickDict = dict([(x_ticks[i], i) for i in range(len(x_ticks))])
        x_pos = [x_tickDict[value] for value in valueList1]
        plt.xticks(x_tickDict.values(), x_tickDict.keys())
    else:
        x_pos = valueList1

    if paraType2 is unicode:
        y_ticks = np.unique(valueList2)
        y_tickDict = dict([(y_ticks[i], i) for i in range(len(y_ticks))])
        y_pos = [y_tickDict[value] for value in valueList2]
        plt.xticks(y_tickDict.values(), y_tickDict.keys())
    else:
        y_pos = valueList2

    plt.scatter(x_pos, y_pos, c=np.repeat((1-scoreList).reshape((-1,1)), 3, axis=1), alpha=0.5)
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title('How {param1} and {param2} Affects Accuracy'.format(param1=param1, param2=param2))
    plt.show()
    print 'Best Score:{best_score}, Best Params:{best_params}'.format(best_score=model.best_score_, best_params=model.best_params_)

def listall(competition):
    fileList = listdir('dump/{competition}/'.format(competition=competition))
    for filepath in fileList:
        label, ext = splitext(basename(filepath))
        if label not in ('train', 'test') and ext == '.dmp':
            print label

def main():
    parser = ArgumentParser(description='Practice For Kaggle')
    parser.add_argument('action', action='store', choices=('fit', 'analyze', 'predict', 'list'), help='Action')
    parser.add_argument('competition', action='store', help='Action')
    parser.add_argument('-l', action='store', dest='label', default='default', help='Label')
    parser.add_argument('--train', action='store', dest='train', default='train.csv', help='Train Set File')
    parser.add_argument('--test', action='store', dest='test', default='test.csv', help='Test Set File')
    parser.add_argument('--submission', action='store', dest='submission', default='submission.csv', help='Test Set File')
    parser.add_argument('--param', action='append', dest='param', default=list(), help='Param')

    args = parser.parse_args()

    assert(args.label not in ('train', 'test'))
    if args.action == 'fit':
        fit(args.competition, args.label, args.train)
    elif args.action == 'analyze':
        n_params = len(args.param)
        if  n_params == 1:
            singleParamAnalyze(args.competition, args.label, args.param[0])
        elif n_params == 2:
            coupleParamAnalyze(args.competition, args.label, args.param[0], args.param[1])
    elif args.action == 'predict':
        predict(args.competition, args.label, args.test, args.submission)
    elif args.action == 'list':
        listall(args.competition)

if __name__ == '__main__':
    main()
