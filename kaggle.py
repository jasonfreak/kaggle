#!/usr/local/bin/python
from os import listdir
from os.path import basename, splitext
from argparse import ArgumentParser
import numpy as np
from sklearn.externals import joblib
from sklearn.grid_search import BaseSearchCV
import complib

def fit(competition, label, train):
    lib = getattr(complib, competition)

    model = lib.model()
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

def analyze(competition, label):
    from matplotlib import pyplot as plt
    fig = plt.figure()

    model = joblib.load('dump/{competition}/{label}.dmp'.format(competition=competition, label=label))
    assert(isinstance(model, BaseSearchCV))
    print 'Best Score:{best_score}, Best Params:{best_params}'.format(best_score=model.best_score_, best_params=model.best_params_)

    n_grid_scores = len(model.grid_scores_)
    if n_grid_scores > 0:
        n_params = len(model.grid_scores_[0][0])
        if n_params == 1:
            _singleParamAnalyze(plt, model)
        elif n_params == 2:
            _coupleParamAnalyze(plt, model)
        else:
            raise

    with PdfPages('pdf/{competition}/{label}.pdf') as pdf:
	    pdf.savefig(fig)
    plt.close()

def _sortValueAndScore(valueList, scoreList):
    valueAndScoreList = sorted(zip(valueList, scoreList), key=lambda x:x[0])
    valueList = [x[0] for x in valueAndScoreList]
    scoreList = [x[1] for x in valueAndScoreList]
    return valueList, scoreList

def _singleParamAnalyze(plt, model):
    n_grid_scores = len(model.grid_scores_)
    param = model.grid_scores_[0].keys()[0]

    paramType = type(model.grid_scores_[0][0][param])
    valueList = np.array([])
    scoreList = np.array([])

    for grid_score in model.grid_scores_:
        valueList = np.append(valueList, grid_score[0][param])
        scoreList = np.append(scoreList, grid_score[1])

    if paramType is unicode:
        x_ticks = np.unique(valueList)
        x_tickDict = dict([(x_ticks[i], i+0.5) for i in range(len(x_ticks))])
        x_pos = [x_tickDict[value] for value in valueList]
        plt.xticks(x_tickDict.values(), x_tickDict.keys())
    else:
        valueList, scoreList = _sortValueAndScore(valueList, scoreList)
        x_pos = np.arange(n_grid_scores) + 0.5
        plt.xticks(x_pos, valueList)
    plt.plot(x_pos, scoreList, '-')

    plt.axis([0, n_grid_scores, 0, 1])
    plt.title('How {param} Affects Accuracy'.format(param=param))
    plt.xlabel(param)
    plt.ylabel('Accuracy')

def _coupleParamAnalyze(plt, model):
    n_grid_scores = len(model.grid_scores_)
    param1, param2 = model.grid_scores_[0].keys()

    paramType1 = type(model.grid_scores_[0][0][param1])
    paramType2 = type(model.grid_scores_[0][0][param2])
    param_grid = getattr(model, 'param_grid')
    n_x_values = len(param_grid[param1])
    n_y_values = len(param_grid[param2])
    param_x_pos = dict(zip(param_grid[param1], range(n_x_values)))
    param_y_pos = dict(zip(param_grid[param2], range(n_y_values)))
    param_shape = (n_y_values, n_x_values)

    scores = np.zeros(param_shape)

    for grid_score in model.grid_scores_:
        value1 = grid_score[0][param1]
        value2 = grid_score[0][param2]
        scores[param_y_pos[value2], param_x_pos[value1]] = grid_score[1] 

    max_x = np.max(scores, axis=1)
    max_y = np.max(scores, axis=0)
    idx_max_x = np.dot((scores == max_x.reshape((-1, 1))), np.arange(n_x_values))
    idx_max_y = np.dot(np.arange(n_y_values), (scores == max_y.reshape((1, -1))))

#    print '{param1}\'s independence:{independence1}\n{param2}\'s independence:{independence2}'.format(param1=param1, independence1=np.std(idx_max_x), param2=param2, independence2=np.std(idx_max_y))
    image = 1 - scores

    plt.xticks(param_x_pos.values(), param_x_pos.keys())
    plt.yticks(param_y_pos.values(), param_y_pos.keys())

    plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    plt.xlabel(param1)
    plt.ylabel(param2)
    plt.title('How {param1} and {param2} Affects Accuracy'.format(param1=param1, param2=param2))

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

    args = parser.parse_args()

    assert(args.label not in ('train', 'test'))
    if args.action == 'fit':
        fit(args.competition, args.label, args.train)
    elif args.action == 'analyze':
        analyze(args.competition, args.label)
    elif args.action == 'predict':
        predict(args.competition, args.label, args.test, args.submission)
    elif args.action == 'list':
        listall(args.competition)

if __name__ == '__main__':
    main()
