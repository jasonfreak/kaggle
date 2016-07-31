#!/usr/local/bin/python
from os import listdir
from os.path import basename, splitext
from argparse import ArgumentParser
import numpy as np
from sklearn.externals import joblib
from sklearn.grid_search import BaseSearchCV
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import complib

def fit(competition, label, train):
    lib = getattr(complib, competition)

    model = lib.model()
    try:
        trainSet = joblib.load('dump/{competition}/train.dmp'.format(competition=competition))
    except IOError, e:
        trainSet = lib.loadTrainSet('data/{competition}/{train}'.format(competition=competition, train=train))
        joblib.dump(trainSet, 'dump/{competition}/train.dmp'.format(competition=competition), compress=3)

    X, y = (trainSet[:,:-1].astype(np.float64), trainSet[:,-1].astype(np.float64))
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
    idList, X = testSet[:,0], testSet[:,1:].astype(np.float64)
    y = model.predict(X)

    lib.saveSubmission('data/{competition}/{submission}'.format(competition=competition, submission=submission), idList, y)

def analyze(competition, label):
    model = joblib.load('dump/{competition}/{label}.dmp'.format(competition=competition, label=label))
    assert(isinstance(model, BaseSearchCV))
    print 'Best Score:{best_score}, Best Params:{best_params}'.format(best_score=model.best_score_, best_params=model.best_params_)

    dynamicParamAndType = _getDynamicParamAndType(model)
    n_grid_scores = len(model.grid_scores_)
    if n_grid_scores > 0:
        n_params = len(dynamicParamAndType)
	if n_params in (1, 2):
            if n_params == 1:
                param, paramType = dynamicParamAndType[0]
                _singleParamAnalyze(plt, model, param, paramType)
            elif n_params == 2:
                param1, paramType1= dynamicParamAndType[0]
                param2, paramType2= dynamicParamAndType[1]
                _coupleParamAnalyze(plt, model, (param1, param2), (paramType1, paramType2))
            plt.savefig('pic/{competition}/{label}.png'.format(competition=competition, label=label))
            plt.close()
        try:
            trainSet = joblib.load('dump/{competition}/train.dmp'.format(competition=competition))
        except IOError, e:
            trainSet = lib.loadTrainSet('data/{competition}/{train}'.format(competition=competition, train=train))
            joblib.dump(trainSet, 'dump/{competition}/train.dmp'.format(competition=competition), compress=3)

        X, y = (trainSet[:,:-1].astype(np.float64), trainSet[:,-1].astype(np.float64))
        print 'Score on Train:{score}'.format(score=model.score(X, y))

def _getDynamicParamAndType(model):
    param_grid = getattr(model, 'param_grid')
    paramList = param_grid.keys()
    dynamicParamList =  filter(lambda x:len(param_grid[x]) > 1, paramList)
    paramTypeList =  map(lambda x:type(param_grid[x][0]), dynamicParamList)
    return zip(dynamicParamList, paramTypeList)

def _singleParamAnalyze(plt, model, param, paramType):
    n_grid_scores = len(model.grid_scores_)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax[1].yaxis.tick_right()

    valueList = np.array([])
    scoreList = np.array([])
    covScoreList = np.array([])

    for grid_score in model.grid_scores_:
        value = grid_score[0][param]
        score = grid_score[1]
        print 'Param:{param}={value} Score:{score}'.format(param=param, value=value, score=score)
        valueList = np.append(valueList, value)
        scoreList = np.append(scoreList, score)
        covScoreList = np.append(covScoreList, np.std(grid_score[2]) / grid_score[1])

    if paramType is unicode:
        x_ticks = np.unique(valueList)
        x_tickDict = dict([(x_ticks[i], i+0.5) for i in range(len(x_ticks))])
        x_pos = [x_tickDict[value] for value in valueList]
        ax[0].set_xticks(x_tickDict.values())
        ax[0].set_xticklabels(x_tickDict.keys(), rotation=90)
        ax[1].set_xticks(x_tickDict.values())
        ax[1].set_xticklabels(x_tickDict.keys(), rotation=90)
    else:
        x_pos = np.arange(n_grid_scores) + 0.5
        fig.canvas.draw()
        ax[0].set_xticks(x_pos)
        ax[0].set_xticklabels(valueList, rotation=90)
        ax[1].set_xticks(x_pos)
        ax[1].set_xticklabels(valueList, rotation=90)

    ax[0].plot(x_pos, scoreList, '-')
    ax[1].plot(x_pos, covScoreList, '-')
    ax[0].set_title('Accuracy@\'{param}\''.format(param=param))
    ax[1].set_title('COV Accuracy@\'{param}\''.format(param=param))
    ax[0].set_xlabel(param)
    ax[1].set_xlabel(param)
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('STD Accuracy')

def _coupleParamAnalyze(plt, model, params, paramsType):
    n_grid_scores = len(model.grid_scores_)
    param1, param2 = params
    paramType1, paramType2 = paramsType

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
        score = grid_score[1] 
        print 'Param:{param1}={value1} {param2}={value2} Score:{score}'.format(param1=param1, value1=value1, param2=param2, value2=value2, score=score)
        scores[param_y_pos[value2], param_x_pos[value1]] = score

    max_x = np.max(scores, axis=1)
    max_y = np.max(scores, axis=0)
    idx_max_x = np.dot((scores == max_x.reshape((-1, 1))), np.arange(n_x_values))
    idx_max_y = np.dot(np.arange(n_y_values), (scores == max_y.reshape((1, -1))))

    image = 1 - scores

    plt.xticks(param_x_pos.values(), param_x_pos.keys(), rotation=90)
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
