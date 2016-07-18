import numpy as np
from sklearn.externals.joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

def main():
    try: 
        trainSet = load('../../dump/Digit Recognizer/train.dmp')
    except:
        trainSet = np.loadtxt('../../data/Digit Recognizer/train.csv', delimiter=',', skiprows=1)
        dump(trainSet, '../../dump/Digit Recognizer/train.dmp')

    try: 
        testSet = load('../../dump/Digit Recognizer/test.dmp')
    except:
        testSet = np.loadtxt('../../data/Digit Recognizer/test.csv', delimiter=',', skiprows=1)
        dump(testSet, '../../dump/Digit Recognizer/test.dmp')

    X = trainSet[:,1:]
    y = trainSet[:,0]

    model = RandomForestClassifier()
    model.fit(X, y)
    print model.score(X, y)

    X = testSet
    y = model.predict(X)

    result = np.vstack((np.arange(1, y.size+1), y)).T

    np.savetxt('../../submission/Digit Recognizer/submission.csv', result, fmt='%d', delimiter=',', header='ImageId,Label')

if __name__ == '__main__':
    main()
