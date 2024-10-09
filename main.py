import numpy as np
import scipy.io as si

#Uses cross-validation to calculate the average error rate of the classifier
def classifier(X, y):
    errRate = np.empty(8)
    for i in range(0,8):
        toAdd = X[i*16:(i+1)*16 - 1]
        yToUse = y[i*16:(i+1)*16 - 1]

        train = X
        yTrain = y
        for j in range(0,16):
            train = np.delete(train, (i*16) + 15 - j, 0)
            yTrain = np.delete(yTrain, (i*16) + 15 - j, 0)
        
        classWeights = calculateWeights(train, yTrain)

        test = toAdd
        yTest = yToUse
        numMissClass = 0
        for j in range(0,len(yTest)):
            if np.sign(test[j] @ classWeights) != yTest[j]:
                numMissClass += 1
    
        errRate[i] = numMissClass / 16

    return errRate

#Calculates the weights for the classifier using weights = (X^T * X)^-1 * X^T * y
def calculateWeights(X, y):
    toInv = X.T @ X
    inv = np.linalg.inv(toInv)
    what = inv @ X.T @ y
    return what

#Loads the data from the .mat file and initialises the X and y values
def initData():
    data = si.loadmat('face_emotion_data.mat')
    y = data['y']
    X = data['X']
    return X, y


if __name__ == '__main__':
    X,y = initData()
    errRate = classifier(X, y)
    avgErrRate = np.mean(errRate)
    print("Average Error Rate: ", avgErrRate)