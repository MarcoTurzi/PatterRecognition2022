import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler

df = pd.read_csv("./data/mnist.csv")
df_values = df.values
print(df_values)
labels = df_values[:, 0]
digits = df_values[:, 1:]
img_size = 28


def getAllDataForLabel(digits, labels, label):
    return [digits[i] for i in range(digits.shape[0]) if labels[i]==label]


scaler = StandardScaler()
digits_scaled = scaler.fit_transform(digits)
'''ink_f_scaled = np.array([np.sum(row) for row in digits])
'''



# pixel variance analysis
def getStds(pixData):
    stds = np.std(pixData, axis=0)
    i = 1
    while plt.fignum_exists(i):
        i += 1
    plt.figure(i)
    plot = plt.imshow(stds.reshape(28, 28))
    plt.colorbar(label = 'Standard deviation over all samples')
    plt.xlabel('x-coordinate pixel')
    plt.ylabel('y-coordinate pixel')
    plt.show()
    #plt.close()
    return stds

#stds = getStds(digits)
def getMeans(pixData, vmax = 255, save = False, saveName = "figure"):
    means = np.mean(pixData, axis=0)
    i = 1
    while plt.fignum_exists(i):
        i += 1
    plt.figure(i)
    plot =plt.imshow(means.reshape(28, 28), vmin = 0, vmax = vmax)
    plt.colorbar(label = 'Mean over all samples')
    plt.xlabel('x-coordinate pixel')
    plt.ylabel('y-coordinate pixel')
    plt.show()
    #plt.close()
    if save:
        plt.savefig(fname="./output/dataplots/" + saveName + ".png",format='png')
    return means

#means = getMeans(digits,150)


def getDigitMeansMinusTotalMeans(digits, labels, digitLabel, save = False, saveName = "figure"):
    meanD = np.mean(getAllDataForLabel(digits,labels,digitLabel), axis = 0)
    meanT = np.mean(digits, axis = 0)
    diff = meanD - meanT
    i = 1
    while plt.fignum_exists(i):
        i += 1
    plt.figure(i)
    plot =plt.imshow(diff.reshape(28, 28),vmin=-125,vmax=150)
    plt.colorbar(label = 'Class mean with respect to mean over all samples')
    plt.xlabel('x-coordinate pixel')
    plt.ylabel('y-coordinate pixel')
    plt.show()
    if save:
        plt.savefig(fname="./output/dataplots/" + saveName + ".png",format='png')
    return diff

def getRelativeMeansPerDigit(digits, labels, save=False):
    for i in range(10):
        getDigitMeansMinusTotalMeans(digits,labels,i, save=save,saveName="Figure rel means digit "+str(i))
def getMeansPerDigit(digits, labels, save = False):
    for i in range(10):
        getMeans(getAllDataForLabel(digits,labels,i),save=save, saveName= "Figure means digit " + str(i))

def getOutsideClassStd(digits, labels):
    res = np.std(digits, axis=0)

    for i in range(10):
        res -= np.std(getAllDataForLabel(digits,labels,i), axis=0)*0.1
    i = 1
    while plt.fignum_exists(i):
        i += 1
    plt.figure(i)
    plot = plt.imshow(res.reshape(28, 28))
    plt.colorbar(label='Standard deviation between classes')
    plt.xlabel('x-coordinate pixel')
    plt.ylabel('y-coordinate pixel')
    plt.show()
    # plt.close()
    return res

def getDataDistribution(labels, returnAsLatexTable= False):
    df = pd.DataFrame(pd.Series(labels).value_counts(),columns=['frequency'])
    df['label'] = df.index
    df = df.sort_values(by='label')
    df = df[['label', 'frequency']]
    if returnAsLatexTable:
        df = df.transpose()
        return df.to_latex()

    return df

def compareClasses(digits,labels,label1, label2, similarityFunction):
    means1 = np.mean(getAllDataForLabel(digits, labels, label1), axis=0)
    means2 = np.mean(getAllDataForLabel(digits, labels, label2), axis = 0)
    return similarityFunction(means1, means2)

def eucledianDistArrays(X1,X2):
    return np.linalg.norm(X1-X2)

def corrCoefArrays(X1,X2):
    return np.corrcoef(X1,X2)[1,0]

def compareAllClassPairs(digits, labels, similarityFunction):
    simMat = np.empty([10,10])
    for x in range(10):
        m1 = np.mean(getAllDataForLabel(digits,labels,x),axis=0)
        for y in range(10):
            m2 = np.mean(getAllDataForLabel(digits, labels, y),axis=0)
            simMat[y,x] = similarityFunction(m1, m2)
    return simMat

def plotDistanceMatrix(matrix):
    plt.imshow(matrix)
    plt.colorbar(label='Dissimilarity in terms of euclidian distance')
    plt.xticks(np.arange(0, 10))
    plt.yticks(np.arange(0, 10))
    plt.xlabel('digit label')
    plt.ylabel('digit label')


