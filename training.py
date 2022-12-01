import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import KFold

df = pd.read_csv('mnist.csv')
df_values = df.values
print(df_values)
labels = df_values[:, 0]
digits = df_values[:, 1:]
img_size = 28
'''print(digits) 
plt.imshow(digits[1].reshape(img_size,img_size))
plt.show()'''

'''labels_u = np.unique(labels)
counts = np.array([])
for i in np.arange(0,10):
    count = 0
    for label in labels:
        if label == i:
            count = count + 1
    counts = np.append(counts, count)
print(counts)
maj = np.argmax(counts)
print(maj)
perc = counts[maj]/np.sum(counts)
print(perc*100,"%")'''

def getAllDataForLabel(digits, labels, label):
    return [digits[i] for i in range(digits.shape[0]) if labels[i]==label]
def ink_features(values):
    ink = np.array([])
    for row in values:
        ink_v = np.sum(row[1:])
        ink = np.append(ink, ink_v)
    return ink


ink_f = ink_features(df_values)
ink_mean = [np.round(np.mean(ink_f[labels == i]), 2) for i in range(10)]
print(ink_mean)
ink_std = [np.round(np.std(ink_f[labels == i]), 2) for i in range(10)]
print(ink_std)

scaler = StandardScaler()
digits_scaled = scaler.fit_transform(digits)
'''ink_f_scaled = np.array([np.sum(row) for row in digits])
'''
ink_f_scaled = scale(ink_f)
ink_mean = [np.round(np.mean(ink_f_scaled[labels == i]), 2) for i in range(10)]
print(ink_mean)
ink_std = [np.round(np.std(ink_f_scaled[labels == i]), 2) for i in range(10)]
print(ink_std)

#cross-validation split and Logistic Regression possible coefficients
kf = KFold(n_splits=10, shuffle=False)
ink_f_scaled = np.reshape(ink_f_scaled, (-1, 1))
# labels = np.reshape(labels, (-1, 1))
cross = LogisticRegressionCV(Cs=10, solver="saga", multi_class="multinomial").fit(ink_f_scaled, labels)
cs = cross.Cs_

i = 0

accuracies = []
classifiers = []

#training and evaluating classifiers
for x_train, x_test in kf.split(digits, labels):
    xtrain_set = ink_f_scaled[x_train]
    ytrain_set = labels[x_train]

    xtest_set = ink_f_scaled[x_test]
    ytest_set = labels[x_test]

    print(np.unique(ytrain_set))
    print(np.unique(ytest_set))

    classifier = LogisticRegression(penalty="l1", C=cs[i], solver="saga", multi_class="multinomial").fit(xtrain_set, ytrain_set)
    classifiers.append(classifier)
    
    i = i + 1

    predict = classifier.predict(xtest_set)
    accuracies.append(accuracy_score(ytest_set,predict ))
    '''plot_confusion_matrix(classifier, xtest_set, ytest_set)
    plt.show()'''

#best classifier based on accuracy and confusion matrix
best_classifier = classifiers[np.argmax(accuracies)]
plot_confusion_matrix(best_classifier, ink_f_scaled, labels)
plt.show()   

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



def getDataDistribution(labels, returnAsLatexTable= False):
    df = pd.DataFrame(pd.Series(labels).value_counts(),columns=['frequency'])
    df['label'] = df.index
    df = df.sort_values(by='label')
    df = df[['label', 'frequency']]
    if returnAsLatexTable:
        df = df.transpose()
        return df.to_latex()

    return df




# Center feature extracting

def getCenterForDim(array2D, dim):
    tot = 0
    totInk = 0
    for x in range(array2D.shape[0]):
        for y in range(array2D.shape[1]):
            point = x, y
            tot += point[dim] * array2D[x, y]
            totInk += array2D[x, y]
    return tot / totInk


def getCenter(array1D):
    array2D = array1D.reshape(28, 28)
    return getCenterForDim(array2D, 0), getCenterForDim(array2D, 1)


def extractCenters(pixData):
    """
    input: np matrix of shape Nsamples, Npixels
    Output: np matrix of shape Nsamples, 2 (one column for the x value of the mean and one for the y)
    """
    X = []
    Y = []
    for sample in pixData:
        x, y = getCenter(sample)
        X.append(x)
        Y.append(y)
    return np.array([X, Y]).T

# Extracting this feature takes quite some time and its usability is questionable
# Uncomment next line if you want to extract it
#XCents = extractCenters(digits)
