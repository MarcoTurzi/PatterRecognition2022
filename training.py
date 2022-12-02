import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import cv2


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
def new_feature(values):
    f_circles = np.array([])
    i = 0
    for row in values:
        img = np.array(row).reshape((28,28))
        count = 0
        for row in img:
            
            check = 0
            for pix in row:
                if pix > 0 and check == 0:
                    check = check + 1
                if pix == 0 and check == 1:
                    check = check + 1
                if pix > 0 and check == 2:
                    count = count + 1
                    break

        f_circles = np.append(f_circles, count)
    return f_circles

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

features = [[sum, avg] for sum,avg in zip(ink_features(digits), new_feature(digits))]
feat_scaled = scale(features)
cross = LogisticRegressionCV(Cs=10,penalty="l1", solver="saga", multi_class="multinomial").fit(feat_scaled, labels)
cs = cross.Cs_

i = 0

accuracies = []
classifiers = []

#training and evaluating classifiers
for x_train, x_test in kf.split(digits, labels):
    xtrain_set = feat_scaled[x_train]
    ytrain_set = labels[x_train]

    xtest_set = feat_scaled[x_test]
    ytest_set = labels[x_test]

    
    classifier = LogisticRegression(penalty="l1", C=cs[i], solver="saga", multi_class="multinomial").fit(xtrain_set, ytrain_set)
    classifiers.append(classifier)
    
    i = i + 1

    predict = classifier.predict(xtest_set)
    accuracies.append(accuracy_score(ytest_set,predict ))
    '''plot_confusion_matrix(classifier, xtest_set, ytest_set)
    plt.show()'''

#best classifier based on accuracy and confusion matrix
best_classifier = classifiers[np.argmax(accuracies)]
print(np.max(accuracies))
plot_confusion_matrix(best_classifier, feat_scaled, labels)
plt.show()   


#5.In this part we use the 784 raw pixel values themselves as features

#reducing dimension of the images

new_digits = []

for dig in digits:
    dig = dig.reshape(28,28)
    dig = np.array(dig, dtype='uint8')
    new_dig = cv2.resize(dig, (14,14),interpolation = cv2.INTER_AREA)
    new_digits.append(np.ravel(new_dig))

#drop features which are always zero
'''columns = df.columns
print(columns)
for col in columns: 
    print(col)
    if(df[col] == 0).all():
        df.drop(col, inplace=True, axis=1)
   
digits = df.values[:,1:]'''

'''plt.imshow(new_digits[1].reshape(14, 14))
plt.colorbar()
plt.show()'''
#scaling features
scaler = MaxAbsScaler()
digits_scaled = scaler.fit_transform(new_digits)



#train and test set split 
x_train, x_test, y_train, y_test = train_test_split(digits, labels, test_size = 37/42, stratify=labels)

cross = LogisticRegressionCV(Cs=10,penalty="l1", solver="saga",tol=0.001, multi_class="multinomial", max_iter=10000).fit(digits_scaled, labels)

cs = cross.Cs_

i = 0

accuracies = []
classifiers = []

#training and evaluating classifiers
for x_train_cv, x_test_cv in kf.split(x_train, y_train):
    xtrain_set = x_train[x_train_cv]
    ytrain_set = y_train[x_train_cv]

    xtest_set = x_train[x_test_cv]
    ytest_set = y_train[x_test_cv]

    print("Training classifier n",str(i))
    classifier = LogisticRegression(penalty="l1", C=cs[i], solver="saga", tol=0.001 ,multi_class="multinomial", max_iter=10000).fit(xtrain_set, ytrain_set)
    classifiers.append(classifier)
    
    i = i + 1

    predict = classifier.predict(xtest_set)
    accuracies.append(accuracy_score(ytest_set,predict ))
    '''plot_confusion_matrix(classifier, xtest_set, ytest_set)
    plt.show()'''

#best classifier based on accuracy and confusion matrix
best_classifier = classifiers[np.argmax(accuracies)]
print(np.max(accuracies))
plot_confusion_matrix(best_classifier, x_test, y_test)
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
