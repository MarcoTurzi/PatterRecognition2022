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


# pixel variance analysis
def getStds(pixData):
    stds = np.std(pixData, axis=0)


    plt.imshow(stds.reshape(28, 28))
    plt.colorbar(label = 'Standard deviation over all samples')
    plt.xlabel('x-coordinate pixel')
    plt.ylabel('y-coordinate pixel')
    plt.show()
    return stds

tds = getStds(digits)
def getMeans(pixData):
    means = np.mean(pixData, axis=0)


    plt.imshow(means.reshape(28, 28))
    plt.colorbar(label = 'Mean over all samples')
    plt.xlabel('x-coordinate pixel')
    plt.ylabel('y-coordinate pixel')
    plt.show()
    return means

means = getMeans(digits)


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
