import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler



df = pd.read_csv("mnist(1).csv")
df_values = df.values
print(df_values)
labels = df_values[:,0]
digits = df_values[:,1:]
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
ink_mean = [np.round(np.mean(ink_f[labels == i]) ,2) for i in range(10)]
print(ink_mean)
ink_std = [np.round(np.std(ink_f[labels == i]),2) for i in range(10)]
print(ink_std)

scaler= StandardScaler()
digits_scaled = scaler.fit_transform(digits)
'''ink_f_scaled = np.array([np.sum(row) for row in digits])
'''
ink_f_scaled = scale(ink_f)
ink_mean = [np.round(np.mean(ink_f_scaled[labels == i]) ,2) for i in range(10)]
print(ink_mean)
ink_std = [np.round(np.std(ink_f_scaled[labels == i]),2) for i in range(10)]
print(ink_std)









