import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.model_selection import KFold, RandomizedSearchCV
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
import cv2
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit




df = pd.read_csv("mnist.csv")


'''columns = df.columns
print(columns)
for col in columns: 
    print(col)
    if(df[col] == 0).all():
        df.drop(col, inplace=True, axis=1)'''

'''columns = df.columns
print(columns)
for col in columns: 
    print(col)
    if(df[col] == 0).all():
        df.drop(col, inplace=True, axis=1)

'''
print(df.describe())
digits = df.values[:,1:]
labels = df.values[:,0]

new_digits = []

for dig in digits:
    dig = dig.reshape(28,28)
    dig = np.array(dig, dtype='uint8')
    new_dig = cv2.resize(dig, (14,14),interpolation = cv2.INTER_AREA)
    new_digits.append(np.ravel(new_dig))

#drop features which are always zero


'''plt.imshow(new_digits[1].reshape(14, 14))
plt.colorbar()
plt.show()'''
#scaling features
scaler = MaxAbsScaler()
digits_scaled = scaler.fit_transform(new_digits)



#train and test set split 
x_train, x_test, y_train, y_test = train_test_split(digits_scaled, labels, test_size = 37/42, stratify=labels,shuffle=True)

svc = SVC()

dictionary = dict(C=uniform(loc=1,scale=2), kernel=['linear', 'poly', 'rbf'])

cross = RandomizedSearchCV(svc,n_iter=20,n_jobs=4,param_distributions=dictionary,cv=10,verbose=3 ).fit(x_train,y_train)

#cross = LogisticRegressionCV(verbose=3,n_jobs=4,cv=10,penalty="l1", solver="saga", multi_class="multinomial", max_iter=1000)

cross.fit(x_train,y_train)

results = cross.cv_results_

# Get the test scores for each iteration
test_scores = results['mean_test_score']

# Plot the results
plt.plot(test_scores)
plt.xlabel('Iteration')
plt.ylabel('Test Score')
plt.show()


#best classifier based on accuracy and confusion matrix
print("Best params: "+str(cross.best_params_))
print("Score", cross.score(x_test,y_test))
print("Accuracy",accuracy_score(y_test, cross.predict(x_test)))
plot_confusion_matrix(cross, x_test, y_test)
plt.show()



