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

svc = SVC(kernel="rbf")
lr = LogisticRegression(penalty="l1", solver="saga", multi_class="multinomial",tol=0.001)

dictionary = dict(C=uniform(loc=1,scale=2))
dictionary_lr = dict(C=uniform(loc=1,scale=2) )

cross_SVC = RandomizedSearchCV(svc,n_iter=50,n_jobs=4,param_distributions=dictionary,cv=10,verbose=3 ).fit(x_train,y_train)

cross_LR = RandomizedSearchCV(lr,n_iter=50,n_jobs=4,param_distributions=dictionary_lr,cv=10,verbose=3 ).fit(x_train,y_train)

#cross = LogisticRegressionCV(verbose=3,n_jobs=4,cv=10,penalty="l1", solver="saga", multi_class="multinomial", max_iter=1000)

lr = cross_LR.best_estimator_
svc = cross_SVC.best_estimator_

C_lr = cross_LR.cv_results_["param_C"]
C_svc = cross_SVC.cv_results_["param_C"]

score_lr = cross_LR.cv_results_["mean_test_score"]
score_svc = cross_SVC.cv_results_["mean_test_score"]

dict_order_lr = {c:s for c,s in zip(C_lr, score_lr )}
dict_order_svc = {c:s for c,s in zip(C_svc, score_svc)}


dict_order_lr = dict(sorted(dict_order_lr.items(), key=lambda item: item[0]))
dict_order_svc = dict(sorted(dict_order_svc.items(), key=lambda item: item[0]))


plt.figure(figsize=(10,6))
plt.plot([c for c,k in dict_order_lr.items()], [k for c,k in dict_order_lr.items()], label="Logistic Regression")
plt.plot([c for c,k in dict_order_svc.items()],[k for c,k in dict_order_svc.items()], label="SVC")
plt.xlabel("Value of parameter C")
plt.ylabel("Average accuracy score")
plt.title("Randomized Search Results")
plt.legend()
plt.show()
# Get the test scores for each iteration
'''plt.figure(figsize=(10, 6))
plt.plot(range(1, len(logistic_results['mean_test_score'])+1), logistic_results['mean_test_score'], label="Logistic Regression")
plt.plot(range(1, len(svc_results['mean_test_score'])+1), svc_results['mean_test_score'], label="SVC")
plt.xlabel('Iteration')
plt.ylabel('Mean Test Score')
plt.title('Random Search Results')
plt.legend()
plt.show()'''

lr.fit(x_train,y_train)
svc.fit(x_train,y_train)


#best classifier based on accuracy and confusion matrix
print("SVC results:")
print("Best params: "+str(cross_SVC.best_params_))
print("Accuracy",accuracy_score(y_test, svc.predict(x_test)))
plot_confusion_matrix(svc, x_test, y_test)
plt.show()

print("LR results:")
print("Best params: "+str(cross_LR.best_params_))
print("Accuracy",accuracy_score(y_test, lr.predict(x_test)))
plot_confusion_matrix(lr, x_test, y_test)
plt.show()


