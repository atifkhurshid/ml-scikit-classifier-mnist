from __future__ import print_function
from preprocessing import *


from sklearn import metrics
from sklearn.linear_model import LogisticRegression

np.set_printoptions(precision=2, linewidth=150, suppress=False)

data = Data()
X_train, y_train = data.load_train_data()
X_test, y_test = data.load_test_data()

print ("Data loaded!")

LR = LogisticRegression(solver='lbfgs')
LR.fit(X_train, y_train)

print ("Logistic Regression complete!")

y_pred = LR.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))