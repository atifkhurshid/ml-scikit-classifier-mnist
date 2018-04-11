from __future__ import print_function
from preprocessing import *

import random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier

np.set_printoptions(precision=2, linewidth=150, suppress=False)


data = Data()
X_train, y_train = data.load_train_data()
X_test, y_test = data.load_test_data()

print ("Data loaded!")

SGD = SGDClassifier()
SGD.fit(X_train, y_train)
y_pred = SGD.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
