from __future__ import print_function
from preprocessing import *

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

np.set_printoptions(precision=2, linewidth=150, suppress=False)

data = Data()
X_train, y_train = data.load_train_data()
X_test, y_test = data.load_test_data()
X = np.append(X_train,X_test)
y = np.append(y_train, y_test)
X = X.reshape((int(X.shape[0]/784),784))

print ("Data loaded!")

LR = LogisticRegression(solver='lbfgs')
LR.fit(X, y)

joblib.dump(LR, "model.pkl")
print("Training complete")
