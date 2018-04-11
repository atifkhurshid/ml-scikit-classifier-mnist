# MNIST-Classifier-in-Scikit-learn
Hand-written digit recognition using Scikit-learn

##Preprocessing.py
```
class Data:
  Methods:
  X_train, y_train = load_train_data():
  X_test, y_test = load_test_data()
```

##Classifier.py

Classifier = Logistic Regression
```
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(solver='lbfgs')
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
```
Performance
```
             precision    recall  f1-score   support

          0       0.95      0.98      0.96       980
          1       0.96      0.98      0.97      1135
          2       0.93      0.88      0.90      1032
          3       0.90      0.91      0.90      1010
          4       0.93      0.93      0.93       982
          5       0.90      0.85      0.87       892
          6       0.94      0.95      0.95       958
          7       0.93      0.92      0.92      1028
          8       0.84      0.88      0.86       974
          9       0.90      0.89      0.90      1009

avg / total       0.92      0.92      0.92     10000

[[ 958    0    0    4    0    3    5    2    6    2]
 [   0 1116    3    1    0    1    4    1    8    1]
 [   8   12  906   18    9    5   10   11   50    3]
 [   3    0   19  916    2   23    5   11   24    7]
 [   1    2    5    3  910    0   11    2   10   38]
 [  11    2    1   40   10  754   16    8   39   11]
 [   7    3    7    2    4   17  911    1    6    0]
 [   3    6   24    4    7    1    1  946    5   31]
 [   9   15    7   22   11   26    7   12  854   11]
 [   9    6    2   13   30    4    0   26   16  903]]

```

##Full_train.py
Merge train and test classes and train LR on complete dataset. 
Store model in model.pkl

##Test.py
Input: Image filename
Output: Processed image + Predicted label
