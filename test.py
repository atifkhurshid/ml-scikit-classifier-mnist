from __future__ import print_function
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import PIL.ImageOps

np.set_printoptions(precision=2, suppress=False, threshold=10000)

height = 28
width = 28


def print_results(y_pred):
    print ("Predictions")
    for i in range(len(y_pred)):
        print ("Image ", i, ": ", y_pred[i])

LR = joblib.load("model.pkl")

img = Image.open('test.jpg').convert('L')
img = img.resize((width, height))
img = PIL.ImageOps.invert(img)

image = [np.asarray(img, dtype=np.uint8).reshape(width*height)]

y_pred = LR.predict(image)
plt.imshow(np.reshape(image, (width, height)))

plt.gray(),plt.show()

print_results(y_pred)
