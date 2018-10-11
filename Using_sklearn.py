import sklearn.linear_model 
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt
import cv2

learning_rate = 1e-3
epochs = 1000
threshold = 0.995

"""Read image as grayscale"""
river = cv2.imread('river.png',0)

test = []
train = []

label_train = [] 
label_test = []

k = 0

""" Dividing top half of image as training set and bottom half as testing set"""

for i in range(int(river.shape[0]/2)):
    for j in range(int(river.shape[1])):
        train.append(river[i][j])
        if ( river[i][j] >= 245 and j >= 130 and j <= 230):
            # print (i,j,river[i][j])
            label_train.append(1.0)
        else:
            label_train.append(0.0)

for i in range(int(river.shape[0]/2)):
    for j in range(int(river.shape[1])):
        test.append(river[i+int(river.shape[0]/2)][j])

        if ( river[i][j] >= 245 and j >= 130 and j <= 230):
            # print (i,j,river[i][j])
            label_test.append(1.0)
        else:
            label_test.append(0.0)

train = np.reshape(train,(-1,1))
test = np.reshape(test,(-1,1))
clf = sklearn.linear_model.LogisticRegression()

clf.fit(train,label_train)

pred_y = clf.predict(test)

temp = [[0 for i in range(river.shape[1])] for j in range (int(river.shape[0]/2))]

k = 0

for i in range (int(river.shape[0]/2)):
    for j in range (river.shape[1]):
        if (pred_y[k] >= threshold):
            # print (pred_y[k], i, j)
            temp[i][j] = 255
        k = k + 1

img = np.array(temp)

im = Image.fromarray(img.astype('uint8')).convert('L')
im.save('result_image.png')
im.show()

riv = cv2.imread('river.png')
cv2.imshow('river',riv)
cv2.waitKey(0)
cv2.destroyAllWindows()

param = []

print("Accuracy - ", clf.score(test,label_test))    
