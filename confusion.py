import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras.utils as np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import os

train_data_path = "C:/Users/Nitin/Documents/projects/vaishnavi/skin cancer detection/new version/data/melanoma_cancer_dataset/train"
test_data_path = "C:/Users/Nitin/Documents/projects/vaishnavi/skin cancer detection/new version/data/melanoma_cancer_dataset/test"
train_dir = os.listdir(train_data_path)
test_dir = os.listdir(test_data_path)

# converting to array
def convToArray(data_path, dir):

  """
    This function converts and image to numpy array and resize it to size (96,96)
  """

  x = []   ## Contains Image array
  y = []   ## contains image class ('benign', 'malignant')
  for clas in dir:
    for img in os.listdir(data_path+'/'+clas):
      img_path = data_path+'/'+clas+'/'+img
      img_act = cv.imread(img_path,0)   # here 2nd argument '0' is for reading images in grayscale / or we can say it converts 3D array to 2D array
      # we have to resize the image because it is very large
      img_act = cv.resize(img_act,(96,96))

      x.append(img_act)
      y.append(clas)

  print("*****Converting complete *****")

  return np.asarray(x), np.asarray(y)

## train
x_train, y_train = convToArray(train_data_path, train_dir)

## test 
x_test, y_test = convToArray(test_data_path, test_dir)

# Shaping data
def shapeData(data):
  """
    Data should be numpy array.
  """
  new_data = data.reshape((data.shape[0], 96, 96, 1)).astype('float32')

  # normalize inputs from 0-255 to 0-1
  new_data = new_data / 255

  print("*****Reshaping Complete*****")

  return data

# Shaping train data
x_train_shaped = shapeData(x_train)

# shaping test data
x_test_shaped = shapeData(x_test)

print("*********Shaping data completed********")

# Encoding
## Creating global object for label encoder.
label_encoder = LabelEncoder()

def labelToNum(data):

  """
    This function converts labels to code format or numerical format
  """

  # integer encode
  integer_encoded = label_encoder.fit_transform(data)

  # binary encode
  onehot_encoder = OneHotEncoder(sparse_output=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

  num_classes = len(set(data))

  print("*****Coding Complete*****")

  return onehot_encoded, integer_encoded, num_classes

# Coding train labels/classes
y_train_coded, y_train_integer, num_classes = labelToNum(y_train)


# Coding test labels/classes
y_test_coded, y_test_integer, num_classes = labelToNum(y_test)

# loading model
k_model = keras.models.load_model("trained_model/SKIN_CANCER_DETECTION")

# predicting values
y_pred = k_model.predict(x_test_shaped)

# creating argmax
y_true = y_test_coded.argmax(axis=1)
y_predicted = y_pred.argmax(axis=1)

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_true, y_predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.title("Melanoma Skin Cancer Detection Confusion Matrix")
# Add labels for TP, TN, FP, FN
plt.text(0.01, 0.2, 'True Negative', fontsize=14, color='white', ha='center', va='center', weight='bold')
plt.text(1, 0.2, 'False Positive', fontsize=14, color='white', ha='center', va='center', weight='bold')
plt.text(0.01, 1.2, 'False Negative', fontsize=14, color='white', ha='center', va='center', weight='bold')
plt.text(1, 1.2, 'True Positive', fontsize=14, color='white', ha='center', va='center', weight='bold')
plt.show()