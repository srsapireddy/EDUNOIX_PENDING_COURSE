# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:05:03 2020

@author: Rahul Sapireddy
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:09:22 2020

@author: Rahul Sapireddy
"""

# importing the required packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor
# from keras.preprocessing.image import load_img

# get the list of images from the dataset path
image_paths = list(paths.list_images("C:\\Users\\Rahul Sapireddy\\Desktop\\EDUNOIX\\Deep Learning//image_classification\\datasets\\animals"))

print("INFO: loading and preprocessing")

# loading and preprocessing images using classes created
dp = DsPreprocessor(32,32)
dl = DsLoader(preprocessor=[dp])
(data, labels) = dl.load(image_paths)

# Reshape from (3000,32,32,3) to (3000, 32*32*3=3072 integers)
data = data.reshape((data.shape[0], 3072))
print("INFO: Memory size of feature matrix {:.1f}MB".format(data.nbytes/(1024*1000)))

# Encode the string labels as integers like 0,1,2.
le = LabelEncoder()
labels = le.fit_transform(labels) # To convert labels in strings to integers

print("INFO: splitting the dataset")
# split 25 percent for testing and rest for training
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 40)

print("INFO: training the model")

# looping through L1,L2 and without regularization
for regularization in ("l1","l2",None):
    print("INFO: training the model with {} regularization".format(regularization))
    model = SGDClassifier(loss="log", penalty=regularization, max_iter=90, learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX,trainY)
    print("INFO: evaluating the model")
    """
    # Evaluating and printing the report based on test data classification
    print(classification_report(testY, model.predict(testX), target_names = le.classes_))
    # target_names [optional] -> Are list of strings. Gives the actual label names rather than the label indexes.
    """
    # Evaluating the accuracy and print it
    accuracy = model.score(testX, testY)
    print("INFO: evaluating accuracy of the model with {} regularization is {:.2f}%".format(regularization, accuracy*100))

