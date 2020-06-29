# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:46:04 2020

@author: Rahul Sapireddy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:09:22 2020

@author: Rahul Sapireddy
"""

# importing the required packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor
from keras.preprocessing.image import load_img,img_to_array

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
# training the KNN classifier using 75 percent of training data
model = KNeighborsClassifier(n_neighbors = 5, n_jobs = -1)
# n_jobs -> number of parallel neighbor searches done by the KNN algorithm. Giving it complete CPU cores to perform the task.
model.fit(trainX,trainY)

"""
print("INFO: evaluating the model")
# Evaluating and printing the report based on test data classification
print(classification_report(testY, model.predict(testX), target_names = le.classes_))
# target_names [optional] -> Are list of strings. Gives the actual label names rather than the label indexes.
"""

# a simple array of clas names in animals dataset
animals_classes = ['cat', 'dog', 'panda']

# loading the unknown image for prediction
unknown_image = load_img("C:\\Users\\Rahul Sapireddy\\Desktop\\EDUNOIX\\Deep Learning\\\images\\test1.png")

# resize the image as 32x32 pixels
unknown_image = unknown_image.resize((32,32))

# convet the resized image to array
unknown_image_array = img_to_array(unknown_image)

unknown_image_array = unknown_image_array.reshape(1,-1) # converting single row array
#unknown_image_array = unknown_image_array.reshape(1,3072)

# do the prediction using model
prediction = model.predict(unknown_image_array)
print("The predicted animal is ")
# print the coresponding label from the array animals_classes
print(str(animals_classes[int(prediction)]))



