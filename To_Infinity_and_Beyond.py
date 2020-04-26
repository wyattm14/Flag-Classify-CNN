# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 13:12:52 2020

@author: ThinkPad X1
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:43:34 2020

@author: ThinkPad X1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import random
import pickle

#################################################################
############# SPLIT IMAGES ######################################
#################################################################
import image_slicer
import glob
from PIL import Image

def crop(im,height,width):
    #im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            # print (i,j)
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)
            
if __name__=='__main__':
    # change the path and the base name of the image files 
    imgdir = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\ISS_AMERICAN\\"
    basename = os.listdir(imgdir)
    #basename = "tester.jpg"
    name_list = []
    filelist = []
    for k in basename:
        filelist.append(os.path.join(imgdir,k))
        name_list.append(str(k))
    print(filelist)
    new_count =1
    for infile in filelist:
        if not "jpg" in infile:
            continue
        #infile='/Users/alex/Documents/PTV/test_splitter/cal/Camera 1-1-9.tif'
        #print (filenum) # keep the numbers as we change them here
        print (infile)
        
        im = Image.open(infile)
        imgwidth, imgheight = im.size
        print ('Image size is: %d x %d ' % (imgwidth, imgheight))
        height = imgheight//6
        width =  imgwidth//6
        start_num = 0
        count = 1
        new_count+=1
        for k,piece in enumerate(crop(im,height,width),start_num):
            #print k
            #print piece
            img=Image.new('RGB', (width,height), 255)
            #print img
            img.paste(piece)
            #path = os.path.join("cam%d_1%05d.tif" % (int(k+1),filenum))
            #img.save(path)
            #os.rename(path,os.path.join("cam%d.1%05d" % (int(k+1),filenum)))
            photoName = str(k)
            img.save('C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\SPLIT\\'+infile[55:]+str(count)+'.jpg', "JPEG")
            count += 1

#################################################################
############# TEACHING CNN TO FIND FLAGS IN GENERAL #############
############# MODEL 1: FLAG/NONFLAG #############################
#################################################################

img_dir = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\First Model\\" # set directory 
data_path = os.path.join(img_dir,'*g') 
    
CATEGORIES = ["Flags", "NotFlags"]

IMG_SIZE = 50

for category in CATEGORIES : 
	path = os.path.join(img_dir, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) 
        
training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(img_dir, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) 
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
            
create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

################################################################
############# TRAINING THE CNN FLAG/NONFLAG MODEL ########
################################################################

# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt

# Opening the files about data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# normalizing data (a pixel goes from 0 to 255)
X = X/255.0

# Building the model
model = Sequential()
# 3 convolutional layers
model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))

model.add(Dense(128))
model.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model.add(Dense(13))
model.add(Activation("softmax"))

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

y = np.stack(y)

# TRAINING THE MODEL, with 100 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, y, batch_size=32, epochs=100, validation_split=0.25) 
    # we switched epochs around from 40, 50, 75, and 100 and chekced accuracy as we went

# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

#################################################################
############# FLAG/NO FLAG MODEL - TEST ON NASA PHOTOS ##########
#################################################################
DATADIR1 = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\SPLIT\\" 

basename2 = os.listdir(DATADIR1)
split_NAMES = [] # append each flag name to split_names list
for k in basename2:
    split_NAMES.append(str(k))

# create dataset
data_path1 = os.path.join(DATADIR1)
testing_data = []
for img in os.listdir(data_path1):
	try :
		img_array = cv2.imread(os.path.join(data_path1, img), cv2.IMREAD_COLOR)
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		testing_data.append([new_array])
	except Exception as e:
		pass
TEST = np.array(testing_data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pred = model.predict_classes(TEST)
print(pred)
index = 0
for i in TEST:    
    X_new = TEST[index].squeeze()
    print("Prediction: ", pred[index]) 
    plt.imshow(X_new)
    plt.show()
    index += 1
    
# index 0 values
count = 0
flagIndex = []
for value in pred:
    if value == 0:
        flagIndex.append(split_NAMES[count])
    count += 1

#################################################################
############# SAVING FLAG/NONFLAG PREDICTIONS TO A FOLDER #######
#################################################################

import shutil

source = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\SPLIT\\"
destination = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\PREDICTED_MODEL1\\"

for files in os.walk(source, topdown = True):
    for thing in files[2]:
        if thing in flagIndex:
            #print(thing)
            fileStr = source + thing
            shutil.copy(fileStr,destination)
    
#################################################################
############# CREATING TRAINING DATA FOR SPECIFIC FLAGS #########
#################################################################

img_dir = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\Specific Flags\\" # set directory 
data_path = os.path.join(img_dir,'*g') 
    
CATEGORIES = ["America", "Canada","Japan","Russia"]

IMG_SIZE = 50

for category in CATEGORIES : 
	path = os.path.join(img_dir, category)
	for img in os.listdir(path):
		img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) 
        
training_data = []

def create_training_data():
	for category in CATEGORIES :
		path = os.path.join(img_dir, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try :
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR) 
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass
            
create_training_data()

random.shuffle(training_data)

X = [] #features
y = [] #labels

for features, label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)


################################################################
############# TRAINING THE CNN MODEL FOR SPECIFIC FLAGS ########
################################################################

# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

# Opening the files about data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# normalizing data (a pixel goes from 0 to 255)
X = X/255.0

# Building the model
model2 = Sequential()
# 3 convolutional layers
model2.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model2.add(Activation("relu"))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(64, (3, 3)))
model2.add(Activation("relu"))
model2.add(MaxPooling2D(pool_size=(2,2)))

model2.add(Conv2D(64, (3, 3)))
model2.add(Activation("relu"))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

# 2 hidden layers
model2.add(Flatten())
model2.add(Dense(128))
model2.add(Activation("relu"))

model2.add(Dense(128))
model2.add(Activation("relu"))

# The output layer with 13 neurons, for 13 classes
model2.add(Dense(13))
model2.add(Activation("softmax"))

# Compiling the model using some basic parameters
model2.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])

y = np.stack(y)


# TRAINING THE MODEL, with 100 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history2 = model2.fit(X, y, batch_size=32, epochs=100, validation_split=0.25)

# Printing a graph showing the accuracy changes during the training phase
print(history2.history.keys())
plt.figure(1)
plt.plot(history2.history['acc'])
plt.plot(history2.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')


##################################################################
################# TESTING SPECIFIC FLAGS ON NASA DATA ############
##################################################################

DATADIR2 = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\PREDICTED_MODEL1\\" 

basename3 = os.listdir(DATADIR2)
final_NAMES = [] # append each flag name to split_names list
for k in basename3:
    final_NAMES.append(str(k))

# create dataset
data_path1 = os.path.join(DATADIR2)
testing_data = []
for img in os.listdir(data_path1):
	try :
		img_array = cv2.imread(os.path.join(data_path1, img), cv2.IMREAD_COLOR)
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
		testing_data.append([new_array])
	except Exception as e:
		pass
TEST = np.array(testing_data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pred = model2.predict_classes(TEST)
print(pred)
index = 0
for i in TEST:    
    X_new = TEST[index].squeeze()
    print("Prediction: ", pred[index]) 
    plt.imshow(X_new)
    plt.show()
    index += 1

#CATEGORIES = ["America", "Canada","Japan","Russia"]
# index 0 values
count = 0
america_index = []
canada_index = []
japan_index = []
russia_index = []
for value in pred:
    if value == 0:
        america_index.append(final_NAMES[count])
    if value == 1:
        canada_index.append(final_NAMES[count])
    if value == 2:
        japan_index.append(final_NAMES[count])
    if value == 3:
        russia_index.append(final_NAMES[count])
    count += 1

#################################################################
#################################################################
############# SAVING FINAL PREDICTIONS TO A FOLDER #############
#################################################################

source = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\PREDICTED_MODEL1\\"
destination_america = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\FINAL_PREDICT\\America\\"
destination_canada = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\FINAL_PREDICT\\Canada\\"
destination_japan = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\FINAL_PREDICT\\Japan\\"
destination_russia = "C:\\Users\\ThinkPad X1\\Downloads\\Wyatt_Train\\FINAL_PREDICT\\Russia\\"

import shutil

# america
for files in os.walk(source, topdown = True):
    for thing in files[2]:
        if thing in america_index:
            #print(thing)
            fileStr = source + thing
            shutil.copy(fileStr,destination_america)

#canada            
for files in os.walk(source, topdown = True):
    for thing in files[2]:
        if thing in canada_index:
            #print(thing)
            fileStr = source + thing
            shutil.copy(fileStr,destination_canada)

#japan           
for files in os.walk(source, topdown = True):
    for thing in files[2]:
        if thing in japan_index:
            #print(thing)
            fileStr = source + thing
            shutil.copy(fileStr,destination_japan)

#russia          
for files in os.walk(source, topdown = True):
    for thing in files[2]:
        if thing in russia_index:
            #print(thing)
            fileStr = source + thing
            shutil.copy(fileStr,destination_russia)