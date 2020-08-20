import numpy as np
import matplotlib.pyplot as plt #no need in this code
import os
import cv2
import random
import pickle

#Download Kaggle Cat vs Dog ZIP file from Microsoft or Kaggle Website
#Extract the downloaded ZIP file

datadir = r"C:\Users\pc\Desktop\Tenserflow\Images" #directry of training images of the file which extrcted 
categories = ["Dog", "Cat"] #must have folders named as elements of this array

training_data = []
img_size = 100

def create_training_data():
    for category in categories:
        path = os.path.join(datadir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

x =[]
y =[]

for features ,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1,  img_size, img_size, 1)  


pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("x.pickle", "rb")
x = pickle.load(pickle_in)
