#!/usr/bin/env python
# coding: utf-8

# # Handwriting Recognition

# ## Import necessary libraries

# In[1]:


from sklearn.model_selection import train_test_split    #to split data
from keras.preprocessing.image import ImageDataGenerator 
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os


# To read input folder

# In[2]:


path= "D:/Project/english_alphabets/"
mylist= os.listdir(path) #to read list of all files from given path
print(len(mylist))


# To read images from sub-folder

# In[3]:


images = []
img_id=[]
a_length = len(mylist)
print(mylist)
for i in range(0,a_length):
    PicList=path+str(mylist[i])
    for filename in os.listdir(PicList):
        img = cv2.imread(PicList+"//"+filename) # to read image
        img = cv2.resize(img,(34,34))  #resize image 34*34
        images.append(img)  #append images in a list
        img_id.append(i)   #append image id in a list
print(len(images))


# In[4]:


images = np.array(images)  #gives array of images
img_id = np.array(img_id)
print(images.shape)
print(img_id.shape)


# ## Spliting dataset

# In[5]:


X_train,X_test,Y_train,Y_test = train_test_split(images,img_id,test_size=0.2)
print(X_train.shape)
print(X_test.shape)


# ## Storing number of samples of each folder in list

# In[6]:


number_of_samples = []
for index in range(0,26):
   # print(len(np.where(Y_train==index)[0]))
    number_of_samples.append(len(np.where(Y_train==index)[0]))
print(number_of_samples)


# ## Converting image into gray image

# In[7]:


def preprocessingimage(img1):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    mean_filter_kernel = np.ones((5,5),np.float64)/(25)
    img1 = img1/255 #normalize    
    return img1


# In[8]:


img1 = preprocessingimage(X_train[2])
img1 = cv2.resize(img1,(100,100))
cv2.imshow("Image Preprocessed1",img1)
cv2.waitKey(0)


# ## Preprocessing of all Images

# In[8]:


X_train = np.array(list(map(preprocessingimage, X_train)))
X_test = np.array(list(map(preprocessingimage, X_test)))


# In[9]:


X_train.shape


# In[10]:


X_train1 = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test1 = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)


# In[11]:


print(X_train1.shape)


# In[12]:


input_shape1 = X_train1.shape[1:]
input_shape1


# ## To Generic Image

# In[13]:


datagen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
datagen.fit(X_train1)


# We need to do one hot encoding here because we have 26 classes and we should expect the shape[1] of y_train and y_test to change from 1 to 26

# In[14]:


#one hot encoding
Y_train1 = to_categorical(Y_train,a_length)
Y_test1 = to_categorical(Y_test,a_length)


# In[15]:


Y_train1.shape


# ## Creating Model

# In[16]:


def myModel():
    model = Sequential()
    model.add((Conv2D(16,(3,3),input_shape=input_shape1,padding='same',activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))  #17
    model.add((Conv2D(32,(3,3),padding='same',activation='relu')))  #17
    model.add(MaxPooling2D(pool_size=(2,2))) #8
    model.add((Conv2D(64,(3,3),padding='same',activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same')) #4
    model.add((Conv2D(96,(3,3),padding='same',activation='relu')))
    model.add(Dropout(0.50)) #reduces overfitting
    model.add(Flatten())   
    model.add(Dense(800,activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(400,activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(a_length,activation='softmax'))
    model.compile(Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# In[17]:


model = myModel()
print(model.summary())


# ## Training model

# In[18]:


hist = model.fit(datagen.flow(X_train1,Y_train1), epochs = 70, validation_data=(X_test1, Y_test1))


# ## Graph of Accuracy & Loss

# In[19]:


plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy']) 
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')


# In[20]:


score = model.evaluate(X_test1,Y_test1)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])


# # Vgg16 Model Prediction

# In[24]:


fname = "D:/Final Project/model_test_vgg16_accuracy.h5"
model.save_weights(fname)
checkpoint = ModelCheckpoint(fname,monitor='val_acc')


# In[29]:


img11 = image.load_img("D:/Final Project/Output/val/E/74.jpg",target_size=(34,34))
img11 = np.asarray(img11)
plt.imshow(img11)
img11 = np.expand_dims(img11, axis=0)


# ## Testing

# In[24]:


fname = "D:/Project/test/model11.h5"
model.save_weights(fname)


# In[21]:


path2= "D:/Project/test/"
mylist2= os.listdir(path2)
print(len(mylist2))


# In[22]:


images2 = []
for filename2 in mylist2:
    img2 = cv2.imread(path2+"//"+filename2)
    img2 = cv2.resize(img2,(34,34))
    images2.append(img2)
    print(filename2,end=", ")  


# In[23]:


images3 = np.array(images2)
print(images3.shape)


# In[24]:


def preprocessingimage1(img2):
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    mean_filter_kernel = np.ones((5,5),np.float64)/(25)
    img2 = img2/255 #normalize
    return img2


# In[33]:


'''img3 = cv2.imread("D:\\Project\\test3\\2F.jpg")   
img3 = preprocessingimage(images3[1])
img3= cv2.resize(img3,(200,200))
cv2.imshow("image processed2",img3)
cv2.waitKey(0)'''


# In[25]:


X_train2 = np.array(list(map(preprocessingimage1, images3)))
X_test2 = np.array(list(map(preprocessingimage1, images3)))
#X_Validation2 = np.array(list(map(preprocessingimage1, images3)))


# In[26]:


img3 = preprocessingimage1(images3[0])
img3= cv2.resize(img3,(200,200))
cv2.imshow("image processed2",img3)
cv2.waitKey(0)


# In[27]:


X_test2.shape


# In[28]:


X_train3 = X_train2.reshape(X_train2.shape[0],X_train2.shape[1],X_train2.shape[2],1)
X_test3 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1],X_test2.shape[2],1)
#X_Validation3 = X_Validation2.reshape(X_Validation2.shape[0],X_Validation2.shape[1],X_Validation2.shape[2],1)
print(X_train3.shape)


# In[29]:


input_shape2 = X_train3.shape[1:]
input_shape2


# In[30]:


pred2 = model.predict(X_train3)


# In[32]:


y_pred = model.predict_classes(X_train3)
print(y_pred)


# In[33]:


for i in range(len(y_pred)):
    
    lst=['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for j in range(26):
        if y_pred[i] == j:
            print(lst[j],end=" ")


# ## Word segmentation

# In[34]:


cnt=0
img = cv2.imread("D:/Project/test6/file.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

i=0 #iterating loop for bounding rect and crop image
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    #cv2.imshow('Each image box',img)
    crop_img = img[y:y+h,x:x+w]
    cv2.imwrite("D:/Project/test5/"+str(i)+".jpg",crop_img)#saving croped image in folder
    cv2.imshow("Croped letter",crop_img)
    cv2.waitKey(0)
    i=i+1


# In[35]:


path2= "D:/Project/test5/"
mylist2= os.listdir(path2)
print(len(mylist2))


# In[36]:


images2 = []
for filename2 in mylist2:
    img2 = cv2.imread(path2+"//"+filename2)
    img2 = cv2.resize(img2,(34,34))
    images2.append(img2)
    print(filename2,end=", ") 


# In[37]:


images3 = np.array(images2)
print(images3.shape)


# In[38]:


def preprocessingimage1(img2):
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    #img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mean_filter_kernel = np.ones((5,5),np.float64)/(25)
    img2 = img2/255 #normalize
    return img2


# In[39]:


X_train2 = np.array(list(map(preprocessingimage1, images3)))
X_test2 = np.array(list(map(preprocessingimage1, images3)))


# In[40]:


X_train3 = X_train2.reshape(X_train2.shape[0],X_train2.shape[1],X_train2.shape[2],1)
X_test3 = X_test2.reshape(X_test2.shape[0],X_test2.shape[1],X_test2.shape[2],1)
print(X_train3.shape)


# In[41]:


input_shape2 = X_train3.shape[1:]
input_shape2


# In[42]:


pred2 = model.predict(X_train3)


# In[43]:


y_pred = model.predict_classes(X_train3)
print(y_pred)


# In[44]:


for i in range(len(y_pred)):
    
    lst=['A','B','C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for j in range(26):
        if y_pred[i] == j:
            print(lst[j],end=" ")


# In[ ]:




