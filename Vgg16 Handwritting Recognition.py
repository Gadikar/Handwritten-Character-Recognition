#!/usr/bin/env python
# coding: utf-8

# In[30]:


import splitfolders


# Storing training and testing datasets in output folder

# In[31]:


input_folder="D:/Final Project/Alphabates/"
output_folder="D:/Final Project/Output/"
splitfolders.ratio(input_folder,output_folder,seed=35,ratio=(.8,.2))


# In[32]:


help(splitfolders.ratio)


# Importing necessary libraries

# In[33]:


import os
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# Path for training and testing datasets

# In[34]:


image_size = [34, 34]

train = 'D:/Final Project/Output/train/'
test = 'D:/Final Project/Output/val/'


# In[35]:


train_set  = os.listdir(train)
num_samples= len(train_set)
print(num_samples)


# Re-size all the images to this

# In[36]:


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

vgg = VGG16(input_shape = image_size + [3], weights='imagenet', include_top=False)
vgg


# don't train existing weights

# In[37]:


for layer in vgg.layers:
    layer.trainable = False


# In[38]:


from glob import glob
# useful for getting number of output classes and  * count all image
glob_value = glob('D:/Final Project/Output/train/*')


# In[39]:


oneD_array = Flatten()(vgg.output)


# Create a model object

# In[40]:


prediction = Dense(len(glob_value), activation='softmax')(oneD_array)

model_test = Model(inputs=vgg.input, outputs=prediction)


# In[41]:


model_test.summary()


# Tell the model_test what cost and optimization method to use

# In[42]:


model_test.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[43]:


# Use the Image Data Generator to import the images from the dataset
datagen_test = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            shear_range=0.1,
                            rotation_range=10)
datagen_test = ImageDataGenerator(rescale = 1./255)


# Make sure you provide the same target size as initialied for the image size

# In[44]:



training_set = datagen_test.flow_from_directory(train,
                                                 target_size = (34, 34),
                                                 class_mode = 'categorical')


# In[45]:


test_set = datagen_test.flow_from_directory(test,
                                            target_size = (34, 34),
                                            class_mode = 'categorical')


# In[46]:


hist_test = model_test.fit(training_set,validation_data=test_set,epochs=5,validation_steps=len(test_set))


# In[47]:


plt.figure(1)
plt.plot(hist_test.history['loss'])
plt.plot(hist_test.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(hist_test.history['accuracy'])
plt.plot(hist_test.history['val_accuracy']) 
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')


# Calculate Accuracy of model_test

# In[48]:


score = model_test.evaluate(test_set)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])


# In[51]:


# save it as a h8 file
import tensorflow as tf

from keras.models import load_model

fname = model_test.save('D:/Final Project/model_test_vgg16_accuracy2.h5')
#model_test.save_weights(fname)
#checkpoint = ModelCheckpoint(fname,monitor='val_acc')


# In[52]:


glob_value


# In[53]:


test_set


# In[54]:


y_pred = model_test.predict(test_set)
y_pred
#print(len(pred))
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)


# In[ ]:




