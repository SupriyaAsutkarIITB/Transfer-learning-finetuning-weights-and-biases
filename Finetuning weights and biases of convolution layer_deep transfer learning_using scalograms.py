#!/usr/bin/env python
# coding: utf-8

# # Transfer learning algorithm to observe accuracy over generalisation problem
# 
# Written by: Supriya Asutkar

# In[1]:


#Import all necessary libraries

import numpy as np
import numpy.random as r
import pandas as pd;
from pandas import ExcelWriter
from pandas import ExcelFile
import math

from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt;


import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import numpy as np
get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[2]:


# Read re-training data (IMS)

class_arr_P = ['H','F']

images = np.empty((4000, 513, 8))         # Spectrogram size=513*18
labels = np.empty(4000)

idx = 0
for i in range(2):
    for j in range(0,2000):
        x = pd.read_csv('Q:/Supriya_ML/Supriya/Deep TL_Exp_Sys/CWRU_IMS/Test_spectrogram/Spec_IMS_'+ str(class_arr_P[i]) + '_'+ str(j) + '.csv',header = None)
        x_np = x[0:513].to_numpy()
        images[idx,:,:] = x_np
        l_np = x[513:514][0].to_numpy()
        labels[idx] = l_np
        idx = idx+1

labels = labels.astype(int)


# In[3]:


#Preprocessing of the data and split the data

for i in range(len(images)):
    maximum = np.amax(images[i])
    minimum = np.amin(images[i])
    images[i] = (images[i] - minimum)/(maximum - minimum)
    
    
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.30)


print(train_images.shape)
print(test_images.shape)


# In[4]:


train_images = train_images.reshape(2800,513,8,1)
train_labels = train_labels.reshape(2800,1)
test_images = test_images.reshape(1200,513,8,1)
test_labels = test_labels.reshape(1200,1)

test_labels.shape


# In[7]:


#Load the base model (trained on CWRU dataset)

import keras
import keras.utils
from keras import utils as np_utils
base_model = keras.models.load_model("Q:/Supriya_ML/Supriya/Deep TL_Exp_Sys/CWRU_IMS/CNN_CWRU_IMS_20.h5")

#base_model_2.layers
#len(base_model_2.layers)
base_model.summary()


# In[8]:


# Freeze all lower layers

for layer in base_model.layers[7:10]:

    layer.trainable = False

for layer in base_model.layers:

    print(layer, layer.trainable)


# In[9]:


base_model.summary()


# In[10]:


#opt = SGD(learning_rate=0.01)
opt = Adam(learning_rate=0.001)
base_model.compile(optimizer=opt, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])


# In[11]:


#Fit the model
import time
s=time.time()
get_ipython().run_line_magic('memit', 'training=base_model.fit(train_images,train_labels,validation_split=0.2, epochs=20, batch_size=200)   #epoch=3')
print(time.time()-s)


# In[12]:


#Visualize training/validation loss and accuracy

import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 14}
labelfont = {'family' : 'Times', 'size'   : 16}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

plt.plot(training.history["accuracy"],linestyle='-',color='blue', linewidth = 2,label = 'Training accuracy')
plt.plot(training.history['val_accuracy'],linestyle='-',color='orange', linewidth = 2,label = 'Validation accuracy')
#plt.plot(training.history['loss'],linestyle='-',color='green', linewidth = 2,label = 'Training loss')
#plt.plot(training.history['val_loss'],linestyle='-',color='red', linewidth = 2,label = 'Validation_loss')
#plt.title("model accuracy")

plt.xlabel('Epoch', **labelfont)
plt.ylabel("Accuracy", **labelfont)
#plt.ylabel("Normalized amplitude (a.u.)", **labelfont)
#plt.ylabel('Z ($V^2$/Hz)', fontsize=18)

plt.xticks(**tickfont)
plt.yticks(**tickfont)
#ax.yaxis.grid(color='gray', linestyle='dashed')

plt.grid(b=True, which='major', color='darkgray', linestyle='-')
plt.grid(b=True, which='minor', color='darkgray', linestyle='-', alpha=0.2)

#plt.text(12,3,'(a)', fontsize=18, fontweight='bold')

plt.xlim((0,19))
#plt.ylim((0.6,1))
#fig= plt.figure(figsize=(16,8))

ax.spines['bottom'].set_color('k')
ax.spines['top'].set_color('k') 
ax.spines['right'].set_color('k')
ax.spines['left'].set_color('k')

plt.legend(prop={'size': 12}, loc = 'lower right')

plt.show()


# In[20]:


#Visualize training/validation loss

import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 14}
labelfont = {'family' : 'Times', 'size'   : 16}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

#plt.plot(training.history["accuracy"],linestyle='-',color='blue', linewidth = 2,label = 'Training accuracy')
#plt.plot(training.history['val_accuracy'],linestyle='-',color='orange', linewidth = 2,label = 'Validation accuracy')
plt.plot(training.history['loss'],linestyle='-',color='green', linewidth = 2,label = 'Training_loss')
plt.plot(training.history['val_loss'],linestyle='-',color='red', linewidth = 2,label = 'Validation_loss')
#plt.title("model accuracy")

plt.xlabel('Epoch', **labelfont)
plt.ylabel("Loss", **labelfont)
#plt.ylabel("Normalized amplitude (a.u.)", **labelfont)
#plt.ylabel('Z ($V^2$/Hz)', fontsize=18)

plt.xticks(**tickfont)
plt.yticks(**tickfont)
#ax.yaxis.grid(color='gray', linestyle='dashed')

plt.grid(b=True, which='major', color='darkgray', linestyle='-')
plt.grid(b=True, which='minor', color='darkgray', linestyle='-', alpha=0.2)

#plt.text(12,3,'(a)', fontsize=18, fontweight='bold')

plt.xlim((0,19))
#plt.ylim((0.6,1))
#fig= plt.figure(figsize=(16,8))

ax.spines['bottom'].set_color('k')
ax.spines['top'].set_color('k') 
ax.spines['right'].set_color('k')
ax.spines['left'].set_color('k')

plt.legend(prop={'size': 12}, loc = 'upper right')

plt.show()


# In[13]:


base_model.save("TLCNN_IMS.h5")


# In[14]:


#load the model

import keras
import keras.utils
from keras import utils as np_utils
reconstructed_model = keras.models.load_model("TLCNN_IMS.h5")


# In[15]:


#use model for test data
predictions = reconstructed_model.predict(test_images)
#reconstructed_model.fit(test_images, test_labels)
#print(test_labels,predictions)
#print(predictions)
for i in range(len(test_labels)):
    if predictions[i] < 0.5:
        predictions[i] =0
    else:
            predictions[i] =1
    print("Test_images=%s, Predicted=%s" % (test_labels[i],(predictions[i])))


# In[16]:


from sklearn.metrics import confusion_matrix
conf_arr = confusion_matrix(test_labels,predictions, sample_weight=None, normalize=None)
tp, fn, fp, tn = conf_arr.reshape(-1)
print(tp, fn, fp, tn)
acc = (tp+tn)/(tp+fn+fp+tn)
print(acc)
print(conf_arr)


# In[17]:


import matplotlib.pyplot as plt
sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 14}
labelfont = {'family' : 'Times', 'size'   : 16}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

df_cm = pd.DataFrame(conf_arr, range(2), range(2))

sns.heatmap(df_cm, cmap='Blues', annot=True,  fmt='.0f', annot_kws={"size": 16}) # font size

#plt.savefig('Confusion matrix_binary class.png',dpi = 300, bbox_inches='tight')
plt.show()


# In[18]:


from sklearn.metrics import classification_report
matrix = classification_report(test_labels, predictions)
print('Classification report : \n',matrix)


# In[19]:


from sklearn.metrics import f1_score
f1_score(test_labels, predictions)


# In[ ]:




