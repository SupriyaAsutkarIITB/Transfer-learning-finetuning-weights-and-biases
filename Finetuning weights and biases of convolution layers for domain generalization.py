#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os import listdir
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statistics
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
get_ipython().run_line_magic('load_ext', 'memory_profiler')


# In[2]:


get_ipython().run_line_magic('cd', '..')
model = tf.keras.models.load_model("CWRU-IMS_1D.h5")
model.summary()


# In[3]:


model.layers[0].trainable = True
model.layers[1].trainable = True
model.layers[2].trainable = True
model.layers[3].trainable = True
model.layers[4].trainable = True
model.layers[5].trainable = False
model.layers[6].trainable = False
model.layers[7].trainable = False


# In[4]:


model.summary()


# In[5]:


get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'IMS_Processed')


# In[6]:


df1 = pd.read_csv("IMS_3.csv")


# In[7]:


frames_pro = [df1[:300],df1[-301:-1]]
pro = pd.concat(frames_pro)


# In[8]:


pro.head()


# In[9]:


pro.tail()


# In[10]:


print(pro.shape)


# In[11]:


y = pd.get_dummies(pro.Anomaly, prefix='Anomaly')
print(y.shape)


# In[12]:


print(y)


# In[13]:


df = pro.drop(['Unnamed: 0', 'Anomaly',"Unnamed: 0.1"], axis=1)
df.head()
print(df.shape)


# In[14]:


scaler = StandardScaler()

print(scaler.fit(df))
data = scaler.transform(df)


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
print(len(x_train), len(x_test), len(y_train), len(y_test))


# In[16]:


lr = 1e-3
opt = tf.keras.optimizers.Adam(lr)
metrics = [BinaryAccuracy(), Recall(), Precision()]
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)


# In[17]:


start=time.time()
get_ipython().run_line_magic('memit', 'history = model.fit(x_train, y_train, epochs=150, batch_size=128, validation_split=0.2)')
print(time.time()-start)


# In[18]:


get_ipython().run_line_magic('cd', '../Transfer Learning-Supervised/CWRU-IMS/Conv Layers')

model.save("./TL_CWRU-IMS_Conv_1D.h5")


# In[19]:


sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 12}
labelfont = {'family' : 'Times', 'size'   : 14}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

plt.plot(history.history['loss'], linestyle='-',color='darkorange', linewidth = 2.5, label = 'Training Loss')
plt.plot(history.history['val_loss'], linestyle='-',color='mediumblue', linewidth = 2.5, label = 'Validation Loss')
plt.title('Model Loss')

plt.xticks(**tickfont)
plt.yticks(**tickfont)

plt.grid(b=True, which='major', color='darkgray', linestyle='-',alpha=0.2)

plt.ylabel('Loss')
plt.xlabel('Epoch')

ax.spines['bottom'].set_color('k')
ax.spines['top'].set_color('k') 
ax.spines['right'].set_color('k')
ax.spines['left'].set_color('k')
plt.legend(prop={'size': 14}, loc = 'upper right')
plt.show()


# In[20]:


sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 12}
labelfont = {'family' : 'Times', 'size'   : 14}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

plt.plot(history.history['binary_accuracy'], linestyle='-',color='darkorange', linewidth = 2.5, label = 'Training Accuracy')
plt.plot(history.history['val_binary_accuracy'], linestyle='-',color='mediumblue', linewidth = 2.5, label = 'Validation Loss')
plt.title('Model Accuracy')

plt.xticks(**tickfont)
plt.yticks(**tickfont)

plt.grid(b=True, which='major', color='darkgray', linestyle='-',alpha=0.2)

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

ax.spines['bottom'].set_color('k')
ax.spines['top'].set_color('k') 
ax.spines['right'].set_color('k')
ax.spines['left'].set_color('k')
plt.legend(prop={'size': 14}, loc = 'lower right')
plt.show()


# In[21]:


test_model = tf.keras.models.load_model("./TL_CWRU-IMS_Conv_1D.h5")

predictions = test_model.predict(x_test)


# In[22]:


pred = np.argmax(predictions, axis=1)
actual = np.argmax(np.array(y_test), axis=1)


# In[23]:


print(actual.shape)


# In[24]:


conf_matrix = confusion_matrix(actual, pred)


# In[25]:


print(conf_matrix)


# In[26]:


sns.set(style="whitegrid") # make background white colored

tickfont = {'family' : 'Times', 'size'   : 14}
labelfont = {'family' : 'Times', 'size'   : 16}

fig = plt.figure(figsize=(8,5), dpi=200)
ax = fig.add_subplot(111)

df_cm = pd.DataFrame(conf_matrix, range(2), range(2))

sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='.0f', annot_kws={"size":16})

plt.show()


# In[27]:


accuracy_score(actual, pred)


# In[ ]:




