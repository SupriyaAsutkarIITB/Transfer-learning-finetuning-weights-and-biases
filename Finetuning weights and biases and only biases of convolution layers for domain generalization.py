#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Finetuning weights and biases of the convolution layers for domain generalization


# In[ ]:


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


# In[ ]:


get_ipython().run_line_magic('cd', '..')
model = tf.keras.models.load_model("CWRU-IMS_1D.h5")
model.summary()


# In[ ]:


model.layers[0].trainable = True
model.layers[1].trainable = True
model.layers[2].trainable = True
model.layers[3].trainable = True
model.layers[4].trainable = True
model.layers[5].trainable = False
model.layers[6].trainable = False
model.layers[7].trainable = False


# In[ ]:


model.summary()


# In[ ]:


get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'IMS_Processed')


# In[ ]:


df1 = pd.read_csv("IMS_3.csv")


# In[ ]:


frames_pro = [df1[:300],df1[-301:-1]]
pro = pd.concat(frames_pro)


# In[ ]:


pro.head()


# In[ ]:


pro.tail()


# In[ ]:


print(pro.shape)


# In[ ]:


y = pd.get_dummies(pro.Anomaly, prefix='Anomaly')
print(y.shape)


# In[ ]:


print(y)


# In[ ]:


df = pro.drop(['Unnamed: 0', 'Anomaly',"Unnamed: 0.1"], axis=1)
df.head()
print(df.shape)


# In[ ]:


scaler = StandardScaler()

print(scaler.fit(df))
data = scaler.transform(df)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
print(len(x_train), len(x_test), len(y_train), len(y_test))


# In[ ]:


lr = 1e-3
opt = tf.keras.optimizers.Adam(lr)
metrics = [BinaryAccuracy(), Recall(), Precision()]
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)start=time.time()
get_ipython().run_line_magic('memit', 'history = model.fit(x_train, y_train, epochs=150, batch_size=128, validation_split=0.2)')
print(time.time()-start)


# In[ ]:


get_ipython().run_line_magic('cd', '../Transfer Learning-Supervised/CWRU-IMS/Conv Layers')

model.save("./TL_CWRU-IMS_Conv_1D.h5")


# In[ ]:


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
plt.show()sns.set(style="whitegrid") # make background white colored 

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


# In[ ]:


test_model = tf.keras.models.load_model("./TL_CWRU-IMS_Conv_1D.h5")

predictions = test_model.predict(x_test)


# In[ ]:


pred = np.argmax(predictions, axis=1)
actual = np.argmax(np.array(y_test), axis=1)


# In[ ]:


print(actual.shape)


# In[ ]:


conf_matrix = confusion_matrix(actual, pred)


# In[ ]:


print(conf_matrix)


# In[ ]:


sns.set(style="whitegrid") # make background white colored

tickfont = {'family' : 'Times', 'size'   : 14}
labelfont = {'family' : 'Times', 'size'   : 16}

fig = plt.figure(figsize=(8,5), dpi=200)
ax = fig.add_subplot(111)

df_cm = pd.DataFrame(conf_matrix, range(2), range(2))

sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='.0f', annot_kws={"size":16})

plt.show()


# In[ ]:


accuracy_score(actual, pred)


# In[ ]:


#Finetuning only biases of the convolution layer for domain generalization


# In[ ]:


get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', '..')
get_ipython().run_line_magic('cd', 'IMS_Processed/')


# In[ ]:


df1 = pd.read_csv("IMS_3.csv")


# In[ ]:


frames_pro = [df1[100:400],df1[-301:-1]]
df = pd.concat(frames_pro)


# In[ ]:


df = df.drop(["Unnamed: 0", "Unnamed: 0.1"],axis=1)


# In[ ]:


y = pd.get_dummies(df.Anomaly, prefix='Anomaly')

df = df.drop(["Anomaly"], axis=1)
df.head()


# In[ ]:


scaler = StandardScaler()
scaler.fit(df)
data = scaler.transform(df)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
print(len(x_train), len(x_test), len(y_train), len(y_test))


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
print(len(x_train), len(x_test), len(y_train), len(y_test))


# In[ ]:


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(128)


# In[ ]:


get_ipython().run_line_magic('cd', '../Bias Training/CWRU-IMS')

model = tf.keras.models.load_model("CWRU-IMS_1D.h5")
model.summary()


# In[ ]:


model.layers[0].trainable = False
model.layers[1].trainable = True
model.layers[2].trainable = True
model.layers[3].trainable = False
model.layers[4].trainable = False
model.layers[5].trainable = False
model.layers[6].trainable = False
model.layers[7].trainable = False


# In[ ]:


model.summary()


# In[ ]:


lr = 1e-1
optimizer = tf.keras.optimizers.Adam(lr)


# In[ ]:


verbose = "Epoch {:2d} Loss: {:.3f} Acc: {:.2%} Val_Loss: {:.3f} Val_Acc: {:.2%} Time Taken: {:.2f}secs"


# In[ ]:


def compute_loss(model, x, y,loss_object, training):
    out = model(x, training=training)
    loss = loss_object(y_true=y, y_pred=out)
    return loss


def get_grad(model, x, y,loss_object,training):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y,loss_object, training)
        to_update = [i for ix, i in enumerate(model.trainable_variables) if ix in (1,3)]
    return loss, tape.gradient(loss, to_update)


# In[ ]:


def fit(epochs, model, train_dataset, loss_object, optimizer):
    loss = []
    acc = []
    val_acc = []
    val_los = []
    for epoch in range(1, epochs+1):
        start = time.time()
        train_loss = tf.metrics.Mean()
        train_acc = tf.metrics.BinaryAccuracy()
        val_acc_metric = tf.metrics.BinaryAccuracy()
        val_loss = tf.metrics.Mean()

        for x, y in train_dataset:
            #print(".\t")
            loss_value, grads = get_grad(model, x, y,loss_object, training=True)
            to_update = [i for ix, i in enumerate(model.trainable_variables) if ix in (1,3)]
            optimizer.apply_gradients(zip(grads, to_update))
            train_loss.update_state(loss_value)
            train_acc.update_state(y, model(x, training=True))
            #print(np.array(train_loss.result()))
        loss.append(np.array(train_loss.result()))
        acc.append(np.array(train_acc.result()))
        
        for x_batch_val, y_batch_val in val_dataset:
            val_loss_value = compute_loss(model, x_batch_val, y_batch_val,loss_object, training=False)
            val_loss.update_state(val_loss_value)
            val_acc_metric.update_state(y_batch_val, model(x_batch_val, training=False))
        
        val_acc.append(np.array(val_acc_metric.result()))
        val_los.append(np.array(val_loss.result()))

        print(verbose.format(epoch,
                             train_loss.result(),
                             train_acc.result(),
                             val_loss.result(),
                             val_acc_metric.result(),
                             time.time()-start))
    return loss, acc, val_los, val_acc


# In[ ]:


loss_object = tf.losses.BinaryCrossentropy()
start=time.time()
get_ipython().run_line_magic('memit', 'loss, acc, val_loss, val_acc = fit(300,model,train_dataset, loss_object, optimizer)')
print(time.time()-start)


# In[ ]:


model.save("IMS_conv.h5")


# In[ ]:


sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 12}
labelfont = {'family' : 'Times', 'size'   : 14}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

plt.plot(loss, linestyle='-',color='darkorange', linewidth = 2.5, label = 'Training Loss')
plt.plot(val_loss, linestyle='-',color='mediumblue', linewidth = 2.5, label = 'Validation Loss')
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


# In[ ]:


sns.set(style="whitegrid") # make background white colored 

tickfont = {'family' : 'Times', 'size'   : 12}
labelfont = {'family' : 'Times', 'size'   : 14}

fig = plt.figure(figsize = (8,5), dpi=200) # set figure size and resolution
ax = fig.add_subplot(111)

plt.plot(acc, linestyle='-',color='darkorange', linewidth = 2.5, label = 'Training Accuracy')
plt.plot(val_acc, linestyle='-',color='mediumblue', linewidth = 2.5, label = 'Validation Accuracy')
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


# In[ ]:


predictions = model.predict(x_test)

pred = np.argmax(predictions, axis=1)
actual = np.argmax(np.array(y_test), axis=1)


# In[ ]:


conf_matrix = confusion_matrix(actual, pred)
print(conf_matrix)


# In[ ]:


accuracy_score(actual, pred)


# In[ ]:


sns.set(style="whitegrid") # make background white colored

tickfont = {'family' : 'Times', 'size'   : 14}
labelfont = {'family' : 'Times', 'size'   : 16}

fig = plt.figure(figsize=(8,5), dpi=200)
ax = fig.add_subplot(111)

df_cm = pd.DataFrame(conf_matrix, range(2), range(2))

sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='.0f', annot_kws={"size":16})

plt.show()


# In[ ]:




