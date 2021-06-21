#! -*- coding: utf-8 -*-

'''Adapted from
   https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,accuracy_score
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from sklearn.decomposition import PCA
from keras.callbacks import LearningRateScheduler
from keras import regularizers
import keras




batch_size = 10
original_dim = 409
latent_dim = 10 # Latent variable dimension
intermediate_dim = 50
epochs = 400

# # Import Data
# datafile = u'******************'
# data = pd.read_excel(datafile)
# data_fea = data.iloc[:, 1:]  #
# data_fea = data_fea.fillna(0)  #
#
# #
# data_mean = data_fea.mean()
# data_std = data_fea.std()
# data_fea = (data_fea - data_mean) / data_std
#
# #
# pca = PCA(n_components=10)
# pca_result = pca.fit_transform(data_fea.values)


def Precision(y_true, y_pred):

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):

    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


#Weighted binary cross entropy loss
def weight_binary_crossentropy(y_true,y_predict):



    return -(y_true * K.log(tf.clip_by_value(y_predict,1e-8,1.0)) + 0.0005*(1 - y_true) * K.log(tf.clip_by_value(1 - y_predict,1e-8,1.0)))

#Binary cross entropy loss
def binary_crossentropy(y_true,y_predict):
    loss = -((1 - y_true) * np.log(1 - y_predict) + y_true * np.log(y_predict))
    return loss


###########
def load_data():
    dfx = pd.read_csv('Cdataset\Cdataset.csv', index_col=0)

    #
    data = pd.read_csv('Cdataset\CdrugSimilarity0.csv', index_col=0)
    #
    data_mean = data.mean()
    data_std = data.std()
    data_fea = (data - data_mean) / data_std
    #
    pca = PCA(n_components=10)
    dfy = pca.fit_transform(data_fea.values)

    all_data = np.c_[np.array(dfx), np.array(dfy)]
    ###print(all_data[:, :409])

    return all_data
###########The sample is divided into test set and training set

def slice1(x):
    return x[:, :original_dim]
def slice2(x):
    return x[:, original_dim:]
def addtensor(args):
    x, y = args
    return x+y


#The structure of the encoder
x = Input(shape=(original_dim+10,))
x1 = Lambda(slice1, output_shape=(original_dim,))(x)
x2 = Lambda(slice2, output_shape=(10,))(x)

h = Dense(intermediate_dim, activation='relu')(x1)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)##########This is the log of variance
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

zadd = Lambda(addtensor, output_shape=(latent_dim,))([z , x2])

# # L1+L2
# decoder_h = Dense(intermediate_dim, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))
# decoder_mean = Dense(original_dim, activation='sigmoid',kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))
# L2
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(zadd)
x_decoded_mean = decoder_mean(h_decoded)

# Modeling
vae = Model(x, x_decoded_mean)


xent_loss = K.sum(weight_binary_crossentropy(x1, x_decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + 0.001*kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()
############################
# x_train,x_test = load_data()
###########################
x_train = load_data()
###############Control the learning rate
def scheduler(epoch):
    #
    if epoch % 150 == 0 and epoch != 0:
        lr = K.get_value(vae.optimizer.lr)
        if epoch == 300:#######Fine-tuning
            K.set_value(vae.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        else:
            K.set_value(vae.optimizer.lr, lr * 0.5)
            print("lr changed to {}".format(lr * 0.5))
    return K.get_value(vae.optimizer.lr)
reduce_lr = LearningRateScheduler(scheduler)
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_train, None),
        callbacks=[reduce_lr])
pred_x_train = vae.predict(x_train)

data = pd.DataFrame(pred_x_train)
data.to_csv('VaeResultdrugall.csv')
##############################################################













