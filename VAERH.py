import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,recall_score,precision_score,f1_score,accuracy_score,precision_recall_curve, auc
df = pd.read_csv('Cdataset\Cdataset.csv', index_col=0)
x_train = np.array(df)

dfx = pd.read_csv('VaeResultdrug.csv',index_col=0)
dfx = np.array(dfx)
dfy = pd.read_csv('VaeResultdisease.csv',index_col=0)
dfy = np.array(dfy).T
print(dfx.shape,dfy.shape)
pred_x_train = (dfx + dfy)/2

# print('roc_auc_score:%f'%roc_auc_score(x_train.flatten(), dfy.flatten()))
# dfy = np.int64(dfy>=0.5)
# print('recall_score:%f'%recall_score(x_train.flatten(), dfy.flatten()))
# print('precision_score:%f'%precision_score(x_train.flatten(), dfy.flatten()))
# print('f1_score:%f'%f1_score(x_train.flatten(), dfy.flatten()))
# print('accuracy_score:%f'%accuracy_score(x_train.flatten(), dfy.flatten()))
print('roc_auc_score:%f'%roc_auc_score(x_train.flatten(), pred_x_train.flatten()))
# pred_scores = np.int64((dfx>=0.5)|(dfy>=0.5))
# pred_scores = np.int64(pred_x_train>=0.5)
precision, recall, _thresholds = precision_recall_curve(x_train.flatten(), pred_x_train.flatten())
aupr = auc(recall, precision)
print(aupr)

################Temporary notes needed
# pred_scores = np.int64((dfx>=0.5)|(dfy>=0.5))
# print('recall_score:%f'%recall_score(x_train.flatten(), pred_scores.flatten()))
# print('precision_score:%f'%precision_score(x_train.flatten(), pred_scores.flatten()))
# print('f1_score:%f'%f1_score(x_train.flatten(), pred_scores.flatten()))
# print('accuracy_score:%f'%accuracy_score(x_train.flatten(), pred_scores.flatten()))