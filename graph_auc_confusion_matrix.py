# Tested with Python 3.9.6
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import roc_auc_score, roc_curve

# AUC et courbe ROC : https://towardsdatascience.com/roc-and-auc-how-to-evaluate-machine-learning-models-in-no-time-fb2304c83a7f
# ROC(Receiver Operating Characteristic) curves to evaluate different thresholds
# for classification machine learning problems.In a nutshell, ROC curve visualizes a confusion matrix for every threshold.

# Ici, le seuil est 0.5 : si proba d'anomalie de rdt > 0.5, on a dépassé le seuil
# La courbe ROC montre le taux de faux positifs sur l'axe X, et des faux négatifs sur l'axe Y

rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

auc_lr = roc_auc_score(y_test, y_pred)
# 0.89 !! Enorme, à vérifier s'il n'y a pas une erreur de calcul
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred)

plt.figure(figsize=(12, 7))
plt.plot(fpr_lr, tpr_lr, label=f'AUC (Logistic Regression) = {auc_lr:.2f}')
plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')
plt.title('ROC Curve', size=20)
plt.xlabel('False Positive Rate', size=14)
plt.ylabel('True Positive Rate', size=14)
plt.legend();


y_test.describe()
y_pred.describe()
y_pred_df = pd.DataFrame(y_pred)

y_test.to_csv('newFile.csv')
y_pred_df.to_csv('newFile_pred.csv')

y_pred_round = np.where(y_pred > 0.5, 1, 0)
y_pred_df = pd.DataFrame(y_pred_round)
y_pred_df.to_csv('newFile_pred.csv')

y_pred_df['test'] = y_test['Class']
y_pred_df['round'] = y_pred_df['Age'].astype(int)

confusion_matrix = metrics.confusion_matrix(y_test, y_pred_df)
array([[531,  27],
       [ 79,  78]], dtype=int64)
# soit 85.2% de bien classés
# Etrangement, on est à peu près sur les mêmes niveau de bien classés qu'avec le réseau de neurones
# Comment expliquer la différence d'AUC ?

auc_lr = roc_auc_score(y_test, y_pred_df)
# 0.72 : passer en mode binaire dégrade déjà de 10 points l'AUC




