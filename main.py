# Tested with Python 3.9.6
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, roc_curve

data = pd.read_csv("data/TrainingDataSet_Wheat.csv", sep=';')

print(data.head())

# On ne prend pas en compte les colonnes vides, ou n'existant pas dans le jeu à prédire
# Certaines colonnes ne sont donc pas listées ici
X = data[["ETP_1", "ETP_10", "ETP_11", "ETP_12", "ETP_2", "ETP_3", "ETP_4", "ETP_5", "ETP_6", "ETP_9", "PR_1", "PR_10",
          "PR_11", "PR_12", "PR_2", "PR_3", "PR_4", "PR_5", "PR_6", "PR_9", "RV_1", "RV_10", "RV_11", "RV_12", "RV_2",
          "RV_3", "RV_4", "RV_5", "RV_6", "RV_9", "SeqPR_1", "SeqPR_10", "SeqPR_11", "SeqPR_12", "SeqPR_2", "SeqPR_3",
          "SeqPR_4", "SeqPR_5", "SeqPR_6", "SeqPR_9", "Tn_1", "Tn_10", "Tn_11", "Tn_12", "Tn_2", "Tn_3", "Tn_4", "Tn_5",
          "Tn_6", "Tn_9", "Tn17.2_1", "Tn17.2_11", "Tn17.2_12", "Tn17.2_2", "Tn17.2_3", "Tn17.2_4",
          "Tx_1", "Tx_10", "Tx_11", "Tx_12", "Tx_2", "Tx_3", "Tx_4", "Tx_5", "Tx_6",
          "Tx_9", "Tx010_1", "Tx010_10", "Tx010_11", "Tx010_12", "Tx010_2", "Tx010_3", "Tx010_4", "Tx010_5",
          "Tx010_9", "Tx34_5", "Tx34_6",
          "Tx34_9"]]

y = data["Class"]

# Exemple avec optimisation des hyperparamètres
# ---------------------------------------------

# Define the xgboost model
xgb_model = xgb.XGBRegressor()

# Define the parameters to be tuned
# Cf https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:squarederror'],
              # small values give the most accuracy, but take longer to compute
              # Inutile de mettre trop de valeurs ici, on va toujours préférer une valeur faible, quitte à avoir un peu plus de temps de calcul
              'learning_rate': [.01, 0.03],
              'max_depth': [6, 7], # decision tree max depth
              'min_child_weight': [3, 4, 5],
              'subsample': [0.3, 0.5, 0.7], # Represents the fraction of observations to be sampled for each tree. A lower values prevent overfitting but might lead to under-fitting.
              'colsample_bytree': [0.5, 0.7], # Represents the fraction of columns to be randomly sampled for each tree. It might improve overfitting.
              'n_estimators': [500, 700]}

# StratifiedKFold : recommended cross-validation method, given that our data set is unbalanced (we have 80% of our
# data with class 0, 20% with class 1
cross_validation_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# We use this cross validation method to grid search the best hyper parameters
# Using n_jobs parameter, we can compute on several threads at the same time
grid = GridSearchCV(xgb_model,
                    parameters,
                    cv=cross_validation_skf,
                    scoring='neg_mean_squared_error',
                    n_jobs = 5,
                    verbose=True)

grid.fit(X, y)

# Print the best parameters
print(grid.best_params_)
# {'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 700,
# 'nthread': 4, 'objective': 'reg:squarederror', 'subsample': 0.7}


# Create the model with the best parameters
best_model = xgb.XGBRegressor(**grid.best_params_)

# Fit the model with the best parameters
best_model.fit(X, y)

# Make predictions on the test set
y_pred = best_model.predict(X)

# Print the performance metrics
print("Mean Squared Error: ", mean_squared_error(y, y_pred))
print("R-Squared: ", r2_score(y, y_pred))

# Mean Squared Error:  0.002966281815730187
# R-Squared:  0.9824119076808946
# On a clairement du surajustement, de telles perfs sont irréalistes

auc_lr = roc_auc_score(y, y_pred)
# 1, ce qui est irréaliste

# calcul du critère AUC en validation croisée
# -------------------------------------------

# On va découper notre jeu de données en 5, et l'ajuster sur ses 4/5e, le dernier 5e servant de jeu de test
cross_validation_n_folds = 5
cross_validation_skf = StratifiedKFold(n_splits=cross_validation_n_folds, shuffle=True, random_state=42)

scores = []

# Perform the cross-validation
for train_index, test_index in cross_validation_skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # We use here the optimized model previously configured
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    scores.append(roc_auc_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    print(roc_auc_score(y_test, y_pred))

# Print the mean score and the standard deviation of the auc criteria
print("Mean Score: ", np.mean(scores))
print("Standard Deviation: ", np.std(scores))

# Mean Score:  0.8794496207438846
# Standard Deviation:  0.013632074227507077
# On arrive à un AUC moyen de 0.88 en validation croisée, ce qui est énorme...
# A voir sur le jeu de test : est-ce qu'il y a une erreur dans le code qui amène à ce résultat
# (mauvaise application de la validation croisée ?),
# ou est-ce que xGBoost est réellement aussi puissant ?

# Application au jeu de test, pour la prédiction définitive
# ---------------------------------------------------------

# On commence par ajuster notre modèle sur l'ensemble du jeu d'entraînement
best_model.fit(X, y)

data_test = pd.read_csv("data/TestDataSet_Wheat_blind.csv", sep=';')

print(data_test.head())

# On ne prend pas en compte les colonnes vides, ou n'existant pas dans le jeu à prédire
# Certaines colonnes ne sont donc pas listées ici
X_test = data_test[["ETP_1", "ETP_10", "ETP_11", "ETP_12", "ETP_2", "ETP_3", "ETP_4", "ETP_5", "ETP_6", "ETP_9", "PR_1", "PR_10",
          "PR_11", "PR_12", "PR_2", "PR_3", "PR_4", "PR_5", "PR_6", "PR_9", "RV_1", "RV_10", "RV_11", "RV_12", "RV_2",
          "RV_3", "RV_4", "RV_5", "RV_6", "RV_9", "SeqPR_1", "SeqPR_10", "SeqPR_11", "SeqPR_12", "SeqPR_2", "SeqPR_3",
          "SeqPR_4", "SeqPR_5", "SeqPR_6", "SeqPR_9", "Tn_1", "Tn_10", "Tn_11", "Tn_12", "Tn_2", "Tn_3", "Tn_4", "Tn_5",
          "Tn_6", "Tn_9", "Tn17.2_1", "Tn17.2_11", "Tn17.2_12", "Tn17.2_2", "Tn17.2_3", "Tn17.2_4",
          "Tx_1", "Tx_10", "Tx_11", "Tx_12", "Tx_2", "Tx_3", "Tx_4", "Tx_5", "Tx_6",
          "Tx_9", "Tx010_1", "Tx010_10", "Tx010_11", "Tx010_12", "Tx010_2", "Tx010_3", "Tx010_4", "Tx010_5",
          "Tx010_9", "Tx34_5", "Tx34_6",
          "Tx34_9"]]

y_test_pred = best_model.predict(X_test)
y_test_pred = pd.DataFrame(y_test_pred)
X_test["Class_pred"] = y_test_pred

print(X_test.head())

X_test.to_csv('results_ble.csv', sep=";")
# Etrangement, on a quelques valeurs légèrement négatives
# Pour un résultat optimal, il faudrait les seuiller à zéro

