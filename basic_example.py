# Tested with Python 3.9.6
import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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

# Define the xgboost model
xgb_model = xgb.XGBRegressor()

# Exemple basique sans optimisation ni validation croisée
# ----------------------------------

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert the data into a DMatrix format
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# Define the xgboost model
params = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "alpha": 0.01
}

# Train the model
model = xgb.train(params=params, dtrain=dtrain)

# Make predictions on the test set
y_pred = model.predict(dtest)

# Print the predictions
print(y_pred)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R-Squared: ", r2_score(y_test, y_pred))
# Mean Squared Error:  0.16096129783031105
# R-Squared:  0.06071000293015605
