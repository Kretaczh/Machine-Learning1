# -*- coding: utf-8 -*-
"""
Projekt na podstawie danych: https://www.kaggle.com/datasets/yanmaksi/big-mac-index-dataset-by-contry
"""

#Ładuje biblioteki
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Ładuję dane
df=pd.read_csv('/content/drive/MyDrive/ML Klejment/projekt/BigMac_Index_by_Contry.csv')
df2=pd.read_csv('/content/drive/MyDrive/ML Klejment/projekt/Inflation_forecast_all_countries.csv')

df.head()

#rozdzielam date

df.date = pd.to_datetime(df.date)
df["Quarter"] = df.date.dt.quarter

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df

#konwersja daty do kwartałów

df['time'] = df['year'].astype(str) + '-' + 'Q'+ df['Quarter'].astype(str)

df

df.isnull().sum()

#sprawdzam df
df.head()
df2.head()
df2.isnull().sum()

#zamieniam dane na kategorie

for name in ['region']:
  df[name] = pd.Categorical(pd.factorize(df[name])[0])

for name in ['location']:
  df2[name] = pd.Categorical(pd.factorize(df2[name])[0])

#łącze obydwa df'y

df1=df.merge(df2, left_on='time', right_on='time')

df1['time'] = df1['time'].str.replace('-Q','.')

df1.head()

#Sprawdzam missing data

df1=df1.dropna()
df1.isnull().sum()

#odrzucam pustam kolumny
#X=df1.drop(['dollar_adj_valuation','euro_adj_valuation', 'sterling_adj_valuation','yen_adj_valuation','yuan_adj_valuation','date','Unnamed: 0_y','value','location','Unnamed: 0_x','region','Quarter','day','month','year','time'],axis=1)
X=df1.drop(['dollar_ppp','sterling_adj_valuation','dollar_adj_valuation','yuan_adj_valuation','local_price','yen_adj_valuation','date','Unnamed: 0_y','value','location','Unnamed: 0_x','region','Quarter','day','month','year','time'],axis=1)
#Sprawdzam czy dobrze odrzuciłem
X.isnull().sum()

#y=df2
y=df1['value']
#df['region']
X

#y=inflancja w danym kwartale w dnym kraju
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#wprowadzam algorytmy

#Random Forest
modelRF=RandomForestRegressor()
modelRF.fit(X_train,y_train)
predRF=modelRF.predict(X_test)
#Support Vector Machine
modelSVM=SVR(kernel = 'rbf')
modelSVM.fit(X_train,y_train)
predSVM=modelSVM.predict(X_test)
#K-nearest neighbors
modelKNN=KNeighborsRegressor()
modelKNN.fit(X_train,y_train)
predKNN=modelKNN.predict(X_test)
#DecisionTree
modelDTR=DecisionTreeRegressor()
modelDTR.fit(X_train,y_train)
predDTR=modelDTR.predict(X_test)
#Multilayer perceptron
modelMLP=MLPRegressor()
modelMLP.fit(X_train,y_train)
predMLP=modelMLP.predict(X_test)
#BayesianRidge
modelBR=BayesianRidge(compute_score=True)
modelBR.fit(X_train,y_train)
predBR=modelBR.predict(X_test)
#Lasso
modelLasso=linear_model.Lasso(alpha=0.1)
modelLasso.fit(X_train,y_train)
predLasso=modelLasso.predict(X_test)
#Multiple Linear Regression MLR
modelMLR=LinearRegression()
modelMLR.fit(X_train,y_train)
predMLR=modelMLR.predict(X_test)

#Random Forest RF
modelRF_r2=r2_score(y_test,predRF)
#Support Vector Machine SVM
modelSVM_r2=r2_score(y_test,predSVM)
#K-nearest neighbors KNN
modelKNN_r2=r2_score(y_test,predKNN)
#DecisionTree DTR
modelDTR_r2=r2_score(y_test,predDTR)
#Multilayer perceptron MLP
modelMLP_r2=r2_score(y_test,predMLP)
#BayesianRidge BR
modelBR_r2=r2_score(y_test,predBR)
#Lasso
modelLasso_r2=r2_score(y_test,predLasso)
#MLR
modelMLR_r2=r2_score(y_test,predMLR)
print("The R2 score RF: ", round(modelRF_r2,4))
print("The R2 score SVM: ", round(modelSVM_r2,4))
print("The R2 score KNN: ", round(modelKNN_r2,4))
print("The R2 score DTR: ", round(modelDTR_r2,4))
print("The R2 score MLP: ", round(modelMLP_r2,4))
print("The R2 score BR: ", round(modelBR_r2,4))
print("The R2 score Lasso: ", round(modelLasso_r2,4))
print("The R2 score MLR: ", round(modelMLR_r2,4))

#Random Forest RF
modelRF_mse=mean_squared_error(y_test,predRF)
#Support Vector Machine SVM
modelSVM_mse=mean_squared_error(y_test,predSVM)
#K-nearest neighbors KNN
modelKNN_mse=mean_squared_error(y_test,predKNN)
#DecisionTree DTR
modelDTR_mse=mean_squared_error(y_test,predDTR)
#Multilayer perceptron MLP
modelMLP_mse=mean_squared_error(y_test,predMLP)
#BayesianRidge BR
modelBR_mse=mean_squared_error(y_test,predBR)
#Lasso
modelLasso_mse=mean_squared_error(y_test,predLasso)
#MLR
modelMLR_mse=mean_squared_error(y_test,predMLR)
print("The MSE score RF: ", round(modelRF_mse,4))
print("The MSE score SVM: ", round(modelSVM_mse,4))
print("The MSE score KNN: ", round(modelKNN_mse,4))
print("The MSE score DTR: ", round(modelDTR_mse,4))
print("The MSE score MLP: ", round(modelMLP_mse,4))
print("The MSE score BR: ", round(modelBR_mse,4))
print("The MSE score Lasso: ", round(modelLasso_mse,4))
print("The MSE score MLR: ", round(modelMLR_mse,4))

#Random Forest RF
modelRF_mae=mean_absolute_error(y_test,predRF)
#Support Vector Machine SVM
modelSVM_mae=mean_absolute_error(y_test,predSVM)
#K-nearest neighbors KNN
modelKNN_mae=mean_absolute_error(y_test,predKNN)
#DecisionTree DTR
modelDTR_mae=mean_absolute_error(y_test,predDTR)
#Multilayer perceptron MLP
modelMLP_mae=mean_absolute_error(y_test,predMLP)
#BayesianRidge BR
modelBR_mae=mean_absolute_error(y_test,predBR)
#Lasso
modelLasso_mae=mean_absolute_error(y_test,predLasso)
#MLR
modelMLR_mae=mean_absolute_error(y_test,predMLR)
print("The MAE score RF: ", round(modelRF_mae,4))
print("The MAE score SVM: ", round(modelSVM_mae,4))
print("The MAE score KNN: ", round(modelKNN_mae,4))
print("The MAE score DTR: ", round(modelDTR_mae,4))
print("The MAE score MLP: ", round(modelMLP_mae,4))
print("The MAE score BR: ", round(modelBR_mae,4))
print("The MAE score Lasso: ", round(modelLasso_mae,4))
print("The MAE score MLR: ", round(modelMLR_mae,4))

#Random Forest RF
modelRF_rmse=math.sqrt(modelRF_mse)
#Support Vector Machine SVM
modelSVM_rmse=math.sqrt(modelSVM_mse)
#K-nearest neighbors KNN
modelKNN_rmse=math.sqrt(modelKNN_mse)
#DecisionTree DTR
modelDTR_rmse=math.sqrt(modelDTR_mse)
#Multilayer perceptron MLP
modelMLP_rmse=math.sqrt(modelMLP_mse)
#BayesianRidge BR
modelBR_rmse=math.sqrt(modelBR_mse)
#Lasso
modelLasso_rmse=math.sqrt(modelLasso_mse)
#MLR
modelMLR_rmse=math.sqrt(modelMLR_mse)
print("The RMSE score RF: ", round(modelRF_rmse,4))
print("The RMSE score SVM: ", round(modelSVM_rmse,4))
print("The RMSE score KNN: ", round(modelKNN_rmse,4))
print("The RMSE score DTR: ", round(modelDTR_rmse,4))
print("The RMSE score MLP: ", round(modelMLP_rmse,4))
print("The RMSE score BR: ", round(modelBR_rmse,4))
print("The RMSE score Lasso: ",round(modelLasso_rmse,4))
print("The RMSE score MLR: ", round(modelMLR_rmse,4))

#Random Forest RF
plt.plot([0,max(y_test)],[0,max(predRF)],'b--')
plt.scatter(y_test,predRF,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("RF prediction",fontsize=14)
plt.show()
#Support Vector Machine SVM
plt.plot([0,max(y_test)],[0,max(predSVM)],'b--')
plt.scatter(y_test,predSVM,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("SVM prediction",fontsize=14)
plt.show()
#K-nearest neighbors KNN
plt.plot([0,max(y_test)],[0,max(predKNN)],'b--')
plt.scatter(y_test,predKNN,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("KNN prediction",fontsize=14)
plt.show()
#DecisionTree DTR
plt.plot([0,max(y_test)],[0,max(predDTR)],'b--')
plt.scatter(y_test,predDTR,color='magenta')
plt.xlabel("Real data",fontsize=14)

plt.ylabel("DTR prediction",fontsize=14)
plt.show()
#Multilayer perceptron MLP
plt.plot([0,max(y_test)],[0,max(predMLP)],'b--')
plt.scatter(y_test,predMLP,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("MLP prediction",fontsize=14)
plt.show()
#BayesianRidge BR
plt.plot([0,max(y_test)],[0,max(predBR)],'b--')
plt.scatter(y_test,predBR,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("BR prediction",fontsize=14)
plt.show()
#Lasso
plt.plot([0,max(y_test)],[0,max(predLasso)],'b--')
plt.scatter(y_test,predLasso,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("Lasso prediction",fontsize=14)
plt.show()
#MLR
plt.plot([0,max(y_test)],[0,max(predMLR)],'b--')
plt.scatter(y_test,predMLR,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("MLR prediction",fontsize=14)
plt.show()
importanceRF = modelRF.feature_importances_
for i,v in enumerate(importanceRF):
  print('Feature: %0d, Score: %.5f' % (i,v))

sorted_idx = importanceRF.argsort()
plt.barh(X_train.columns[sorted_idx],

importanceRF[sorted_idx])
plt.xlabel("Random Forest feature importance",fontsize=11)
plt.show()

#najlepsze wyniki pokazuje randomforest więc przeprowadzam hiperparametryzacje

param_gridRF={
'bootstrap': [True],
"n_estimators":[25,50],
"max_depth":[1,10,15,50,100,500],
"min_samples_leaf":[10,50,100],
'min_samples_split': [20,50,100],
#'min_impurity_decrease': [0.1],
'random_state': [420],
'max_features': [2, 3]
}

param_gridRF = {
    'bootstrap': [True],
    'max_depth': [80,  100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300]
}

from sklearn.model_selection import GridSearchCV

grid_searchRF = GridSearchCV(estimator = modelRF,
param_grid = param_gridRF,
cv = 3, n_jobs = -1,
verbose = 2)
grid_searchRF.fit(X_train,y_train)
grid_searchRF.best_params_
grid_searchRF.best_estimator_
resultsRF=grid_searchRF.cv_results_
for mean_score,params in zip(resultsRF["mean_test_score"],resultsRF["params"]):
  print(mean_score,params)
  print(resultsRF["mean_test_score"].max())
  print(resultsRF["mean_test_score"].min())

#wyniki po hiperparametryzacji

plt.plot([0,max(y_test)],[0,max(predRF)],'b--')
plt.scatter(y_test,predRF,color='magenta')
plt.xlabel("Real data",fontsize=14)
plt.ylabel("RF prediction",fontsize=14)
plt.show()

#okazuje się że domyślne parametry dla mojego modelu są najlepsze