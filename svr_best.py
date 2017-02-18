import numpy as np
import pandas as pandas
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

my_cols = ["Temperature", "Vent", "Humidite", "TemperatureRosee",
            "Pression", "Description", "Heure de la journee", "Jour de la semaine", "Jour ferie"]
dataframe = pandas.read_csv('dataToLearn.csv', usecols=my_cols,  delimiter=";", engine='python', skipfooter=3)
X = dataframe.values
print(type(X))
cible = pandas.read_csv('dataToLearn.csv', usecols=[20],  delimiter=";", engine='python', skipfooter=3)
y = cible.values


X = preprocessing.scale(X)
y = y.ravel()

# Division en deux parties pour l apprentissage et pour le test
train_size = int(len(cible) * 0.67)
test_size = len(cible) - train_size

train, test = X[0:train_size,:], X[train_size:len(dataset),:]
y_train, y_test = X[0:train_size,:], X[train_size:len(dataset),:]

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(train, y_train).predict(test)
y_lin = svr_lin.fit(train, y_train).predict(test)
y_poly = svr_poly.fit(train, y_train).predict(test)

lw = 2
plt.plot(y, color='darkorange', label='data')
plt.hold('on')
plt.plot(y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(y_lin, color='c', lw=lw, label='Linear model')
plt.plot(y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
print(mean_squared_error(y, y_rbf))
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()