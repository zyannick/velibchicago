import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

my_cols = ["Temperature", "Vent", "Humidite", "TemperatureRosee",
            "Pression", "Description", "Heure de la journee", "Jour de la semaine", "Jour ferie"]
dataframe = pandas.read_csv('dataToLearn.csv', usecols=my_cols,  delimiter=";", engine='python', skipfooter=3)
X = dataframe.values
print(type(X))
cible = pandas.read_csv('dataToLearn.csv', usecols=[20],  delimiter=";", engine='python', skipfooter=3)
y = cible.values

# Division en deux parties pour l apprentissage et pour le test
train_size = int(len(cible) * 0.67)
test_size = len(cible) - train_size

train, test = X[0:train_size,:], X[train_size:len(dataset),:]
y_train, y_test = X[0:train_size,:], X[train_size:len(dataset),:]


regressor = RandomForestRegressor(n_estimators=150, min_samples_split=2)
regressor.fit(train, y_train.ravel())
print(mean_squared_error(y_test, regressor.predict(test)))

plt.plot(cible, color='darkorange', label='data')
plt.hold('on')
plt.plot(regressor.predict(train), label='La prediction')
plt.plot(regressor.predict(test), label='Le test')
plt.xlabel('temps')
plt.ylabel('disponibilite')
plt.title('Random Forest')
plt.legend()
plt.show()