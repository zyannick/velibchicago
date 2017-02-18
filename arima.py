from pandas import read_csv
from pandas import datetime
import pandas as pandas
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def parser(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S %p')


my_cols = ["Artesian" ]
series = read_csv('Artesian.csv', usecols=my_cols,  delimiter=";", squeeze=True)
series.reset_index()
series.astype(float)
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test, color='blue', label='data')
pyplot.hold('on')
pyplot.plot(predictions, color='red', label='La prediction')
pyplot.xlabel('temps')
pyplot.ylabel('disponibilite')
pyplot.title('Arima')
pyplot.legend()
pyplot.show()



"""
from pandas import read_csv
from pandas import datetime
import pandas as pandas
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot

def parser(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S %p')


my_cols = ["Artesian" ]
series = read_csv('Artesian.csv', usecols=my_cols,  delimiter=";", squeeze=True)
series.reset_index()
series.astype(float)
autocorrelation_plot(series)
pyplot.show()

"""


"""
from pandas import read_csv
from pandas import datetime
import pandas as pandas
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot

def parser(x):
    return datetime.strptime(x, '%d/%m/%Y %H:%M:%S %p')


my_cols = ["Artesian" ]
series = read_csv('Artesian.csv', usecols=my_cols,  delimiter=";", squeeze=True)
series.reset_index()
series.astype(float)
print(series.head())
series.plot()
pyplot.show()
"""