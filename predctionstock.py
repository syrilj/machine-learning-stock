from black import out
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.layers.core import Dense,Activation,Dropout
from keras.models import Sequential
import time


# Load Data
company = 'MSFT'

start = dt.datetime(2009,1, 1)
end = dt.datetime(2021,1,1)

data = web.DataReader(company, 'yahoo', start, end)



# Prep data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Adj Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Model
model = Sequential()

model.add(LSTM(units=256, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=64)

'''Test the model accuracy on exsiting data'''
# load test data

test_start = dt.datetime(2021,1,1)
test_end = dt.datetime(2022,1,2)

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_prices = test_data['Adj Close'].values

total_dataset = pd.concat((data['Adj Close'], test_data['Adj Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# make predictions

x_test = []

for x in range(prediction_days, len(model_inputs+1)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# plot
plt.plot(actual_prices, color='black', label=f'Actual {company} Price')
plt.plot(prediction_prices, color='red', label=f'Predictied {company} Price')
plt.title(f'{company} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()

# predciting future day
X_train, y_train, X_test, y_test = LSTM.load_data(total_dataset,50,True )
model = Sequential()
model.add(LSTM(
    input_dim =1,
    output_dim =20,
    return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim = 1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('Compilation time: ') , time.time()-start 
model.fit(
    X_train, 
    y_train,
    batch_size=512,
    nb_enoch=1,
    validation=0.05
)
# Predicting the next day's price.
predictions = LSTM.predict_sequential(model, x_test, 50, 50)
LSTM.plot_results_multiple(predictions, y_test, 50)


#additional code to help use the predcition
#real_data = (model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0])
#real_data = np.array(real_data)
#real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

#print(scaler.inverse_transform(real_data[-1]))
#prediction = model.predict(real_data)
#prediction = scaler.inverse_transform(prediction)
