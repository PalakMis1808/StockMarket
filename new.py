import pandas_datareader as pdr
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
import math

# Set your API key for Tiingo
key = "d09c7902020a6a4f6f7e6b55ab196ce6da18dd11"
url = f"https://api.tiingo.com/tiingo/daily/AAPL/prices?token={key}"
response = requests.get(url)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data)
print("Initial Data:")
print(df.head())

# Using yfinance to fetch stock data for 'AAPL'
df = yf.download('AAPL')
print("Downloaded Data:")
print(df.head())

# Resetting the DataFrame and selecting 'Close' prices for processing
df1 = pd.read_csv('AAPL.csv')
df1 = df1.reset_index()['close']

# Plot the close price data
plt.plot(df1)
plt.title('AAPL Close Price')
plt.show()

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

# Split dataset into training and test sets
training_size = int(len(df1) * 0.65)
train_data, test_data = df1[:training_size], df1[training_size:]

# Function to create dataset matrix for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Prepare train and test data for LSTM
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] as required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Make predictions and inverse transform to get actual values
train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

# Calculate and print RMSE
train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Prepare data for plotting predictions
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict) + time_step, :] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (time_step * 2) + 1:len(df1) - 1, :] = test_predict

# Plot actual, train predictions, and test predictions
plt.plot(scaler.inverse_transform(df1), label="Actual Prices")
plt.plot(trainPredictPlot, label="Train Predictions")
plt.plot(testPredictPlot, label="Test Predictions")
plt.title("Stock Price Predictions")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()

# Forecast for the next 30 days
x_input = test_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []

for i in range(30):
    if len(temp_input) > time_step:
        x_input = np.array(temp_input[1:])
    else:
        x_input = np.array(temp_input)
    x_input = x_input.reshape((1, time_step, 1))
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    temp_input = temp_input[1:]
    lst_output.append(yhat[0][0])

# Plot future predictions
day_new = np.arange(1, time_step + 1)
day_pred = np.arange(time_step + 1, time_step + 31)

plt.plot(day_new, scaler.inverse_transform(df1[-time_step:]), label="Recent Prices")
plt.plot(day_pred, scaler.inverse_transform(np.array(lst_output).reshape(-1, 1)), label="Forecast")
plt.title("Future 30-Day Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()

# Create DataFrame for predictions to detect anomalies
predictions_df = pd.DataFrame(np.concatenate((train_predict, test_predict), axis=0), columns=['Predicted'])

# Calculate Z-scores for anomaly detection
predictions_df['Z_score'] = (predictions_df['Predicted'] - predictions_df['Predicted'].mean()) / predictions_df['Predicted'].std()
predictions_df['Anomaly'] = predictions_df['Z_score'].abs() > 2  # Threshold of 2 for anomalies

# Plotting predictions with detected anomalies
plt.figure(figsize=(14, 7))
plt.plot(predictions_df['Predicted'], label='Predicted Prices', color='blue', linewidth=2)

# Highlight anomalies
anomalies = predictions_df[predictions_df['Anomaly']]
if not anomalies.empty:
    plt.scatter(anomalies.index, anomalies['Predicted'], color='red', label='Anomalies', marker='o')

# Add mean and threshold lines
mean_price = predictions_df['Predicted'].mean()
plt.axhline(mean_price, color='green', linestyle='--', label='Mean Price')
std_dev = predictions_df['Predicted'].std()
plt.axhline(mean_price + 2 * std_dev, color='orange', linestyle='--', label='Upper Threshold')
plt.axhline(mean_price - 2 * std_dev, color='orange', linestyle='--', label='Lower Threshold')

plt.title('Stock Price Predictions with Detected Anomalies')
plt.xlabel('Time Steps')
plt.ylabel('Predicted Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display detected anomalies
print("Detected Anomalies:")
print(anomalies[['Predicted', 'Z_score']])
