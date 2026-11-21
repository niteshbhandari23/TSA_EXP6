# Ex.No: 6               HOLT WINTERS METHOD
### DATASET: YAHOO_STOCK

### Date: 21/11/2026
### NAME : NITESH BHANDARI K
### REGISTER NUMBER : 212225240101



### AIM:
To implement and evaluate the Holt-Winters Exponential Smoothing model in Python on the yahoo_stock dataset using Date as the time index and Volume as the target variable, including monthly resampling, time series decomposition, train-test evaluation, and future volume forecasting.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('yahoo_stock.csv', parse_dates=['Date'], index_col='Date')[['Volume']]
print(data.head())

data_monthly = data.resample('MS').sum()
print(data_monthly.head())

ax = data_monthly['Volume'].plot(figsize=(10, 4), title='Monthly Stock Volume')
ax.set_xlabel('Date')
ax.set_ylabel('Volume')
plt.tight_layout()
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly[['Volume']]).flatten(),
    index=data_monthly.index
)

ax_scaled = scaled_data.plot(figsize=(10, 4), title='Scaled Monthly Stock Volume')
ax_scaled.set_xlabel('Date')
ax_scaled.set_ylabel('Scaled Volume')
plt.tight_layout()
plt.show()

decomposition = seasonal_decompose(data_monthly['Volume'], model='additive', period=12)
decomposition.plot()
plt.tight_layout()
plt.show()

scaled_data = scaled_data + 1
split_idx = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split_idx]
test_data = scaled_data[split_idx:]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax_eval = train_data.plot(figsize=(10, 5), label='train_data')
test_predictions_add.plot(ax=ax_eval, label='test_predictions_add')
test_data.plot(ax=ax_eval, label='test_data')
ax_eval.legend(['train_data', 'test_predictions_add', 'test_data'])
ax_eval.set_xlabel('Date')
ax_eval.set_ylabel('Scaled Volume (+1)')
ax_eval.set_title('Visual Evaluation')
plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
mae = mean_absolute_error(test_data, test_predictions_add)
print('RMSE:', rmse)
print('MAE:', mae)
print('Std and Mean:', scaled_data.std(), scaled_data.mean())

final_model = ExponentialSmoothing(
    data_monthly['Volume'],
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

final_predictions = final_model.forecast(steps=int(len(data_monthly) / 4))

ax_final = data_monthly['Volume'].plot(figsize=(10, 5), label='data_monthly')
final_predictions.plot(ax=ax_final, label='final_predictions')
ax_final.legend(['data_monthly', 'final_predictions'])
ax_final.set_xlabel('Date')
ax_final.set_ylabel('Volume')
ax_final.set_title('Prediction')
plt.tight_layout()
plt.show()
```
### OUTPUT:


<img width="361" height="256" alt="image" src="https://github.com/user-attachments/assets/ce5f8ab3-d995-46b2-8d49-4977ae3b3b2d" />


<img width="1020" height="376" alt="image" src="https://github.com/user-attachments/assets/9136a265-d7af-436e-a13d-4adeb4ec598c" />


<img width="1048" height="390" alt="image" src="https://github.com/user-attachments/assets/e0e16cb2-62b0-4fcc-80a2-913b21a7ecee" />


<img width="754" height="470" alt="image" src="https://github.com/user-attachments/assets/1fc892f0-0c3d-4fa0-8137-1934cf921774" />

<img width="478" height="60" alt="image" src="https://github.com/user-attachments/assets/214c9921-0502-4ec1-a5fc-02a02a8d97c5" />

TEST_PREDICTION

<img width="1020" height="484" alt="image" src="https://github.com/user-attachments/assets/8db68786-3a9a-40c8-a60a-d7dcb125b026" />



FINAL_PREDICTION

<img width="1009" height="476" alt="image" src="https://github.com/user-attachments/assets/4039b07f-9b9f-4a26-90d0-9795ff688078" />






### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
