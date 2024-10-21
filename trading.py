
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pickle

  
# Load and prepare data
data = pd.read_csv('./BTCUSDT_Historical_Data.csv')
data = data[['Close', 'Low', 'High', 'Open', 'Volume']].dropna()
features = data.values

# Initialize scalers
scalers = {}
scaled_data = np.zeros_like(features)

# Scale each feature separately
for i, column in enumerate(['Close', 'Low', 'High', 'Open', 'Volume']):
    scalers[column] = MinMaxScaler(feature_range=(0, 1))
    scaled_data[:, i] = scalers[column].fit_transform(features[:, i].reshape(-1, 1)).flatten()

# Create the sliding window dataset with multiple features
def create_multifeature_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 0])  # Predicting 'Close'
    return np.array(X), np.array(Y)

time_step = 200
X, y = create_multifeature_dataset(scaled_data, time_step)

# Adjust input shape to include features
num_features = X.shape[2]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model with adjusted input shape
model = Sequential([
    LSTM(400, return_sequences=True, input_shape=(time_step, num_features)),
    LSTM(400, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # Predicts the 'Close' value
])

model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5, verbose=1, mode='min')

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=300, batch_size=64, verbose=1)

# Evaluate the model
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, train_predict))
print("Test MSE:", mean_squared_error(y_test, test_predict))

# Save the model and scalers
with open('./scalers_400.pkl', 'wb') as f:
    pickle.dump(scalers, f)

model.save('./predict_multifeature_400.h5')


# Predict the next closing price using the last window
last_window = X[-1].reshape(1, time_step, num_features)
next_closing_price = model.predict(last_window)
predicted_price = scalers['Close'].inverse_transform(next_closing_price)
print("Predicted next closing price:", predicted_price[0][0])

