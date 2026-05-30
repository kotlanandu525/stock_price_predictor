import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Load dataset
df = pd.read_csv("ADANIPORTS.csv")

data = df[['Close']]

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
X = []
y = []

for i in range(30, len(scaled_data)):
    X.append(scaled_data[i-30:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Build Model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(30,1)),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

# Train
model.fit(
    X,
    y,
    epochs=20,
    batch_size=32
)

# Save model
model.save("stock_price_rnn.h5")

# Save scaler
joblib.dump(
    scaler,
    "scaler.pkl"
)

print("Model Saved Successfully")