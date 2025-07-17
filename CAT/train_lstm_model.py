import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Load your dataset
DATA_PATH = r"synthetic_machinery_dataset_with_gradual_anomalies_123456789.csv"
df = pd.read_csv(DATA_PATH)

# Define sensor columns
sensor_cols = [
    'Engine_Temperature', 'Fuel_Level', 'Fuel_Pressure', 'Water_in_Fuel',
    'Engine_Oil_Pressure', 'Engine_RPM', 'Hydraulic_Oil_Temp', 'Hydraulic_Pressure',
    'Transmission_Oil_Temp', 'Brake_Pressure', 'Coolant_Temperature',
    'Air_Filter_Pressure_Drop', 'Battery_Voltage', 'PTO_RPM',
    'Engine_Load', 'Exhaust_Temperature',
    'Transmission_Gear_State', 'Machine_Operating_Mode'
]


# One-hot encode categorical values
categorical_cols = ['Transmission_Gear_State', 'Machine_Operating_Mode']
df_encoded = pd.get_dummies(df[sensor_cols], columns=categorical_cols)

# Save the columns used for one-hot encoding
import numpy as np
np.save('model_columns.npy', df_encoded.columns.values, allow_pickle=True)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_encoded)

# Save the scaler
joblib.dump(scaler, "scaler.save")

# Create time sequences
sequence_length = 30  # past 30 mins
prediction_offset = 5  # predict 5 mins later

X, y = [], []
for i in range(len(scaled_data) - sequence_length - prediction_offset):
    X.append(scaled_data[i:i+sequence_length])
    y.append(scaled_data[i+sequence_length+prediction_offset-1])  # predict ahead

X, y = np.array(X), np.array(y)

print("Input Shape:", X.shape)
print("Target Shape:", y.shape)

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(y.shape[1])  # Predict all features
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(X, y, epochs=15, batch_size=32, validation_split=0.1, verbose=1)

# Save model for later real-time inference
model.save("lstm_forecast_model.h5")

print("Training complete. Model and scaler saved.")
