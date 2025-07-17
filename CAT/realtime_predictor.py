def get_latest_engine_state(log_file='seatbelt_simulation_log.csv'):
    try:
        df = pd.read_csv(log_file)
        if len(df) == 0:
            return None
        latest = df.iloc[-1]
        # engine_on: True (on), False (off), None (idle)
        return latest['engine_on']
    except Exception as e:
        print(f"[WARN] Could not read engine state: {e}")
        return None
import numpy as np
import pandas as pd
import time
import joblib
from tensorflow.keras.models import load_model
import random
from datetime import datetime
import csv
import os

# Load model and scaler
MODEL_PATH = r"lstm_forecast_model.h5"
SCALER_PATH = "scaler.save"
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# Define the columns (should match training)
sensor_cols = [
    'Engine_Temperature', 'Fuel_Level', 'Fuel_Pressure', 'Water_in_Fuel',
    'Engine_Oil_Pressure', 'Engine_RPM', 'Hydraulic_Oil_Temp', 'Hydraulic_Pressure',
    'Transmission_Oil_Temp', 'Brake_Pressure', 'Coolant_Temperature',
    'Air_Filter_Pressure_Drop', 'Battery_Voltage', 'PTO_RPM',
    'Engine_Load', 'Exhaust_Temperature',
    'Transmission_Gear_State', 'Machine_Operating_Mode'
]

# Load the exact columns used during training
import numpy as np
all_cols = np.load('model_columns.npy', allow_pickle=True).tolist()

# Simulate a single sensor row (random realistic values)
def generate_sensor_row():
    return {
        'Engine_Temperature': random.uniform(70, 110),
        'Fuel_Level': random.uniform(0, 100),
        'Fuel_Pressure': random.uniform(20, 70),
        'Water_in_Fuel': random.uniform(0, 10),
        'Engine_Oil_Pressure': random.uniform(15, 80),
        'Engine_RPM': random.uniform(300, 3500),
        'Hydraulic_Oil_Temp': random.uniform(50, 120),
        'Hydraulic_Pressure': random.uniform(10, 100),
        'Transmission_Oil_Temp': random.uniform(60, 140),
        'Brake_Pressure': random.uniform(0, 100),
        'Coolant_Temperature': random.uniform(70, 115),
        'Air_Filter_Pressure_Drop': random.uniform(0, 5),
        'Battery_Voltage': random.uniform(10.5, 15.5),
        'PTO_RPM': random.uniform(0, 1000),
        'Engine_Load': random.uniform(0, 100),
        'Exhaust_Temperature': random.uniform(100, 250),
        'Transmission_Gear_State': random.randint(1, 6),
        'Machine_Operating_Mode': random.randint(0, 1)
    }

# Anomaly classifier (simple rules, can be extended)
def classify_anomaly(pred):
    # pred is denormalized, in same order as all_cols
    pred_dict = dict(zip(all_cols, pred))
    if pred_dict.get('Engine_Temperature', 0) > 105:
        return "Engine Overheating"
    if pred_dict.get('Fuel_Level', 100) < 10:
        return "Low Fuel"
    if pred_dict.get('Engine_RPM', 0) > 3200:
        return "RPM Surge"
    if pred_dict.get('Brake_Pressure', 100) < 10:
        return "Brake Failure Risk"
    # Add more rules as needed
    return None

# Calculate anomaly percentage based on MSE (relative to a threshold)

# Adaptive anomaly percentage scaling based on rolling max MSE
rolling_max_mse = [0.2]  # Start with default
def anomaly_percentage(mse, threshold=0.03, max_mse=None):
    # Use rolling max MSE for scaling if not provided
    if max_mse is None:
        max_mse = max(rolling_max_mse)
    # Update rolling max
    if mse > rolling_max_mse[-1]:
        rolling_max_mse.append(mse)
        if len(rolling_max_mse) > 100:
            rolling_max_mse.pop(0)
    percent = min(100, max(0, (mse / max(threshold, max_mse)) * 100))
    return percent

# Real-time simulation and prediction


def run_realtime_simulation(window_size=30, interval=1, avg_window=15):
    buffer = []
    anomaly_percents = []
    print("Starting real-time simulation and anomaly prediction...\n")
    tick = 0
    while True:
        engine_state = get_latest_engine_state()
        if engine_state in [True, None]:  # True = on, None = idle
            # Simulate new row
            row = generate_sensor_row()
            row_df = pd.DataFrame([row])
            row_encoded = pd.get_dummies(row_df, columns=['Transmission_Gear_State', 'Machine_Operating_Mode'])
            # Ensure all columns present
            for col in all_cols:
                if col not in row_encoded:
                    row_encoded[col] = 0
            row_encoded = row_encoded[all_cols]
            buffer.append(row_encoded.values[0])
            if len(buffer) > window_size:
                buffer.pop(0)
            print(f"Simulated row at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            # Only predict if buffer is full
            if len(buffer) == window_size:
                input_seq = np.array(buffer, dtype=np.float32).reshape(1, window_size, -1)
                pred_scaled = model.predict(input_seq)
                pred = scaler.inverse_transform(pred_scaled)[0]
                # Calculate anomaly percentage using MSE between prediction and last input
                last_input = scaler.inverse_transform(np.array([buffer[-1]]))[0]
                mse = np.mean((last_input - pred) ** 2)
                percent = anomaly_percentage(mse)
                anomaly_percents.append(percent)
                if len(anomaly_percents) > avg_window:
                    anomaly_percents.pop(0)
                print(f"Anomaly likelihood: {percent:.1f}% (MSE: {mse:.5f}, max_MSE_scale: {max(rolling_max_mse):.5f})")
                if percent >= 80:
                    print(f"\n🚨 HIGH WARNING: Anomaly likelihood is very high!\n")
                if classify_anomaly(pred):
                    print(f"\n🚨 WARNING: {classify_anomaly(pred)} predicted in the near future!\n")
                else:
                    print(f"No anomaly predicted in the next step. (Likelihood: {percent:.1f}%)")
                tick += 1
                if tick % avg_window == 0:
                    avg_percent = sum(anomaly_percents) / len(anomaly_percents)
                    print(f"\n=== Average anomaly likelihood over last {avg_window} seconds: {avg_percent:.1f}% ===\n")
        else:
            print(f"Engine is OFF at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Prediction paused.")
        time.sleep(interval)

if __name__ == "__main__":
    run_realtime_simulation()
