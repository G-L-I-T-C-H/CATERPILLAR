import requests

BACKEND_URL = "http://192.168.1.18:5000/api/moni"  # Replace with your backend's IP and port

def send_alert_to_backend(data):
    try:
        response = requests.post(BACKEND_URL, json=data, timeout=2)
        print("Sent to backend:", response.status_code)
    except Exception as e:
        print("Error sending to backend:", e)

BACKEND_URL2 = "http://192.168.1.18:5000/api/anomaly" 

def send_alert_to_back(data):
    try:
        response = requests.post(BACKEND_URL2, json=data, timeout=2)
        print("Sent to backend:", response.status_code)
    except Exception as e:
        print("Error sending to backend:", e)
        
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
    data = {
        'Engine_Temperature': round(random.uniform(70, 110), 2),
        'Fuel_Level': round(random.uniform(0, 100), 2),
        'Fuel_Pressure': round(random.uniform(20, 70), 2),
        'Water_in_Fuel': round(random.uniform(0, 10), 2),
        'Engine_Oil_Pressure': round(random.uniform(15, 80), 2),
        'Engine_RPM': round(random.uniform(300, 3500), 2),
        'Hydraulic_Oil_Temp': round(random.uniform(50, 120), 2),
        'Hydraulic_Pressure': round(random.uniform(10, 100), 2),
        'Transmission_Oil_Temp': round(random.uniform(60, 140), 2),
        'Brake_Pressure': round(random.uniform(0, 100), 2),
        'Coolant_Temperature': round(random.uniform(70, 115), 2),
        'Air_Filter_Pressure_Drop': round(random.uniform(0, 5), 2),
        'Battery_Voltage': round(random.uniform(10.5, 15.5), 2),
        'PTO_RPM': round(random.uniform(0, 1000), 2),
        'Engine_Load': round(random.uniform(0, 100), 2),
        'Exhaust_Temperature': round(random.uniform(100, 250), 2),
        'Transmission_Gear_State': random.randint(1, 6),
        'Machine_Operating_Mode': random.randint(0, 1)
    }
    send_alert_to_backend(data)
    return data

# Anomaly classifier (simple rules, can be extended)
def classify_anomaly(pred):
    # pred is denormalized, in same order as all_cols
    pred_dict = dict(zip(all_cols, pred))
    if pred_dict.get('Engine_Temperature', 0) > 105:
        #send_alert_to_back({"anomaly": "Engine Overheating", "value": pred_dict.get('Engine_Temperature', 0)})
        return "Engine Overheating"
    if pred_dict.get('Fuel_Level', 100) < 10:
        #send_alert_to_back({"anomaly": "Low Fuel", "value": pred_dict.get('Fuel_Level', 100)})
        return "Low Fuel"
    if pred_dict.get('Engine_RPM', 0) > 3200:
        #send_alert_to_back({"anomaly": "RPM Surge", "value": pred_dict.get('Engine_RPM', 0)})
        return "RPM Surge"
    if pred_dict.get('Brake_Pressure', 100) < 10:
        #send_alert_to_back({"anomaly": "Brake Failure Risk", "value": pred_dict.get('Brake_Pressure', 100)})
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
                print(f"Anomaly likelihood: {percent:.1f}%")
                if percent >= 80:
                    print("\n🚨 HIGH WARNING: Anomaly likelihood is very high!\n")
                    send_alert_to_back({"anomaly": "High Anomaly Likelihood", "likelihood_percent": percent})
                anomaly_type = classify_anomaly(pred)
                if anomaly_type:
                    send_alert_to_back({"anomaly": anomaly_type, "likelihood_percent": percent})
                    print(f"\n🚨 WARNING: {anomaly_type} predicted in the near future!\n")
                else:
                    print(f"No anomaly predicted in the next step. (Likelihood: {percent:.1f}%)")
                    send_alert_to_back({"anomaly": "No anomaly predicted in the next step.", "likelihood_percent": percent})
                tick += 1
                if tick % avg_window == 0:
                    avg_percent = sum(anomaly_percents) / len(anomaly_percents)
                    print(f"\n=== Average anomaly likelihood over last {avg_window} seconds: {avg_percent:.1f}% ===\n")
        else:
            print(f"Engine is OFF at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Prediction paused.")
        time.sleep(interval)

if __name__ == "__main__":
    run_realtime_simulation()
