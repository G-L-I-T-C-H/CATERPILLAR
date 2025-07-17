import random
import time
from datetime import datetime
import threading
import pandas as pd
import os

PROXIMITY_LOG_FILE = 'proximity_simulation_log.csv'

PROXIMITY_COLUMNS = [
    'timestamp',
    'engine_on',
    'proximity_distance_m',
    'direction',
    'danger_level',
    'proximity_alert_triggered'
]

def get_engine_state(log_file='seatbelt_simulation_log.csv'):
    try:
        if not os.path.exists(log_file):
            return None
        df = pd.read_csv(log_file)
        if len(df) == 0:
            return None
        latest = df.iloc[-1]
        return latest['engine_on']
    except Exception as e:
        print(f"[WARN] Could not read engine state: {e}")
        return None

def simulate_proximity_entry(engine_on):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    directions = ['Front', 'Rear', 'Left', 'Right', 'Front-Left', 'Front-Right', 'Rear-Left', 'Rear-Right']
    if engine_on in [True, None]:  # On or idling
        # Simulate proximity in meters (0.2m = very close, 10m = safe)
        proximity_distance = round(random.uniform(0.2, 10.0), 2)
        direction = random.choice(directions)
        # Danger level logic
        if proximity_distance < 1.0:
            danger_level = 'HIGH'
            alert = True
        elif proximity_distance < 3.0:
            danger_level = 'MEDIUM'
            alert = False
        else:
            danger_level = 'LOW'
            alert = False
    else:
        proximity_distance = None
        direction = None
        danger_level = 'NONE'
        alert = False
    return {
        'timestamp': timestamp,
        'engine_on': engine_on,
        'proximity_distance_m': proximity_distance,
        'direction': direction,
        'danger_level': danger_level,
        'proximity_alert_triggered': alert
    }

def save_proximity_log(df):
    df.to_csv(PROXIMITY_LOG_FILE, index=False)

def load_proximity_log():
    if os.path.exists(PROXIMITY_LOG_FILE):
        return pd.read_csv(PROXIMITY_LOG_FILE)
    else:
        return pd.DataFrame(columns=PROXIMITY_COLUMNS)

def proximity_simulation_loop(stop_event):
    df = load_proximity_log()
    while not stop_event.is_set():
        engine_on = get_engine_state()
        entry = simulate_proximity_entry(engine_on)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        save_proximity_log(df)
        if entry['danger_level'] == 'HIGH':
            print(f"\n🚨 DANGER: Object detected VERY CLOSE ({entry['proximity_distance_m']}m, {entry['direction']})! Alert triggered!\n")
        elif entry['danger_level'] == 'MEDIUM':
            print(f"⚠️  Warning: Object detected nearby ({entry['proximity_distance_m']}m, {entry['direction']}). Stay alert.")
        elif entry['danger_level'] == 'LOW':
            print(f"Safe: No immediate danger. Closest object at {entry['proximity_distance_m']}m, {entry['direction']}.")
        else:
            print("Engine is OFF. Proximity monitoring paused.")
        time.sleep(1)

def main():
    stop_event = threading.Event()
    sim_thread = threading.Thread(target=proximity_simulation_loop, args=(stop_event,), daemon=True)
    sim_thread.start()
    try:
        while True:
            cmd = input("Type 'exit' to quit proximity simulation: ")
            if cmd == 'exit':
                stop_event.set()
                sim_thread.join()
                print("Proximity simulation stopped.")
                break
    except KeyboardInterrupt:
        stop_event.set()
        sim_thread.join()
        print("Proximity simulation stopped.")

if __name__ == '__main__':
    main()
