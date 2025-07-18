import pandas as pd
import random
import time
from datetime import datetime
import threading
import os
import sys
if sys.platform == 'win32':
    import winsound

import requests

BACKEND_URL = "http://192.168.1.18:5000/api/ess"  # Replace with your backend's IP and port

def send_alert_to_backend(data):
    try:
        response = requests.post(BACKEND_URL, json=data, timeout=2)
        print("Sent to backend:", response.status_code)
    except Exception as e:
        print("Error sending to backend:", e)

LOG_FILE = 'seatbelt_simulation_log.csv'
MACHINE_ID = 'EXC001'
OPERATOR_ID = 'OP1001'

COLUMNS = [
    'timestamp',
    'machine_id',
    'operator_id',
    'engine_on',
    'seatbelt_status',
    'safety_alert_triggered'
]

def generate_entry():
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Randomly choose engine state: ON (True), OFF (False), or IDLE (None)
    engine_state = random.choices(['on', 'off', 'idle'], weights=[0.7, 0.2, 0.1])[0]
    if engine_state == 'on':
        engine_on = True
        seatbelt_fastened = random.random() < 0.9
    elif engine_state == 'off':
        engine_on = False
        seatbelt_fastened = random.random() < 0.1
    else:
        engine_on = None
        seatbelt_fastened = random.random() < 0.1
    seatbelt_status = 'Fastened' if seatbelt_fastened else 'Unfastened'
    # Safety alert only if engine_on is True and seatbelt is Unfastened
    safety_alert = (engine_on is True) and seatbelt_status == 'Unfastened'
    entry = {
        'engine_on': engine_on,
        'seatbelt_status': seatbelt_status,
        'safety_alert_triggered': safety_alert
    }
    send_alert_to_backend(entry)
    return {
        'timestamp': timestamp,
        'machine_id': MACHINE_ID,
        'operator_id': OPERATOR_ID,
        'engine_on': engine_on,
        'seatbelt_status': seatbelt_status,
        'safety_alert_triggered': safety_alert
    }

def save_log(df):
    df.to_csv(LOG_FILE, index=False)

def load_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    else:
        return pd.DataFrame(columns=COLUMNS)

override_next = {'engine_on': None, 'seatbelt_status': None, 'count': 0}

def simulation_loop(stop_event, pause_event):
    global override_next
    df = load_log()
    while not stop_event.is_set():
        # Wait if pause_event is set (pause requested)
        if pause_event.is_set():
            print("Simulation paused for override input...")
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.1)
            if stop_event.is_set():
                break
        # Check for override
        if override_next['count'] > 0:
            engine_on = override_next['engine_on']
            if engine_on is None:
                engine_on_val = None
            else:
                engine_on_val = engine_on
            seatbelt_status = override_next['seatbelt_status'] if override_next['seatbelt_status'] is not None else ('Fastened' if random.random() < 0.9 else 'Unfastened')
            safety_alert = (engine_on_val is True) and seatbelt_status == 'Unfastened'
            send_alert_to_backend({
        'engine_on': engine_on,
        'seatbelt_status': seatbelt_status,
        'safety_alert_triggered': safety_alert})
            entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'machine_id': MACHINE_ID,
                'operator_id': OPERATOR_ID,
                'engine_on': engine_on_val,
                'seatbelt_status': seatbelt_status,
                'safety_alert_triggered': safety_alert
            }
            override_next['count'] -= 1
        else:
            entry = generate_entry()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        save_log(df)
        print(f"Logged: {entry}")
        # Play sound if safety alert is triggered
        if entry['safety_alert_triggered']:
            try:
                if sys.platform == 'win32':
                    winsound.Beep(1000, 500)  # 1000 Hz for 0.5 sec
                else:
                    print("\a", end='')  # ASCII bell for non-Windows
            except Exception as e:
                print(f"[Warning] Could not play sound: {e}")
        sleep_time = 1
        for _ in range(sleep_time):
            if stop_event.is_set():
                break
            time.sleep(1)

def manual_override(pause_event):
    pause_event.set()
    print("Set override for the next 10 generated entries.")
    # Engine status override (ON/OFF/IDLE)
    while True:
        new_engine = input("Override next 10 entries: Engine status [ON/OFF/IDLE or Enter to skip]: ")
        if not new_engine:
            new_engine_val = None
            break
        val = new_engine.strip().lower()
        if val in ['on', 'off', 'idle']:
            if val == 'on':
                new_engine_val = True
            elif val == 'off':
                new_engine_val = False
            else:
                new_engine_val = None
            break
        else:
            print("Invalid input. Please enter 'ON', 'OFF', 'IDLE', or press Enter to skip.")

    # Seatbelt status override
    while True:
        new_seatbelt = input("Override next 20 entries: Seatbelt status [FASTENED/UNFASTENED or Enter to skip]: ")
        if not new_seatbelt:
            new_seatbelt_val = None
            break
        if new_seatbelt.strip().lower() in ['fastened', 'unfastened']:
            new_seatbelt_val = new_seatbelt.strip().capitalize()
            break
        else:
            print("Invalid input. Please enter 'FASTENED', 'UNFASTENED', or press Enter to skip.")

    global override_next
    override_next['engine_on'] = new_engine_val
    override_next['seatbelt_status'] = new_seatbelt_val
    override_next['count'] = 20
    print("Override will be applied to the next 20 generated entries.")
    pause_event.clear()

def main():
    stop_event = threading.Event()
    pause_event = threading.Event()
    sim_thread = threading.Thread(target=simulation_loop, args=(stop_event, pause_event), daemon=True)
    sim_thread.start()
    try:
        while True:
            cmd = input("Type 'override' to manually override, 'exit' to quit: ")
            if cmd == 'override':
                manual_override(pause_event)
            elif cmd == 'exit':
                stop_event.set()
                sim_thread.join()
                print("Simulation stopped.")
                break
    except KeyboardInterrupt:
        stop_event.set()
        sim_thread.join()
        print("Simulation stopped.")

if __name__ == '__main__':
    main()