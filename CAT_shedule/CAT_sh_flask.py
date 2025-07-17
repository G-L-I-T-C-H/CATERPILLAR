import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ✅ Load Models
time_model = load_model("./task_time_prediction.h5", compile=False)
complexity_model = load_model("./task_complexity_model.h5", compile=False)

# ✅ Load Encoders & Scalers
encoders_time = joblib.load("./encoders_time.pkl")
encoders_complexity = joblib.load("./encoders_complexity.pkl")
scaler_X = joblib.load("./scaler_X.pkl")
scaler_y = joblib.load("./scaler_y.pkl")

# ✅ OpenWeather API Config
API_KEY = "c9cf60692be3767531393fcadc8dedf8"  # Your API key
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_current_weather(city="Coimbatore", country="IN"):
    params = {
        "q": f"{city},{country}",
        "appid": API_KEY,
        "units": "metric"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if response.status_code == 200:
        weather_main = data['weather'][0]['main']
        temp_c = data['main']['temp']
        return weather_main, temp_c
    else:
        return "Unknown", 25.0

def apply_break_policy(base_minutes, interval_min=120, break_min=10):
    extra = int(base_minutes // interval_min) * break_min
    return base_minutes + extra

def predict_task(operator_id, machine_id, machine_model, terrain_type, task_type, task_output, city="Coimbatore"):
    # ✅ Get Weather
    weather, temp = get_current_weather(city)

    # ✅ Predict Complexity
    complexity_input = []
    for col in ['Machine Model', 'Terrain Type', 'Task Type', 'Task Target/Output', 'Weather Condition']:
        if col == 'Task Target/Output':
            complexity_input.append(task_output)
        elif col == 'Weather Condition':
            val = encoders_complexity[col].transform([weather])[0] if weather in encoders_complexity[col].classes_ else -1
            complexity_input.append(val)
        else:
            key = col.lower().replace(" ", "_")
            val = encoders_complexity[col].transform([locals()[key]])[0] if locals()[key] in encoders_complexity[col].classes_ else -1
            complexity_input.append(val)

    complexity_input = np.array(complexity_input).reshape(1, -1)
    predicted_complexity_class = np.argmax(complexity_model.predict(complexity_input))
    complexity_label = ['Low', 'Medium', 'High'][predicted_complexity_class]

    # ✅ Predict Time
    input_dict = {
        'Operator ID': operator_id,
        'Machine ID': machine_id,
        'Machine Model': machine_model,
        'Terrain Type': terrain_type,
        'Weather Condition': weather,
        'Task Type': task_type,
        'Task Complexity Level': complexity_label,
        'Task Target/Output': task_output
    }

    encoded = []
    # Use only features that match training
    for col in ['Operator ID', 'Machine Model', 'Terrain Type', 'Task Type', 'Weather Condition']:
        if input_dict[col] in encoders_time[col].classes_:
            val = encoders_time[col].transform([input_dict[col]])[0]
        else:
            val = -1
        encoded.append(val)

    task_output_scaled = scaler_X.transform([[task_output]])[0][0]
    encoded.append(task_output_scaled)  # now total = 6 features

    X_input = np.array(encoded).reshape(1, -1)
    predicted_scaled = time_model.predict(X_input)
    predicted_minutes = scaler_y.inverse_transform(predicted_scaled)[0][0]
    predicted_minutes = apply_break_policy(predicted_minutes)

    # ✅ Compute Start & End Times
    tz = ZoneInfo("Asia/Kolkata")
    start = datetime.now(tz=tz)
    end = start + timedelta(minutes=float(predicted_minutes))

    return {
        "Predicted Minutes": round(predicted_minutes, 2),
        "Predicted Task Complexity": complexity_label,
        "Weather": weather,
        "Temperature (°C)": temp,
        "Start Time": start.isoformat(timespec="seconds"),
        "End Time": end.isoformat(timespec="seconds")
    }

# ✅ Test
if __name__ == "__main__":
    result = predict_task(
        operator_id="OP1005",
        machine_id="CAT123",
        machine_model="CAT320D",
        terrain_type="Rocky",
        task_type="Digging",
        task_output=50,
        city="Coimbatore"
    )
    print("\nPrediction Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
