import requests
payload = {
    "operator_id": "OP1005",
    "machine_id": "CAT123",
    "machine_model": "CAT320D",
    "terrain_type": "Rocky",
    "task_type": "Digging",
    "task_output": 50,
    "city": "Coimbatore"
}
r = requests.post("http://127.0.0.1:5000/predict", json=payload)
print(r.status_code)
print(r.json())
