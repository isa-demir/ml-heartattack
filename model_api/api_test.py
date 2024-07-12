import json
import requests

url = 'https://fastapi-app-ynuk.onrender.com/heart_prediction'

input_data_positive = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trtbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalachh": 150,
    "exng": 0,
    "oldpeak": 2.3,
    "slp": 0,
    "caa": 0,
    "thall": 1
}

input_data_positive2 = {
    "age": 40,
    "sex": 1,
    "cp": 3,
    "trtbps": 140,
    "chol": 199,
    "fbs": 0,
    "restecg": 1,
    "thalachh": 178,
    "exng": 1,
    "oldpeak": 1.4,
    "slp": 2,
    "caa": 0,
    "thall": 3
}

input_data_negative = {
    "age": 48,
    "sex": 1,
    "cp": 1,
    "trtbps": 110,
    "chol": 229,
    "fbs": 0,
    "restecg": 1,
    "thalachh": 168,
    "exng": 0,
    "oldpeak": 1,
    "slp": 0,
    "caa": 0,
    "thall": 3,
}

input_data_negative2 = {
    "age": 66,
    "sex": 0,
    "cp": 0,
    "trtbps": 178,
    "chol": 228,
    "fbs": 1,
    "restecg": 1,
    "thalachh": 165,
    "exng": 1,
    "oldpeak": 1,
    "slp": 1,
    "caa": 2,
    "thall": 3,
}




input_json = json.dumps(input_data_negative2)

response = requests.post(url, data=input_json)

print(response.text)
