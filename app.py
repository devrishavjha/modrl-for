from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ----------------------
# 1. Generate synthetic data
# ----------------------
np.random.seed(42)
n_samples = 5000

# Normal ranges for sensors
hr = np.random.randint(50, 120, n_samples)           # Heart Rate
systolic = np.random.randint(90, 180, n_samples)     # Systolic BP
diastolic = np.random.randint(60, 120, n_samples)    # Diastolic BP
spo2 = np.random.randint(85, 100, n_samples)         # Oxygen Saturation
temp = np.random.uniform(35.0, 40.0, n_samples)     # Body Temp
movement = np.random.randint(0, 2, n_samples)        # Bed movement 0/1

X = np.column_stack([hr, systolic, diastolic, spo2, temp, movement])

# Label generation based on medical rules
y = []
for i in range(n_samples):
    score = 0
    if hr[i] < 60 or hr[i] > 100:
        score += 1
    if systolic[i] < 90 or systolic[i] > 140:
        score += 1
    if diastolic[i] < 60 or diastolic[i] > 90:
        score += 1
    if spo2[i] < 92:
        score += 2
    if temp[i] < 36 or temp[i] > 38:
        score += 1
    if movement[i] == 1:
        score += 0  # Optional: movement score

    # Map to 0-4
    if score == 0:
        y.append(0)
    elif score <= 2:
        y.append(1)
    elif score <= 4:
        y.append(2)
    elif score == 5:
        y.append(3)
    else:
        y.append(4)

y = np.array(y)

# ----------------------
# 2. Train RandomForest model
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save the model
if not os.path.exists("models"):
    os.mkdir("models")
pickle.dump(model, open("models/patient_model.pkl", "wb"))

print("Model trained and saved!")

# ----------------------
# 3. Flask API endpoint with rule-based override
# ----------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hr_val = data["HR"]
    sys_val = data["Systolic"]
    dia_val = data["Diastolic"]
    spo2_val = data["SpO2"]
    temp_val = data["Temp"]
    move_val = data["Movement"]

    # ML prediction
    features = [hr_val, sys_val, dia_val, spo2_val, temp_val, move_val]
    X_input = np.array(features).reshape(1, -1)
    pred = model.predict(X_input)[0]

    # ----------------------
    # Rule-based overrides for critical values
    # ----------------------
    # Emergency override
    if (hr_val < 40 or hr_val > 160 or
        sys_val < 80 or sys_val > 190 or
        dia_val < 50 or dia_val > 130 or
        spo2_val < 85 or
        temp_val < 35 or temp_val > 41 or
        move_val > 2):
        pred = 4  # Emergency

    # Alert override
    elif (hr_val < 50 or hr_val > 140 or
          sys_val < 90 or sys_val > 180 or
          dia_val < 55 or dia_val > 120 or
          spo2_val < 90 or
          temp_val < 35.5 or temp_val > 39 or
          move_val == 2):
        pred = max(pred, 3)  # Alert

    return jsonify({"Seriousness": int(pred)})

