
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = {
    "Diabetes": ("diabetes/model_d.pkl", "diabetes/scaler_d.pkl", 8),
    "Heart Disease": ("heart/model_h.pkl", None, 13),
    "Parkinson's": ("parkinson/model_pk.pkl", "parkinson/scaler_pk.pkl", 22),
    "Kidney Disease": ("kidney/model_kd.pkl", "kidney/scaler_kd.pkl", 24),
    "Liver Disease": ("liver/model_li.pkl", "liver/scaler_li.pkl", 10),
    "Stroke": ("stroke/model_sp.pkl", "stroke/scaler_sp.pkl", 10),
}

for name, (model_path, scaler_path, features) in models.items():
    try:
        with open(os.path.join(BASE_DIR, model_path), "rb") as f:
            model = pickle.load(f)

        sample = np.zeros((1, features))

        if scaler_path:
            with open(os.path.join(BASE_DIR, scaler_path), "rb") as f:
                scaler = pickle.load(f)

            sample = scaler.transform(sample)

        prediction = model.predict(sample)

        print(f"✅ {name}: Working | Prediction: {prediction[0]}")

    except Exception as error:
        print(f"❌ {name}: Failed | {error}")
