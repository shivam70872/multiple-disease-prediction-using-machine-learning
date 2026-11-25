import os
import pickle

import numpy as np
import streamlit as st

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------- Local MediBot Configuration --------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b"


MEDIBOT_INSTRUCTIONS = """

You are MediBot, the AI assistant of Shivam's
Multi-Disease Prediction System.

You have access to the following verified project documentation.
You MUST use these facts when answering questions about this project.

==============================
PROJECT OVERVIEW
==============================

The project is a Streamlit-based Multi-Disease Prediction System.

It contains six independently trained machine learning models.

==============================
VERIFIED MODEL INFORMATION
==============================

1. DIABETES
Algorithm: XGBoost Classifier
Input features: 8
Scaler: StandardScaler

Features:
Pregnancies, Glucose, Blood Pressure, Skin Thickness,
Insulin, BMI, Diabetes Pedigree Function, Age.

2. HEART DISEASE
Algorithm: Logistic Regression
Input features: 13
Scaler: No scaler is applied in the current app.py.

Features:
Age, Sex, Chest Pain Type, Resting Blood Pressure,
Cholesterol, Fasting Blood Sugar, Rest ECG,
Maximum Heart Rate, Exercise Induced Angina,
Oldpeak, Slope, CA, Thal.

3. PARKINSON'S DISEASE
Algorithm: Support Vector Classifier (SVC)
Kernel: Linear
Input features: 22
Scaler: StandardScaler

PARKINSONS_FEATURE_DEFINITIONS = 

Parkinson's prediction uses 22 voice-analysis features.

The model uses SVC(kernel='linear').

IMPORTANT: These are acoustic measurements.
Do not invent medical expansions for their abbreviations.

Verified terminology:

MDVP: Multi-Dimensional Voice Program
Fo: Average fundamental frequency
Fhi: Maximum fundamental frequency
Flo: Minimum fundamental frequency
Jitter(%): Percentage variation in fundamental frequency
Jitter(Abs): Absolute jitter
RAP: Relative Average Perturbation
PPQ: Pitch Period Perturbation Quotient
DDP: Difference of Differences of Periods
Shimmer: Variation in voice amplitude
Shimmer(dB): Shimmer expressed in decibels
APQ3: Three-point Amplitude Perturbation Quotient
APQ5: Five-point Amplitude Perturbation Quotient
APQ: Amplitude Perturbation Quotient
DDA: Difference of Differences of Amplitudes
NHR: Noise-to-Harmonics Ratio
HNR: Harmonics-to-Noise Ratio
RPDE: Recurrence Period Density Entropy
DFA: Detrended Fluctuation Analysis
spread1: Nonlinear measure of fundamental frequency variation
spread2: Nonlinear measure of fundamental frequency variation
D2: Correlation dimension
PPE: Pitch Period Entropy

Never invent expansions for technical terms.
If uncertain, explain the feature using its abbreviation
and say that its exact definition needs verification.



4. KIDNEY DISEASE
Algorithm: XGBoost Classifier
Input features: 24
Scaler: StandardScaler

Features:
Age, Blood Pressure, Specific Gravity, Albumin,
Sugar, Red Blood Cells, Pus Cell, Pus Cell Clumps,
Bacteria, Blood Glucose Random, Blood Urea,
Serum Creatinine, Sodium, Potassium, Hemoglobin,
Packed Cell Volume, White Blood Cell Count,
Red Blood Cell Count, Hypertension, Diabetes Mellitus,
Coronary Artery Disease, Appetite, Pedal Edema,
Anemia.

5. LIVER DISEASE
Algorithm: XGBoost Classifier
Input features: 10
Scaler: StandardScaler

Features:
Age, Gender, Total Bilirubin, Direct Bilirubin,
Alkaline Phosphotase, SGPT, SGOT, Total Proteins,
Albumin, Albumin and Globulin Ratio.

6. STROKE
Algorithm: XGBoost Classifier
Input features: 10
Scaler: StandardScaler

Features:
Gender, Age, Hypertension, Heart Disease,
Ever Married, Work Type, Residence Type,
Average Glucose Level, BMI, Smoking Status.

==============================
IMPORTANT PROJECT RULES
==============================

1. There are exactly six disease prediction modules.
2. Do not invent additional models or algorithms.
3. Do not claim that the system uses Random Forest,
   Decision Trees, Neural Networks, KNN or a hybrid
   ensemble unless verified project documentation supports it.
4. Do not invent input features.
5. Do not confuse the features of different diseases.
6. If asked how many features a model uses, give the
   exact number listed above.
7. If asked about an unknown project detail, say:
   "I don't have verified information about that
   implementation detail."
8. Do not invent model accuracy, dataset size,
   validation results or performance statistics.
9. Explain the difference between the project's actual
   implementation and possible future improvements.
10. Give answers in simple, understandable language.

==============================
YOUR RESPONSIBILITIES
==============================

- Explain machine learning concepts.
- Explain the six disease modules.
- Explain medical terminology.
- Explain the model algorithms.
- Help users navigate the application.
- Help students understand this project.
- Answer general educational health questions.

==============================
MEDICAL SAFETY
==============================

- You are not a doctor.
- Never diagnose a patient.
- Never prescribe medication.
- Never guarantee prediction accuracy.
- Machine learning outputs are experimental screening
  results, not medical diagnoses.
- Recommend consulting qualified healthcare professionals.
- For emergencies, recommend contacting emergency services.
- Never request personally identifiable medical information.

Be friendly, accurate, concise and honest.


STRICT FACTUAL ACCURACY RULES:

1. Never invent expansions of medical or technical abbreviations.
2. Never invent dataset characteristics.
3. Never describe an algorithm as high-dimensional without evidence.
4. Distinguish verified project facts from general ML knowledge.
5. If you don't know something, admit it.
6. Do not present assumptions as verified facts.
7. Use simple language when explaining technical features.
"""

# -------------------- Local LLM Response --------------------

def get_medibot_response(chat_history):

    messages = [
        {
            "role": "system",
            "content": MEDIBOT_INSTRUCTIONS
        }
    ]

    # Keep recent conversation context
    for message in chat_history[-10:]:

        if message["role"] in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"]
            })

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 3072,
                    "num_predict": 500,
                    "top_p": 0.8
                }
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

    except requests.exceptions.ConnectionError:

        return (
            "⚠️ MediBot cannot connect to Ollama. "
            "Please open Ollama and make sure its local server is running."
        )

    except requests.exceptions.Timeout:

        return (
            "⏳ MediBot is taking too long to respond. "
            "Please try a shorter question."
        )

    except requests.exceptions.RequestException:

        return (
            "⚠️ Something went wrong while communicating "
            "with your local AI model."
        )

    except (KeyError, ValueError):

        return "⚠️ MediBot received an invalid response."



# -------------------- MediBot AI Function --------------------
    

st.set_page_config(
    page_title="Multi Disease Prediction",
    page_icon="🩺",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_pickle(*path_parts):
    """Load a model/scaler stored relative to app.py."""
    file_path = os.path.join(BASE_DIR, *path_parts)

    if not os.path.exists(file_path):
        st.error(f"Required file not found: {file_path}")
        st.stop()

    with open(file_path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_models():
    return {
        "dbt_model": load_pickle("diabetes", "model_d.pkl"),
        "dbt_scaler": load_pickle("diabetes", "scaler_d.pkl"),

        "heart_model": load_pickle("heart", "model_h.pkl"),

        "pk_model": load_pickle("parkinson", "model_pk.pkl"),
        "pk_scaler": load_pickle("parkinson", "scaler_pk.pkl"),

        "kd_model": load_pickle("kidney", "model_kd.pkl"),
        "kd_scaler": load_pickle("kidney", "scaler_kd.pkl"),

        "li_model": load_pickle("liver", "model_li.pkl"),
        "li_scaler": load_pickle("liver", "scaler_li.pkl"),

        "sp_model": load_pickle("stroke", "model_sp.pkl"),
        "sp_scaler": load_pickle("stroke", "scaler_sp.pkl"),
    }


models = load_models()


def show_result(prediction, disease_name):
    if int(prediction[0]) == 1:
        st.error(f"⚠️ The person is likely to have **{disease_name}**.")
    else:
        st.success(f"✅ The person is not likely to have **{disease_name}**.")


st.sidebar.title("🩺 Multi-Disease Prediction System")

page = st.sidebar.radio(
    "Go to:",
    (
        "🏠 Home",
        "Diabetes",
        "Heart Disease",
        "Parkinson's",
        "Kidney Disease",
        "Liver Disease",
        "Stroke",
    ),
)


# -------------------- Home + MediBot AI --------------------

if page == "🏠 Home":

    st.title("🩺 Multi-Disease Prediction System")

    st.markdown(
        "### An AI-powered healthcare screening platform"
    )

    st.info(
        "👈 Select a disease from the sidebar to start a prediction."
    )

    st.divider()

    # MediBot Header

    col1, col2 = st.columns([1, 5])

    with col1:
        st.markdown("# 🤖")

    with col2:
        st.subheader("MediBot AI")
        st.caption("Your intelligent health information assistant")

    st.divider()

    # Initialize chat history

    if "medibot_history" not in st.session_state:

        st.session_state.medibot_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I'm MediBot, your AI health assistant.\n\n"
                    "I can help you understand our six disease "
                    "prediction modules, medical terminology and "
                    "how to use this application.\n\n"
                    "How can I help you today?"
                )
            }
        ]

    # Clear Chat Button

    if st.button("🗑️ Clear Conversation"):

        st.session_state.medibot_history = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 I'm MediBot. "
                    "How can I help you today?"
                )
            }
        ]

        st.rerun()

    # Display Previous Messages

    for message in st.session_state.medibot_history:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # User Input

    user_prompt = st.chat_input(
        "Ask MediBot anything..."
    )

    if user_prompt:

        # Save user message

        st.session_state.medibot_history.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        # Display user message

        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate AI response

        with st.chat_message("assistant"):

            with st.spinner("MediBot is thinking..."):

                answer = get_medibot_response(
                    st.session_state.medibot_history
                )

                st.markdown(answer)

        # Save AI response

        st.session_state.medibot_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    st.divider()

    st.caption(
        "⚕️ MediBot provides educational information only. "
        "It does not replace professional medical advice."
    )



# -------------------- Diabetes --------------------

elif page == "Diabetes":
    st.title("🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose Level", 0, 300)
        blood_pressure = st.number_input("Blood Pressure", 0, 200)
        skin_thickness = st.number_input("Skin Thickness", 0, 100)

    with col2:
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 70.0, key="diabetes_bmi")
        pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 5.0)
        age = st.number_input("Age", 1, 120, key="diabetes_age")

    if st.button("🔍 Predict Diabetes"):
        features = np.array([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            pedigree,
            age,
        ]])

        scaled_features = models["dbt_scaler"].transform(features)
        prediction = models["dbt_model"].predict(scaled_features)

        show_result(prediction, "Diabetes")


# -------------------- Heart Disease --------------------

elif page == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, key="heart_age")

        sex_text = st.selectbox("Sex", ("Female", "Male"))
        sex = 1 if sex_text == "Male" else 0

        cp = st.number_input("Chest Pain Type (0-3)", 0, 3)
        trestbps = st.number_input("Resting Blood Pressure", 0, 250)
        chol = st.number_input("Cholesterol", 0, 600)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))

    with col2:
        restecg = st.number_input("Rest ECG (0-2)", 0, 2)
        thalach = st.number_input("Max Heart Rate", 0, 300)
        exang = st.selectbox("Exercise Induced Angina", (0, 1))
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
        slope = st.number_input("Slope (0-2)", 0, 2)
        ca = st.number_input("CA (0-4)", 0, 4)
        thal = st.number_input("Thal (0-3)", 0, 3)

    if st.button("🔍 Predict Heart Disease"):
        features = np.array([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
        ]])

        prediction = models["heart_model"].predict(features)

        show_result(prediction, "Heart Disease")


# -------------------- Parkinson's --------------------

elif page == "Parkinson's":
    st.title("🧠 Parkinson's Disease Prediction")

    parameters = [
        "MDVP:Fo(Hz)",
        "MDVP:Fhi(Hz)",
        "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)",
        "MDVP:Jitter(Abs)",
        "MDVP:RAP",
        "MDVP:PPQ",
        "Jitter:DDP",
        "MDVP:Shimmer",
        "MDVP:Shimmer(dB)",
        "Shimmer:APQ3",
        "Shimmer:APQ5",
        "MDVP:APQ",
        "Shimmer:DDA",
        "NHR",
        "HNR",
        "RPDE",
        "DFA",
        "spread1",
        "spread2",
        "D2",
        "PPE",
    ]

    features = []

    for parameter in parameters:
        features.append(st.number_input(parameter, value=0.0))

    if st.button("🔍 Predict Parkinson's"):
        features = np.array([features])

        scaled_features = models["pk_scaler"].transform(features)
        prediction = models["pk_model"].predict(scaled_features)

        show_result(prediction, "Parkinson's Disease")


# -------------------- Kidney Disease --------------------

elif page == "Kidney Disease":
    st.title("🧪 Chronic Kidney Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 120, 45, key="kidney_age")
        bp = st.number_input("Blood Pressure", 0, 200, 60)
        sg = st.number_input("Specific Gravity", 1.0, 2.0, 1.02, step=0.01)
        al = st.number_input("Albumin", 0.0, 10.0, 0.0)
        su = st.number_input("Sugar", 0.0, 10.0, 0.0)

        rbc_text = st.selectbox("Red Blood Cells", ["Abnormal (0)", "Normal (1)"])
        rbc = 1 if rbc_text == "Normal (1)" else 0

        pc_text = st.selectbox("Pus Cell", ["Abnormal (0)", "Normal (1)"])
        pc = 1 if pc_text == "Normal (1)" else 0

        pcc_text = st.selectbox("Pus Cell Clumps", ["No (0)", "Yes (1)"])
        pcc = 1 if pcc_text == "Yes (1)" else 0

        ba_text = st.selectbox("Bacteria", ["No (0)", "Yes (1)"])
        ba = 1 if ba_text == "Yes (1)" else 0

        bgr = st.number_input("Blood Glucose Random", 0.0, 500.0, 114.0)
        bu = st.number_input("Blood Urea", 0.0, 200.0, 26.0)
        sc = st.number_input("Serum Creatinine", 0.0, 15.0, 0.7)

    with col2:
        sod = st.number_input("Sodium", 100.0, 200.0, 141.0)
        pot = st.number_input("Potassium", 2.0, 10.0, 4.2)
        hemo = st.number_input("Hemoglobin", 3.0, 20.0, 15.0)
        pcv = st.number_input("Packed Cell Volume", 10.0, 60.0, 43.0)
        wc = st.number_input("White Blood Cell Count", 1000.0, 30000.0, 9200.0)
        rc = st.number_input("Red Blood Cell Count", 2.0, 10.0, 5.8)

        htn = st.selectbox("Hypertension", (0, 1))
        dm = st.selectbox("Diabetes Mellitus", (0, 1))
        cad = st.selectbox("Coronary Artery Disease", (0, 1))

        appetite_text = st.selectbox("Appetite", ["Good (0)", "Poor (1)"])
        appetite = 1 if appetite_text == "Poor (1)" else 0

        pe = st.selectbox("Pedal Edema", (0, 1))
        ane = st.selectbox("Anemia", (0, 1))

    if st.button("🔍 Predict Kidney Disease"):
        features = np.array([[
            age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc,
            sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appetite, pe, ane,
        ]])

        scaled_features = models["kd_scaler"].transform(features)
        prediction = models["kd_model"].predict(scaled_features)

        show_result(prediction, "Kidney Disease")


# -------------------- Liver Disease --------------------

elif page == "Liver Disease":
    st.title("🧬 Liver Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, key="liver_age")

        gender_text = st.selectbox(
            "Gender",
            ("Female", "Male"),
            key="liver_gender",
        )
        gender = 1 if gender_text == "Male" else 0

        total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0)
        direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 10.0)
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 0, 2000)

    with col2:
        sgpt = st.number_input("Alanine Aminotransferase (SGPT)", 0, 2000)
        sgot = st.number_input("Aspartate Aminotransferase (SGOT)", 0, 2000)
        total_proteins = st.number_input("Total Proteins", 0.0, 10.0)
        albumin = st.number_input("Albumin", 0.0, 10.0)
        ag_ratio = st.number_input("Albumin and Globulin Ratio", 0.0, 5.0)

    if st.button("🔍 Predict Liver Disease"):
        features = np.array([[
            age,
            gender,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            sgpt,
            sgot,
            total_proteins,
            albumin,
            ag_ratio,
        ]])

        scaled_features = models["li_scaler"].transform(features)
        prediction = models["li_model"].predict(scaled_features)

        show_result(prediction, "Liver Disease")


# -------------------- Stroke --------------------

elif page == "Stroke":
    st.title("🧠 Stroke Prediction")

    col1, col2 = st.columns(2)

    with col1:
        gender_text = st.selectbox(
            "Gender",
            ("Female", "Male"),
            key="stroke_gender",
        )
        gender = 1 if gender_text == "Male" else 0

        age = st.number_input("Age", 1, 120, key="stroke_age")
        hypertension = st.selectbox("Hypertension", (0, 1), key="stroke_hypertension")
        heart_disease = st.selectbox("Heart Disease", (0, 1), key="stroke_heart_disease")
        ever_married = st.selectbox("Ever Married", (0, 1))

    with col2:
        work_type = st.number_input("Work Type (0-3)", 0, 3)

        residence_text = st.selectbox("Residence Type", ("Rural", "Urban"))
        residence_type = 1 if residence_text == "Urban" else 0

        avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0)
        bmi = st.number_input("BMI", 0.0, 70.0, key="stroke_bmi")
        smoking_status = st.number_input("Smoking Status (0-3)", 0, 3)

    if st.button("🔍 Predict Stroke"):
        features = np.array([[
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            avg_glucose_level,
            bmi,
            smoking_status,
        ]])

        scaled_features = models["sp_scaler"].transform(features)
        prediction = models["sp_model"].predict(scaled_features)

        show_result(prediction, "Stroke")