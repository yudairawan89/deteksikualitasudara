# === streamlit_app.py ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Prediksi Kualitas Udara dan Kebakaran", layout="wide")
st.markdown("""
    <h1 style='color:#004488;'>📡 Sistem Monitoring Kualitas Udara dan Prediksi Kebakaran</h1>
    <p style='font-size:16px;'>Aplikasi ini menampilkan data kualitas udara terkini (PM2.5, PM10, CO) dari sensor serta memprediksi tingkat risiko kebakaran menggunakan model GRU berbasis data Google Sheets.</p>
""", unsafe_allow_html=True)

# === Load Model dan Preprocessor ===
model = load_model("models/gru_ispu_model.h5")
scaler = joblib.load("models/scaler_ispu.pkl")
le = joblib.load("models/label_encoder_ispu.pkl")

# === Load Google Sheet ===
@st.cache_data(ttl=60)
def load_google_sheet():
    url = "https://docs.google.com/spreadsheets/d/1o6Adwn28BXco-6OrWqKJfg973rNksENoM3naJh4joYE/export?format=csv"
    return pd.read_csv(url)

data = load_google_sheet()
latest = data.iloc[-1]

# === Prediksi dengan GRU ===
input_arr = scaler.transform([[latest['PM2.5'], latest['PM10'], latest['CO']]]).reshape(1, 1, 3)
pred_prob = model.predict(input_arr)
pred_idx = np.argmax(pred_prob)
pred_label = le.inverse_transform([pred_idx])[0]

# === Tampilkan Data Sensor ===
st.subheader("📊 Data Sensor Terkini")
col1, col2, col3 = st.columns(3)
col1.metric("PM2.5", f"{latest['PM2.5']}")
col2.metric("PM10", f"{latest['PM10']}")
col3.metric("CO", f"{latest['CO']}")

st.markdown(f"""
    <div style='background-color:#0055aa;padding:15px;border-radius:10px;'>
        <h4 style='color:white;'>📡 Prediksi GRU: <u>{pred_label}</u></h4>
    </div>
""", unsafe_allow_html=True)

# === Riwayat Data + Prediksi ===
st.subheader("📋 Riwayat Data Sensor + Prediksi GRU")
data['Prediksi GRU'] = [
    le.inverse_transform([np.argmax(model.predict(scaler.transform([[row['PM2.5'], row['PM10'], row['CO']]]).reshape(1, 1, 3)))])[0]
    for _, row in data.iterrows()
]
st.dataframe(data)

# === Form Manual ===
st.subheader("🧪 Uji Prediksi Manual")
pm25 = st.number_input("PM2.5", 0.0, 500.0, 100.0)
pm10 = st.number_input("PM10", 0.0, 600.0, 150.0)
co = st.number_input("CO", 0.0, 30000.0, 12000.0)

if st.button("Prediksi Manual"):
    arr = scaler.transform([[pm25, pm10, co]]).reshape(1, 1, 3)
    pred = model.predict(arr)
    label = le.inverse_transform([np.argmax(pred)])[0]
    st.markdown(f"""
        <div style='background-color:#0066cc;padding:15px;border-radius:10px;'>
            <h4 style='color:white;'>🧪 Prediksi Manual: <u>{label}</u></h4>
        </div>
    """, unsafe_allow_html=True)

# === Tombol Refresh ===
if st.button("🔄 Tarik Data Baru"):
    st.cache_data.clear()
    st.success("✅ Data terbaru telah di-refresh.")
