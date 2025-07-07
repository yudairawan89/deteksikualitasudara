import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from catboost import CatBoostClassifier

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Prediksi Kualitas Udara dan Kebakaran", layout="wide")
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("upi.png", width=120)
with col_title:
    st.markdown("""
        <h1 style='color:#004488;margin-bottom:0;'>📡 Sistem Monitoring Kualitas Udara dan Prediksi Kebakaran</h1>
        <p style='font-size:16px;'>Aplikasi ini menampilkan data kualitas udara terkini dari sensor IoT serta memprediksi tingkat risiko kebakaran menggunakan model Machine Learning CatBoost.</p>
    """, unsafe_allow_html=True)

# === Load Model dan Preprocessor ===
model = CatBoostClassifier()
model.load_model("catboost_ispu_model.cbm")
scaler = joblib.load("scaler_ispu.pkl")
le = joblib.load("label_encoder_ispu.pkl")

# === Load Google Sheet ===
@st.cache_data(ttl=60)
def load_google_sheet():
    url = "https://docs.google.com/spreadsheets/d/1o6Adwn28BXco-6OrWqKJfg973rNksENoM3naJh4joYE/export?format=csv"
    return pd.read_csv(url)

# === Tombol Refresh ===
if st.button("🔄 Tarik Data Baru"):
    st.cache_data.clear()
    st.success("✅ Data terbaru berhasil dimuat ulang.")

# === Ambil data terakhir ===
data = load_google_sheet()
latest = data.iloc[-1]

# === Prediksi dengan CatBoost ===
input_arr = scaler.transform([[latest['PM2.5'], latest['PM10'], latest['CO']]])
pred_idx = int(model.predict(input_arr)[0])
pred_label = le.inverse_transform([pred_idx])[0]

# === Tampilkan Data Sensor ===
st.subheader("📊 Data Sensor Terkini")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p style='color:#004488;font-size:20px;font-weight:bold;'>PM2.5</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{latest['PM2.5']}")

with col2:
    st.markdown("<p style='color:#004488;font-size:20px;font-weight:bold;'>PM10</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{latest['PM10']}")

with col3:
    st.markdown("<p style='color:#004488;font-size:20px;font-weight:bold;'>CO</p>", unsafe_allow_html=True)
    st.metric(label="", value=f"{latest['CO']}")

# === Format Tanggal Hari Ini ===
now = datetime.now()
hari = now.strftime("%A")
tanggal = now.strftime("%d %B %Y")

# === Tampilkan Prediksi Kualitas Udara ===
st.markdown(f"""
    <div style='background-color:#0000cc;padding:18px;border-radius:10px;margin-top:10px;'>
        <h4 style='color:white;text-align:center;'>Pada hari <b>{hari}</b>, tanggal <b>{tanggal}</b>, Berdasarkan dari data sensor daerah ini memiliki kualitas udara: 
        <u style='font-size:20px;'> {pred_label} </u></h4>
    </div>
""", unsafe_allow_html=True)

# === Riwayat Data + Prediksi ===
st.subheader("📋 Riwayat Data Sensor dan Hasil Daeteksi ")
data['Prediksi CatBoost'] = [
    le.inverse_transform([int(model.predict(scaler.transform([[row['PM2.5'], row['PM10'], row['CO']]]))[0])])[0]
    for _, row in data.iterrows()
]
st.dataframe(data)

# === Form Manual ===
st.subheader("🧪 Uji Prediksi Manual")
pm25 = st.number_input("PM2.5", 0.0, 500.0, 100.0)
pm10 = st.number_input("PM10", 0.0, 600.0, 150.0)
co = st.number_input("CO", 0.0, 30000.0, 12000.0)

if st.button("Prediksi Manual"):
    arr = scaler.transform([[pm25, pm10, co]])
    pred = model.predict(arr)
    label = le.inverse_transform([int(pred[0])])[0]
    st.markdown(f"""
        <div style='background-color:#0066cc;padding:15px;border-radius:10px;margin-top:10px;'>
            <h4 style='color:white;text-align:center;'>🧪 Hasil Prediksi Manual: <u>{label}</u></h4>
        </div>
    """, unsafe_allow_html=True)

# === Footer ===
st.markdown("""<hr style='margin-top:40px;'>""", unsafe_allow_html=True)
st.markdown("""
    <div style='background-color:blue;padding:15px;border-radius:10px;text-align:center;'>
        <h4 style='color:white;'>Deteksi Kualitas Udara Menggunakan IoT - Machine Learning Model</h4>
        <p style='color:lightgray;'>Dikembangkan oleh Dosen Universitas Putera Indonesia YPTK Padang Tahun 2025</p>
    </div>
""", unsafe_allow_html=True)
