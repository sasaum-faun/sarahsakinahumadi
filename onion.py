import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Kualitas Benih Bawang", layout="wide", page_icon="🌱")

# --- LOAD DATA & MODEL TRAINING ---
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv('onion_seed_quality_dataset.csv')
    
    # Encoding kolom kategori (Seed_Color)
    le_color = LabelEncoder()
    df['Seed_Color_Encoded'] = le_color.fit_transform(df['Seed_Color'])
    
    # Fitur yang digunakan untuk prediksi
    features = [
        'Seed_Weight (mg)', 'Seed_Size (mm)', 'Seed_Color_Encoded', 
        'Moisture_Content (%)', 'Genetic_Marker_1', 'Genetic_Marker_2', 
        'Soil_pH', 'Temperature (°C)', 'Humidity (%)', 'Germination_Rate (%)'
    ]
    
    X = df[features]
    y = df['Seed_Quality']
    
    # Inisialisasi dan latih model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, le_color

model, le_color = train_model()

# --- INTERFACE UTAMA ---
st.title("🌱 Onion Seed Quality Predictor")
st.markdown("""
Aplikasi ini memprediksi apakah benih bawang masuk kategori **High**, **Medium**, atau **Low** berdasarkan parameter fisik dan lingkungan.
""")

st.sidebar.header("📥 Masukkan Parameter Benih")

def user_input_features():
    # Membuat slider berdasarkan rentang data di dataset
    weight = st.sidebar.slider('Seed Weight (mg)', 2.0, 4.0, 3.0)
    size = st.sidebar.slider('Seed Size (mm)', 1.5, 3.0, 2.2)
    color = st.sidebar.selectbox('Seed Color', ['Light', 'Medium', 'Dark'])
    moisture = st.sidebar.slider('Moisture Content (%)', 7.0, 10.0, 8.5)
    marker1 = st.sidebar.radio('Genetic Marker 1', [0, 1])
    marker2 = st.sidebar.radio('Genetic Marker 2', [0, 1])
    ph = st.sidebar.slider('Soil pH', 6.0, 8.0, 7.0)
    temp = st.sidebar.slider('Temperature (°C)', 20.0, 35.0, 25.0)
    hum = st.sidebar.slider('Humidity (%)', 50.0, 80.0, 65.0)
    germination = st.sidebar.slider('Germination Rate (%)', 50.0, 100.0, 85.0)
    
    # Transform warna ke angka
    color_encoded = le_color.transform([color])[0]
    
    data = {
        'Seed_Weight (mg)': weight,
        'Seed_Size (mm)': size,
        'Seed_Color_Encoded': color_encoded,
        'Moisture_Content (%)': moisture,
        'Genetic_Marker_1': marker1,
        'Genetic_Marker_2': marker2,
        'Soil_pH': ph,
        'Temperature (°C)': temp,
        'Humidity (%)': hum,
        'Germination_Rate (%)': germination
    }
    return pd.DataFrame(data, index=[0])

# Mendapatkan input dari user
input_df = user_input_features()

# --- BAGIAN PREDIKSI ---
st.subheader("🔍 Hasil Analisis Prediksi")

col1, col2 = st.columns([1, 1])

with col1:
    # Lakukan prediksi
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Styling hasil prediksi
    result = prediction[0]
    if result == 'High':
        st.success(f"### Kualitas Benih: {result}")
    elif result == 'Medium':
        st.warning(f"### Kualitas Benih: {result}")
    else:
        st.error(f"### Kualitas Benih: {result}")
        
    st.write("Model memprediksi kategori kualitas berdasarkan input yang Anda berikan di sidebar.")

with col2:
    st.write("**Probabilitas Prediksi:**")
    prob_df = pd.DataFrame(prediction_proba, columns=model.classes_)
    st.bar_chart(prob_df.T)

st.divider()

# Menampilkan parameter yang diinput user dalam bentuk tabel
st.subheader("📋 Detail Parameter Input")
st.table(input_df)