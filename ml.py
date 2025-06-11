import streamlit as st
import pandas as pd
import pickle

# Load model dan encoder
with open("model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    le = pickle.load(encoder_file)

# Judul aplikasi
st.title("📱 Rekomendasi Smartphone Berdasarkan Kebutuhan Pengguna")

# Input dari pengguna
st.subheader("Masukkan Spesifikasi yang Diinginkan:")
internal = st.number_input("💾 Memori Internal (GB)", min_value=1, max_value=1024, value=128)
baterai = st.number_input("🔋 Kapasitas Baterai (mAh)", min_value=1000, max_value=10000, value=5000)

# Prediksi
if st.button("🔍 Prediksi Kebutuhan"):
    data_input = pd.DataFrame([[internal, baterai]],
                              columns=["memory_internal", "battery"])
    
    prediksi_kode = clf.predict(data_input)[0]
    kebutuhan = le.inverse_transform([prediksi_kode])[0]

    st.success(f"Kategori Kebutuhan Smartphone: **{kebutuhan}**")

    # Tampilkan deskripsi berdasarkan kategori
    if kebutuhan == "Gaming":
        st.info("🎮 **Gaming**: Direkomendasikan untuk pengguna yang bermain game berat. Memiliki RAM besar dan chipset performa tinggi.")
    elif kebutuhan == "Multimedia":
        st.info("🎥 **Multimedia**: Cocok untuk streaming, menonton video, dan konsumsi media. Biasanya memiliki baterai besar.")
    elif kebutuhan == "Basic":
        st.info("📱 **Basic**: Digunakan untuk kebutuhan ringan seperti chat, browsing, dan aplikasi standar.")
