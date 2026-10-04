import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Konfigurasi Halaman Web
st.set_page_config(
    page_title="Credit Risk Assessment & Explainable AI",
    page_icon="💳",
    layout="wide"
)

# 1. Memuat Artefak Model & Scaler
@st.cache_resource
def load_models():
    model = joblib.load('xgboost_credit_model.pkl')
    scaler = joblib.load('scaler_credit.pkl')
    return model, scaler

model, scaler = load_models()

# Inisialisasi SHAP Explainer
explainer = shap.TreeExplainer(model)

st.title("💳 Sistem Penilaian Risiko Gagal Bayar Kredit Nasabah")
st.markdown("""
Aplikasi purwarupa berbasis **Machine Learning (XGBoost)** dan **Explainable AI (SHAP)** 
untuk memprediksi probabilitas gagal bayar (*default payment*) serta mengurai kontribusi faktor risiko calon debitur.
""")

# 2. Form Input Parameter Nasabah pada Sidebar
st.sidebar.header("📋 Form Parameter Calon Nasabah")

limit_bal = st.sidebar.number_input("Batas Kredit / Saldo Limit (LIMIT_BAL in NT$)", min_value=10000, max_value=1000000, value=50000, step=10000)
sex = st.sidebar.selectbox("Jenis Kelamin (SEX)", options=[1, 2], format_func=lambda x: "1 - Laki-laki" if x == 1 else "2 - Perempuan")
education = st.sidebar.selectbox("Tingkat Pendidikan (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "1 - Pascasarjana (S2/S3)", 2: "2 - Universitas (S1)", 3: "3 - SMA", 4: "4 - Lainnya"}[x])
marriage = st.sidebar.selectbox("Status Pernikahan (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "1 - Menikah", 2: "2 - Lajang", 3: "3 - Lainnya"}[x])
age = st.sidebar.slider("Usia Nasabah (AGE)", min_value=18, max_value=80, value=30)

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Keterlambatan Bayar (6 Bulan Terakhir)")
pay_status_options = {-2: "Tidak ada tagihan", -1: "Bayar tepat waktu", 0: "Gunakan revolving credit", 1: "Telat 1 bulan", 2: "Telat 2 bulan", 3: "Telat 3 bulan atau lebih"}
pay_0 = st.sidebar.selectbox("Status Bayar Sept 2005 (PAY_0)", options=list(pay_status_options.keys()), index=3, format_func=lambda x: f"{x} ({pay_status_options[x]})")
pay_2 = st.sidebar.selectbox("Status Bayar Agt 2005 (PAY_2)", options=list(pay_status_options.keys()), index=2, format_func=lambda x: f"{x} ({pay_status_options[x]})")
pay_3 = st.sidebar.selectbox("Status Bayar Jul 2005 (PAY_3)", options=list(pay_status_options.keys()), index=2, format_func=lambda x: f"{x} ({pay_status_options[x]})")
pay_4 = st.sidebar.selectbox("Status Bayar Jun 2005 (PAY_4)", options=list(pay_status_options.keys()), index=2, format_func=lambda x: f"{x} ({pay_status_options[x]})")
pay_5 = st.sidebar.selectbox("Status Bayar Mei 2005 (PAY_5)", options=list(pay_status_options.keys()), index=2, format_func=lambda x: f"{x} ({pay_status_options[x]})")
pay_6 = st.sidebar.selectbox("Status Bayar Apr 2005 (PAY_6)", options=list(pay_status_options.keys()), index=2, format_func=lambda x: f"{x} ({pay_status_options[x]})")

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Nominal Tagihan (BILL_AMT)")
bill_amt1 = st.sidebar.number_input("Tagihan Sept 2005 (BILL_AMT1)", value=30000)
bill_amt2 = st.sidebar.number_input("Tagihan Agt 2005 (BILL_AMT2)", value=25000)
bill_amt3 = st.sidebar.number_input("Tagihan Jul 2005 (BILL_AMT3)", value=20000)
bill_amt4 = st.sidebar.number_input("Tagihan Jun 2005 (BILL_AMT4)", value=15000)
bill_amt5 = st.sidebar.number_input("Tagihan Mei 2005 (BILL_AMT5)", value=10000)
bill_amt6 = st.sidebar.number_input("Tagihan Apr 2005 (BILL_AMT6)", value=5000)

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Pembayaran Sebelumnya (PAY_AMT)")
pay_amt1 = st.sidebar.number_input("Pembayaran Sept 2005 (PAY_AMT1)", value=2000)
pay_amt2 = st.sidebar.number_input("Pembayaran Agt 2005 (PAY_AMT2)", value=2000)
pay_amt3 = st.sidebar.number_input("Pembayaran Jul 2005 (PAY_AMT3)", value=1500)
pay_amt4 = st.sidebar.number_input("Pembayaran Jun 2005 (PAY_AMT4)", value=1000)
pay_amt5 = st.sidebar.number_input("Pembayaran Mei 2005 (PAY_AMT5)", value=1000)
pay_amt6 = st.sidebar.number_input("Pembayaran Apr 2005 (PAY_AMT6)", value=1000)

# Tombol Eksekusi
btn_predict = st.sidebar.button("🔍 Analisis Kelayakan Kredit")

# 3. Proses Inferensi dan Visualisasi
if btn_predict:
    # Membentuk 23 fitur persis sesuai urutan data latih
    feature_dict = {
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'PAY_4': pay_4,
        'PAY_5': pay_5,
        'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1,
        'BILL_AMT2': bill_amt2,
        'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4,
        'BILL_AMT5': bill_amt5,
        'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1,
        'PAY_AMT2': pay_amt2,
        'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4,
        'PAY_AMT5': pay_amt5,
        'PAY_AMT6': pay_amt6
    }
    
    input_df = pd.DataFrame([feature_dict])
    
    # Transformasi Skala pada Kolom Numerik
    num_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    input_df_scaled = input_df.copy()
    input_df_scaled[num_cols] = scaler.transform(input_df[num_cols])
    
    # Inferensi Prediksi
    prob_default = model.predict_proba(input_df_scaled)[0][1]
    prediction = int(prob_default >= 0.5)
    
    st.subheader("📊 Hasil Penilaian Risiko")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("🚨 **Rekomendasi: PENGAJUAN DITOLAK**")
            st.write("Nasabah memiliki profil risiko tinggi mengalami gagal bayar (*Default*).")
        else:
            st.success("✅ **Rekomendasi: PENGAJUAN DISETUJUI**")
            st.write("Nasabah dinilai memenuhi kriteria kelayakan pembayaran (*Non-Default*).")
            
    with col2:
        st.metric(
            label="Estimasi Probabilitas Gagal Bayar",
            value=f"{prob_default * 100:.2f}%",
            delta=f"{'+Tinggi' if prob_default >= 0.5 else '-Rendah'}"
        )
        
    st.markdown("---")
    st.subheader("🔍 Keterjelasan Keputusan Model (Explainable AI - SHAP)")
    st.write("Grafik berikut membedah faktor finansial mana yang mendorong kenaikan risiko (warna merah) atau penurunan risiko (warna biru) secara spesifik pada nasabah ini:")
    
    # Kalkulasi Nilai SHAP Lokal
    shap_vals = explainer(input_df_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
    st.pyplot(fig)
    plt.clf()
else:
    st.info("👈 Silakan lengkapi parameter calon debitur di bilah sisi (*sidebar*) dan tekan tombol **'Analisis Kelayakan Kredit'**.")
