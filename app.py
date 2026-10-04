import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment & Explainable AI",
    page_icon="💳",
    layout="wide"
)

# Kustomisasi CSS Antarmuka
st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    .badge-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-borderline {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    .prob-number {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .why-box-good {
        background-color: #f0fdf4;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        border-left: 5px solid #16a34a;
    }
    .why-box-bad {
        background-color: #fef2f2;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
        border-left: 5px solid #dc2626;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. MEMUAT MODEL, SCALER, & SHAP EXPLAINER
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('xgboost_credit_model.pkl')
    scaler = joblib.load('scaler_credit.pkl')
    explainer = shap.TreeExplainer(model)
    return model, scaler, explainer

model, scaler, explainer = load_assets()

# -----------------------------------------------------------------------------
# 2. DEFINISI PRESET PROFILES (DEMO CEPAT)
# -----------------------------------------------------------------------------
presets = {
    "lancar": {
        'customer_name': 'Budi Santoso',
        'limit_bal': 200000, 'sex': 2, 'education': 1, 'marriage': 2, 'age': 35,
        'pay_0': -1, 'pay_2': -1, 'pay_3': -1, 'pay_4': -1, 'pay_5': -1, 'pay_6': -1,
        'bill_amt1': 25000, 'bill_amt2': 22000, 'bill_amt3': 18000,
        'bill_amt4': 15000, 'bill_amt5': 12000, 'bill_amt6': 8000,
        'pay_amt1': 25000, 'pay_amt2': 22000, 'pay_amt3': 18000,
        'pay_amt4': 15000, 'pay_amt5': 12000, 'pay_amt6': 8000
    },
    "macet": {
        'customer_name': 'Rian Hidayat',
        'limit_bal': 30000, 'sex': 1, 'education': 3, 'marriage': 1, 'age': 26,
        'pay_0': 2, 'pay_2': 2, 'pay_3': 1, 'pay_4': 0, 'pay_5': 0, 'pay_6': 0,
        'bill_amt1': 29500, 'bill_amt2': 29000, 'bill_amt3': 28000,
        'bill_amt4': 25000, 'bill_amt5': 22000, 'bill_amt6': 20000,
        'pay_amt1': 0, 'pay_amt2': 1000, 'pay_amt3': 1000,
        'pay_amt4': 1000, 'pay_amt5': 500, 'pay_amt6': 500
    },
    "borderline": {
        'customer_name': 'Siti Rahmawati',
        'limit_bal': 80000, 'sex': 1, 'education': 2, 'marriage': 2, 'age': 29,
        'pay_0': 1, 'pay_2': 0, 'pay_3': 0, 'pay_4': -1, 'pay_5': 0, 'pay_6': 0,
        'bill_amt1': 48000, 'bill_amt2': 42000, 'bill_amt3': 38000,
        'bill_amt4': 30000, 'bill_amt5': 25000, 'bill_amt6': 20000,
        'pay_amt1': 2000, 'pay_amt2': 2000, 'pay_amt3': 2000,
        'pay_amt4': 2000, 'pay_amt5': 1500, 'pay_amt6': 1500
    }
}

def apply_preset(preset_key):
    p = presets[preset_key]
    for k, v in p.items():
        st.session_state[k] = v

if 'limit_bal' not in st.session_state:
    apply_preset('lancar')

# -----------------------------------------------------------------------------
# 3. SIDEBAR: PRESET CONTROLS & FORM INPUT
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ Uji Coba Cepat (Preset Profiles)")
st.sidebar.caption("Klik tombol untuk mengisi otomatis, atau edit nilai form di bawah:")
col_p1, col_p2, col_p3 = st.sidebar.columns(3)

with col_p1:
    if st.button("🟢 Lancar", help="Memuat sampel nasabah ideal/low-risk", use_container_width=True):
        apply_preset('lancar')
with col_p2:
    if st.button("🔴 Macet", help="Memuat sampel nasabah gagal bayar/high-risk", use_container_width=True):
        apply_preset('macet')
with col_p3:
    if st.button("🟡 Ragu", help="Memuat sampel nasabah abu-abu/borderline", use_container_width=True):
        apply_preset('borderline')

st.sidebar.markdown("---")
st.sidebar.header("📋 Borrower Information")

# Input Nama Nasabah (Metadata UI & Laporan)
customer_name = st.sidebar.text_input("Nama Lengkap Calon Nasabah", key='customer_name')

limit_bal = st.sidebar.number_input("Batas Kredit (LIMIT_BAL in NT$)", min_value=10000, max_value=1000000, step=10000, key='limit_bal')
sex = st.sidebar.selectbox("Jenis Kelamin", options=[1, 2], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan", key='sex')
education = st.sidebar.selectbox("Tingkat Pendidikan", options=[1, 2, 3, 4], format_func=lambda x: {1: "Pascasarjana (S2/S3)", 2: "Universitas (S1)", 3: "SMA", 4: "Lainnya"}[x], key='education')
marriage = st.sidebar.selectbox("Status Pernikahan", options=[1, 2, 3], format_func=lambda x: {1: "Menikah", 2: "Lajang", 3: "Lainnya"}[x], key='marriage')
age = st.sidebar.slider("Usia", min_value=18, max_value=80, key='age')

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Keterlambatan Bayar (6 Bulan)")
pay_status_options = {-2: "Tidak ada tagihan", -1: "Tepat waktu (Paid duly)", 0: "Revolving credit", 1: "Telat 1 bulan", 2: "Telat 2 bulan", 3: "Telat 3+ bulan"}

pay_0 = st.sidebar.selectbox("Sept 2005 (PAY_0)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_0')
pay_2 = st.sidebar.selectbox("Agt 2005 (PAY_2)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_2')
pay_3 = st.sidebar.selectbox("Jul 2005 (PAY_3)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_3')
pay_4 = st.sidebar.selectbox("Jun 2005 (PAY_4)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_4')
pay_5 = st.sidebar.selectbox("Mei 2005 (PAY_5)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_5')
pay_6 = st.sidebar.selectbox("Apr 2005 (PAY_6)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_6')

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Tagihan Bulanan (NT$)")
bill_amt1 = st.sidebar.number_input("Tagihan Sept 2005 (BILL_AMT1)", key='bill_amt1')
bill_amt2 = st.sidebar.number_input("Tagihan Agt 2005 (BILL_AMT2)", key='bill_amt2')
bill_amt3 = st.sidebar.number_input("Tagihan Jul 2005 (BILL_AMT3)", key='bill_amt3')
bill_amt4 = st.sidebar.number_input("Tagihan Jun 2005 (BILL_AMT4)", key='bill_amt4')
bill_amt5 = st.sidebar.number_input("Tagihan Mei 2005 (BILL_AMT5)", key='bill_amt5')
bill_amt6 = st.sidebar.number_input("Tagihan Apr 2005 (BILL_AMT6)", key='bill_amt6')

st.sidebar.markdown("---")
st.sidebar.subheader("Riwayat Pembayaran Sebelumnya (NT$)")
pay_amt1 = st.sidebar.number_input("Bayar Sept 2005 (PAY_AMT1)", key='pay_amt1')
pay_amt2 = st.sidebar.number_input("Bayar Agt 2005 (PAY_AMT2)", key='pay_amt2')
pay_amt3 = st.sidebar.number_input("Bayar Jul 2005 (PAY_AMT3)", key='pay_amt3')
pay_amt4 = st.sidebar.number_input("Bayar Jun 2005 (PAY_AMT4)", key='pay_amt4')
pay_amt5 = st.sidebar.number_input("Bayar Mei 2005 (PAY_AMT5)", key='pay_amt5')
pay_amt6 = st.sidebar.number_input("Bayar Apr 2005 (PAY_AMT6)", key='pay_amt6')

btn_predict = st.sidebar.button("🔍 Assess Credit Risk", use_container_width=True)

# -----------------------------------------------------------------------------
# FUNGSI MEMBUAT PLOTLY RISK GAUGE METER
# -----------------------------------------------------------------------------
def plot_risk_gauge(probability):
    prob_pct = probability * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={'suffix': "%", 'font': {'size': 38, 'color': '#0f172a'}},
        title={'text': "<b>Risk Gauge</b><br><span style='font-size:0.8em;color:gray'>Estimated Default Probability</span>", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': "#0f172a", 'thickness': 0.22},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#cbd5e1",
            'steps': [
                {'range': [0, 20], 'color': "#86efac"},   # Low (Hijau)
                {'range': [20, 50], 'color': "#fde047"},  # Medium (Kuning)
                {'range': [50, 100], 'color': "#fca5a5"}  # High (Merah)
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 4},
                'thickness': 0.75,
                'value': 50.0
            }
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=35, b=20))
    return fig

# -----------------------------------------------------------------------------
# 4. KONTEN UTAMA: DASHBOARD INFERENSI & VISUALISASI
# -----------------------------------------------------------------------------
st.title("🏦 Credit Risk Assessor")
st.markdown("**Decision Support System Berbasis Extreme Gradient Boosting (XGBoost) & Explainable AI (SHAP)**")
st.caption("Gunakan profil uji coba cepat di bilah samping atau sesuaikan data secara manual, lalu klik tombol **Assess Credit Risk**.")

if btn_predict:
    # 23 Fitur Murni untuk Inferensi Model (Nama nasabah tidak diikutkan agar model objektif)
    feature_dict = {
        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
    }
    
    input_df = pd.DataFrame([feature_dict])
    
    # Penskalaan Fitur Numerik menggunakan RobustScaler
    num_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    input_df_scaled = input_df.copy()
    input_df_scaled[num_cols] = scaler.transform(input_df[num_cols])
    
    # Prediksi Probabilitas
    prob_default = model.predict_proba(input_df_scaled)[0][1]
    
    # Kalkulasi Metrik Finansial
    utilization_rate = max(0, min(100, int((bill_amt1 / max(1, limit_bal)) * 100)))
    total_late_months = sum([1 for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6] if p > 0])
    payment_ratio = max(0, min(100, int((pay_amt1 / max(1, bill_amt1)) * 100))) if bill_amt1 > 0 else 100
    
    st.subheader(f"📊 Hasil Penilaian Kredit: {customer_name if customer_name else 'Calon Nasabah'}")
    
    # --- SECTION 1: BADGE STATUS & RISK GAUGE ---
    col_card, col_gauge = st.columns([1, 1.2])
    
    with col_card:
        if prob_default >= 0.5:
            badge_html = '<div class="badge-high">🚨 HIGH RISK</div>'
            prob_color = '#dc2626'
            recom_text = "Rekomendasi: Pengajuan Ditolak (Potensi Default Tinggi)"
        elif prob_default >= 0.35:
            badge_html = '<div class="badge-borderline">⚠️ BORDERLINE RISK</div>'
            prob_color = '#d97706'
            recom_text = "Rekomendasi: Tinjau Ulang Manual (Risiko Moderat)"
        else:
            badge_html = '<div class="badge-low">✅ LOW RISK</div>'
            prob_color = '#16a34a'
            recom_text = "Rekomendasi: Pengajuan Disetujui (Profil Likuiditas Baik)"
            
        st.markdown(f"""
        <div class="metric-container">
            {badge_html}
            <div class="prob-number" style="color: {prob_color};">{prob_default * 100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.95rem; font-weight: 500;">Estimated probability of default</div>
            <div style="margin-top: 10px; font-weight: 600; font-size: 0.9rem; color: #334155;">{recom_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_gauge:
        st.plotly_chart(plot_risk_gauge(prob_default), use_container_width=True)
        
    st.markdown("---")
    
    # --- SECTION 2: WHY THIS SCORE? (EXPLANATION INSIGHTS) ---
    st.subheader("🔍 Why This Score?")
    
    # Analisis Credit Utilization
    if utilization_rate <= 40:
        st.markdown(f"""
        <div class="why-box-good">
            <b>🟢 Healthy credit utilization</b><br>
            Hanya <b>{utilization_rate}%</b> dari plafon batas kredit yang terpakai — sinyal disiplin finansial yang sehat.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="why-box-bad">
            <b>🔴 High credit utilization</b><br>
            Pemakaian saldo mencapai <b>{utilization_rate}%</b> dari plafon kredit — rasio utang tinggi meningkatkan risiko likuiditas.
        </div>
        """, unsafe_allow_html=True)
        
    # Analisis Riwayat Keterlambatan
    if total_late_months == 0:
        st.markdown("""
        <div class="why-box-good">
            <b>🟢 Clean payment history</b><br>
            Tidak ada riwayat keterlambatan bayar yang tercatat selama 6 bulan terakhir — indikator likuiditas yang sangat positif.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="why-box-bad">
            <b>🔴 Payment delay detected</b><br>
            Tercatat penundaan pembayaran sebanyak <b>{total_late_months} bulan</b> pada 6 bulan terakhir — pendorong dominan naiknya risiko.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- SECTION 3: BORROWER SUMMARY (METRICS) ---
    st.subheader("📋 Borrower Summary")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Credit Utilization", f"{utilization_rate}%")
    kpi2.metric("Payment / Bill Ratio", f"{payment_ratio}%")
    kpi3.metric("Total Late Months", f"{total_late_months} mos")
    kpi4.metric("Age", f"{age} yrs")

    st.markdown("---")

    # --- SECTION 4: EXPLAINABLE AI TABS (LOCAL WATERFALL VS GLOBAL BAR CHART) ---
    st.subheader("📈 Model Interpretability (Explainable AI)")
    tab_local, tab_global = st.tabs(["🔍 Individual Explanation (SHAP Waterfall)", "🌐 What Matters Most (Global Feature Importance)"])
    
    with tab_local:
        st.write("Dekomposisi atribusi nilai Shapley per individu: Balok **merah (+)** mendorong kenaikan risiko, balok **biru (-)** meredam risiko calon debitur ini:")
        shap_vals = explainer(input_df_scaled)
        
        plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

    with tab_global:
        st.write("Peringkat 10 faktor penentu paling dominan pada portofolio model XGBoost secara keseluruhan:")
        importance_df = pd.DataFrame({
            'Feature': model.feature_names_in_,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig_global, ax_glob = plt.subplots(figsize=(10, 5))
        ax_glob.barh(importance_df['Feature'], importance_df['Importance'], color='#2563eb')
        ax_glob.set_xlabel('Feature Importance Weight (Gain)')
        plt.tight_layout()
        st.pyplot(fig_global)
        plt.clf()

    # --- SECTION 5: DOWNLOAD REPORT BUTTON ---
    st.markdown("---")
    report_data = input_df.copy()
    report_data.insert(0, 'Nama_Nasabah', customer_name if customer_name else "Calon Nasabah")
    report_data['Default_Probability'] = f"{prob_default * 100:.2f}%"
    report_data['Recommendation'] = "REJECTED" if prob_default >= 0.5 else "APPROVED"
    csv_report = report_data.to_csv(index=False).encode('utf-8')
    
    file_clean_name = (customer_name if customer_name else "Nasabah").replace(" ", "_")
    st.download_button(
        label="📥 Download Assessment Report (CSV)",
        data=csv_report,
        file_name=f"Laporan_Kredit_{file_clean_name}.csv",
        mime="text/csv",
        help="Unduh berkas ringkasan evaluasi untuk arsip persetujuan kredit"
    )
    
    st.caption("⚠️ **Disclaimer:** This tool is for educational, research, and thesis portfolio demonstration purposes only. Not intended as an actual financial credit decision-making system.")

else:
    st.info("👈 Silakan pilih profil uji coba cepat di bilah samping (*sidebar*) atau sesuaikan data secara manual, lalu tekan tombol **'Assess Credit Risk'**.")
