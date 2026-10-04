import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Assessment & Explainable AI",
    page_icon="💳",
    layout="wide"
)

# Custom Styling CSS (Modern Card & Badge UI)
st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
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
        'limit_bal': 200000, 'sex': 2, 'education': 1, 'marriage': 2, 'age': 35,
        'pay_0': -1, 'pay_2': -1, 'pay_3': -1, 'pay_4': -1, 'pay_5': -1, 'pay_6': -1,
        'bill_amt1': 25000, 'bill_amt2': 22000, 'bill_amt3': 18000,
        'bill_amt4': 15000, 'bill_amt5': 12000, 'bill_amt6': 8000,
        'pay_amt1': 25000, 'pay_amt2': 22000, 'pay_amt3': 18000,
        'pay_amt4': 15000, 'pay_amt5': 12000, 'pay_amt6': 8000
    },
    "macet": {
        'limit_bal': 30000, 'sex': 1, 'education': 3, 'marriage': 1, 'age': 26,
        'pay_0': 2, 'pay_2': 2, 'pay_3': 1, 'pay_4': 0, 'pay_5': 0, 'pay_6': 0,
        'bill_amt1': 29500, 'bill_amt2': 29000, 'bill_amt3': 28000,
        'bill_amt4': 25000, 'bill_amt5': 22000, 'bill_amt6': 20000,
        'pay_amt1': 0, 'pay_amt2': 1000, 'pay_amt3': 1000,
        'pay_amt4': 1000, 'pay_amt5': 500, 'pay_amt6': 500
    },
    "borderline": {
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

# Inisialisasi session state awal jika belum ada
if 'limit_bal' not in st.session_state:
    apply_preset('lancar')

# -----------------------------------------------------------------------------
# 3. SIDEBAR: PRESET CONTROLS & FORM INPUT
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ Uji Coba Cepat (Preset Profiles)")
st.sidebar.caption("Klik untuk mengisi data sampel secara instan, atau ketik manual di bawah:")
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

limit_bal = st.sidebar.number_input("Credit Limit (LIMIT_BAL in NT$)", min_value=10000, max_value=1000000, step=10000, key='limit_bal')
sex = st.sidebar.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female", key='sex')
education = st.sidebar.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: {1: "Graduate School (S2/S3)", 2: "University (S1)", 3: "High School", 4: "Others"}[x], key='education')
marriage = st.sidebar.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x], key='marriage')
age = st.sidebar.slider("Age", min_value=18, max_value=80, key='age')

st.sidebar.markdown("---")
st.sidebar.subheader("Payment History (6 Months)")
pay_status_options = {-2: "No consumption", -1: "Paid duly (On time)", 0: "Revolving credit", 1: "Delay 1 month", 2: "Delay 2 months", 3: "Delay 3+ months"}

pay_0 = st.sidebar.selectbox("Sept 2005 (PAY_0)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_0')
pay_2 = st.sidebar.selectbox("Aug 2005 (PAY_2)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_2')
pay_3 = st.sidebar.selectbox("Jul 2005 (PAY_3)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_3')
pay_4 = st.sidebar.selectbox("Jun 2005 (PAY_4)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_4')
pay_5 = st.sidebar.selectbox("May 2005 (PAY_5)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_5')
pay_6 = st.sidebar.selectbox("Apr 2005 (PAY_6)", options=list(pay_status_options.keys()), format_func=lambda x: f"{x} ({pay_status_options[x]})", key='pay_6')

st.sidebar.markdown("---")
st.sidebar.subheader("Bill Statements (NT$)")
bill_amt1 = st.sidebar.number_input("Bill Sept 2005 (BILL_AMT1)", key='bill_amt1')
bill_amt2 = st.sidebar.number_input("Bill Aug 2005 (BILL_AMT2)", key='bill_amt2')
bill_amt3 = st.sidebar.number_input("Bill Jul 2005 (BILL_AMT3)", key='bill_amt3')
bill_amt4 = st.sidebar.number_input("Bill Jun 2005 (BILL_AMT4)", key='bill_amt4')
bill_amt5 = st.sidebar.number_input("Bill May 2005 (BILL_AMT5)", key='bill_amt5')
bill_amt6 = st.sidebar.number_input("Bill Apr 2005 (BILL_AMT6)", key='bill_amt6')

st.sidebar.markdown("---")
st.sidebar.subheader("Previous Payments (NT$)")
pay_amt1 = st.sidebar.number_input("Paid Sept 2005 (PAY_AMT1)", key='pay_amt1')
pay_amt2 = st.sidebar.number_input("Paid Aug 2005 (PAY_AMT2)", key='pay_amt2')
pay_amt3 = st.sidebar.number_input("Paid Jul 2005 (PAY_AMT3)", key='pay_amt3')
pay_amt4 = st.sidebar.number_input("Paid Jun 2005 (PAY_AMT4)", key='pay_amt4')
pay_amt5 = st.sidebar.number_input("Paid May 2005 (PAY_AMT5)", key='pay_amt5')
pay_amt6 = st.sidebar.number_input("Paid Apr 2005 (PAY_AMT6)", key='pay_amt6')

btn_predict = st.sidebar.button("🔍 Assess Credit Risk", use_container_width=True)

# -----------------------------------------------------------------------------
# 4. KONTEN UTAMA: DASHBOARD INFERENSI & XAI
# -----------------------------------------------------------------------------
st.title("🏦 Credit Risk Assessor")
st.markdown("**Powered by Extreme Gradient Boosting (XGBoost) & Explainable AI (SHAP)**")
st.caption("Pilih profil sampel di atas bilah samping atau sesuaikan data secara manual, lalu klik tombol **Assess Credit Risk**.")

if btn_predict:
    # Mengumpulkan 23 fitur
    feature_dict = {
        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6
    }
    
    input_df = pd.DataFrame([feature_dict])
    
    # Scaling numerik via RobustScaler
    num_cols = ['LIMIT_BAL'] + [f'BILL_AMT{i}' for i in range(1, 7)] + [f'PAY_AMT{i}' for i in range(1, 7)]
    input_df_scaled = input_df.copy()
    input_df_scaled[num_cols] = scaler.transform(input_df[num_cols])
    
    # Inferensi Probabilitas
    prob_default = model.predict_proba(input_df_scaled)[0][1]
    
    # Kalkulasi Metrik Bisnis Peminjam
    utilization_rate = max(0, min(100, int((bill_amt1 / max(1, limit_bal)) * 100)))
    total_late_months = sum([1 for p in [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6] if p > 0])
    payment_ratio = max(0, min(100, int((pay_amt1 / max(1, bill_amt1)) * 100))) if bill_amt1 > 0 else 100
    
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
            recom_text = "Rekomendasi: Tinjau Manual (Profil Berisiko Sedang)"
        else:
            badge_html = '<div class="badge-low">✅ LOW RISK</div>'
            prob_color = '#16a34a'
            recom_text = "Rekomendasi: Pengajuan Disetujui (Profil Sehat)"
            
        st.markdown(f"""
        <div class="metric-container">
            {badge_html}
            <div class="prob-number" style="color: {prob_color};">{prob_default * 100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.95rem; font-weight: 500;">Estimated probability of default within period</div>
            <div style="margin-top: 10px; font-weight: 600; font-size: 0.9rem; color: #334155;">{recom_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_gauge:
        st.subheader("📊 Risk Gauge")
        st.progress(float(prob_default))
        st.caption(f"Posisi Risiko: **{prob_default * 100:.1f}%** | Ambang Batas Default (*Cut-off*): **50.0%**")
        
        if prob_default >= 0.5:
            st.error("⚠️ **Peringatan Kredit:** Probabilitas risiko melampaui toleransi perbankan (>50%). Diperlukan penjamin atau agunan tambahan.")
        elif prob_default >= 0.35:
            st.warning("⚡ **Perhatian:** Profil debitur berada di zona netral/waspada. Rekomendasikan limit awal yang lebih konservatif.")
        else:
            st.success("🛡️ **Aman:** Skor risiko berada jauh di bawah ambang batas kritis. Nasabah memiliki kapasitas pengembalian kredit prima.")
            
    st.markdown("---")
    
    # --- SECTION 2: WHY THIS SCORE? (INSIGHTS) ---
    st.subheader("🔍 Why This Score?")
    
    # Analisis Credit Utilization
    if utilization_rate <= 40:
        st.markdown(f"""
        <div class="why-box-good">
            <b>🟢 Healthy credit utilization</b><br>
            Hanya <b>{utilization_rate}%</b> dari plafon limit kredit yang digunakan — sinyal kendali likuiditas yang disiplin.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="why-box-bad">
            <b>🔴 High credit utilization</b><br>
            Penggunaan saldo mencapai <b>{utilization_rate}%</b> dari batas limit — ketergantungan utang yang tinggi meningkatkan beban bunga.
        </div>
        """, unsafe_allow_html=True)
        
    # Analisis Riwayat Keterlambatan
    if total_late_months == 0:
        st.markdown("""
        <div class="why-box-good">
            <b>🟢 Clean payment history</b><br>
            Tidak ada riwayat keterlambatan bayar yang tercatat selama 6 bulan terakhir — indikator kelayakan pembayaran terbaik.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="why-box-bad">
            <b>🔴 Payment delay detected</b><br>
            Tercatat keterlambatan bayar sebanyak <b>{total_late_months} bulan</b> pada 6 bulan terakhir — pendorong dominan naiknya risiko.
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

    # --- SECTION 4: EXPLAINABLE AI WATERFALL PLOT ---
    st.subheader("📈 Local Feature Attribution (Explainable AI - SHAP)")
    st.write("Dekomposisi atribusi nilai Shapley: Balok **merah (+)** mendorong kenaikan risiko, balok **biru (-)** meredam risiko calon debitur ini.")
    
    shap_vals = explainer(input_df_scaled)
    
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    # --- SECTION 5: DOWNLOAD REPORT BUTTON ---
    st.markdown("---")
    report_data = input_df.copy()
    report_data['Default_Probability'] = f"{prob_default * 100:.2f}%"
    report_data['Recommendation'] = "REJECTED" if prob_default >= 0.5 else "APPROVED"
    csv_report = report_data.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Assessment Report (CSV)",
        data=csv_report,
        file_name=f"credit_assessment_report_age_{age}.csv",
        mime="text/csv",
        help="Unduh berkas ringkasan evaluasi untuk arsip persetujuan kredit"
    )
    
    st.caption("⚠️ **Disclaimer:** This tool is for educational, research, and thesis portfolio demonstration purposes only. Not intended as an actual financial credit decision-making system.")

else:
    st.info("👈 Silakan pilih profil uji coba cepat di bilah samping (*sidebar*) atau sesuaikan data secara manual, lalu tekan tombol **'Assess Credit Risk'**.")
