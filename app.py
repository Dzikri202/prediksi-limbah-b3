import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ─── KONFIGURASI HALAMAN ─────────────────────────────────────
st.set_page_config(
    page_title="Sistem Prediksi Volume Limbah B3 - PT Surya Agrolika Reksa",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS KUSTOM VIA ST.MARKDOWN ──────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background-color: #eef2f7;
        color: #1a2540;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0a2351 0%, #1a4080 100%);
        border-right: none;
        box-shadow: 4px 0 24px rgba(10,35,81,0.18);
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #d4e2f8 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.12) !important;
    }

    .block-container {
        padding-top: 4rem !important;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    header[data-testid="stHeader"] {
        background: #eef2f7;
        border-bottom: 1px solid #d5e3f0;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #d5e3f0;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 14px;
        box-shadow: 0 2px 10px rgba(10,35,81,0.07);
        transition: transform 0.15s, box-shadow 0.15s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(10,35,81,0.12);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #7a92b8;
        margin-bottom: 8px;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 600;
        color: #0a2351;
        line-height: 1.2;
    }

    .metric-sub {
        font-size: 12px;
        color: #9aaece;
        margin-top: 6px;
        font-weight: 500;
    }

    .metric-good  { border-left: 5px solid #0a9e70; }
    .metric-warn  { border-left: 5px solid #e09b0a; }
    .metric-info  { border-left: 5px solid #1a6fc4; }
    .metric-muted { border-left: 5px solid #9aaece; }

    .section-header {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #7a92b8;
        border-bottom: 2px solid #d5e3f0;
        padding-bottom: 10px;
        margin: 36px 0 18px 0;
    }

    .page-title {
        font-size: 30px;
        font-weight: 800;
        color: #0a2351;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .page-sub {
        font-size: 15px;
        color: #7a92b8;
        margin-top: 6px;
        margin-bottom: 28px;
        font-weight: 400;
    }

    .badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        letter-spacing: 0.06em;
    }

    .badge-green  { background: #e6f9f3; color: #0a9e70; border: 1.5px solid #0a9e70; }
    .badge-yellow { background: #fef9e6; color: #b77f0a; border: 1.5px solid #e09b0a; }
    .badge-red    { background: #fdecea; color: #c0392b; border: 1.5px solid #e74c3c; }

    .info-box {
        background: #e8f1fc;
        border: 1px solid #b3cef5;
        border-left: 5px solid #1a6fc4;
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 14px;
        color: #2c4a7c;
        margin: 12px 0;
        font-weight: 500;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #d5e3f0;
        border-radius: 10px;
        overflow: hidden;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1a4080 0%, #1a6fc4 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 14px;
        padding: 10px 24px;
        width: 100%;
        transition: opacity 0.2s, transform 0.15s;
        box-shadow: 0 2px 8px rgba(26,64,128,0.3);
    }

    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #7a92b8;
    }

    hr { border-color: #d5e3f0; }
</style>
""", unsafe_allow_html=True)

# ─── FUNGSI UTAMA LOAD MODEL (.JOBLIB) ───────────────────────
@st.cache_resource
def muat_model_dan_encoder():
    model     = joblib.load('model_rf_limbah.joblib')
    le_jenis  = joblib.load('encoder_jenis_limbah.joblib')
    le_sumber = joblib.load('encoder_sumber.joblib')
    
    # Disesuaikan pas dengan nama kolom saat ditraining pada Colab baru Anda
    fitur = ['Bulan', 'Tahun', 'Kuartal', 'Hari_dalam_Bulan',
             'Jenis_Encoded', 'Sumber_Encoded', 'Sisa_di_TPS_Ton']
    
    return model, le_jenis, le_sumber, fitur

def set_plot_style(fig, ax_list):
    fig.patch.set_facecolor('#ffffff')
    for ax in ax_list:
        ax.set_facecolor('#f8fafd')
        ax.tick_params(colors='#7a92b8', labelsize=9)
        ax.xaxis.label.set_color('#4a6490')
        ax.yaxis.label.set_color('#4a6490')
        ax.title.set_color('#0a2351')
        for spine in ax.spines.values():
            spine.set_edgecolor('#d5e3f0')
    return fig

@st.cache_data
def muat_data(file):
    df = pd.read_csv(file, parse_dates=['Tanggal'])
    return df

# ─── SIDEBAR NAVIGASI ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 4px 0;">
        <div style="font-size:13px; font-weight:800; letter-spacing:0.05em; color:#ffffff; margin-bottom:4px;">
            SISTEM PREDIKSI LIMBAH B3
        </div>
        <div style="font-size:12px; font-weight:500; color:#a0bcdf;">
            PT Surya Agrolika Reksa
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    menu = st.radio(
        "Navigasi",
        ["Beranda", "Eksplorasi Data", "Evaluasi Model",
         "Forecasting", "Tentang Sistem"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#484f58;">Prediksi Volume Limbah B3<br>'
        'Menggunakan Model Joblib Random Forest</div>',
        unsafe_allow_html=True
    )

# ─── BERANDA ─────────────────────────────────────────────────
if menu == "Beranda":
    st.markdown('<div class="page-title">Sistem Prediksi Volume Limbah B3</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">PT Surya Agrolika Reksa — Upload data untuk memulai analisis dan prediksi</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload File Data (CSV)",
        type=["csv"],
        help="Upload file data_limbah_b3_augmented.csv"
    )

    if uploaded:
        df = muat_data(uploaded)
        st.session_state['df'] = df

        st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-info">
                <div class="metric-label">Total Data</div>
                <div class="metric-value">{len(df):,}</div>
                <div class="metric-sub">baris</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-info">
                <div class="metric-label">Data Asli</div>
                <div class="metric-value">{len(df[df['Sumber_Data']=='Asli']):,}</div>
                <div class="metric-sub">dari PT Sawit</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-muted">
                <div class="metric-label">Data Augmentasi</div>
                <div class="metric-value">{len(df[df['Sumber_Data']=='Augmentasi']):,}</div>
                <div class="metric-sub">data sintetis</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card metric-info">
                <div class="metric-label">Jenis Limbah B3</div>
                <div class="metric-value">{df['Jenis_Limbah_B3'].nunique()}</div>
                <div class="metric-sub">kategori</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Periode Data</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            Rentang data: <strong>{df['Tanggal'].min().strftime('%d %B %Y')}</strong>
            sampai <strong>{df['Tanggal'].max().strftime('%d %B %Y')}</strong>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Pratinjau Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    else:
        st.markdown("""
        <div class="info-box">
            Belum ada data yang diupload. Silakan upload file
            <strong>data_limbah_b3_augmented.csv</strong> untuk memulai.
        </div>""", unsafe_allow_html=True)

# ─── EKSPLORASI DATA ─────────────────────────────────────────
elif menu == "Eksplorasi Data":
    st.markdown('<div class="page-title">Eksplorasi Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Analisis distribusi dan pola volume limbah B3</div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.markdown('<div class="info-box">Upload data terlebih dahulu di halaman Beranda.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state['df']

        st.markdown('<div class="section-header">Distribusi Jenis Limbah B3</div>', unsafe_allow_html=True)
        dist = df.groupby('Jenis_Limbah_B3')['Volume_Masuk_Ton'].agg(['sum','mean','count']).round(4)
        dist.columns = ['Total Volume (Ton)', 'Rata-rata (Ton)', 'Jumlah Transaksi']
        st.dataframe(dist, use_container_width=True)

        st.markdown('<div class="section-header">Tren Volume per Jenis Limbah</div>', unsafe_allow_html=True)
        jenis_pilih = st.selectbox("Pilih Jenis Limbah", options=df['Jenis_Limbah_B3'].unique())

        subset = df[df['Jenis_Limbah_B3'] == jenis_pilih].sort_values('Tanggal')
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(subset['Tanggal'], subset['Volume_Masuk_Ton'], color='#1a6fc4', linewidth=1.5, marker='o', markersize=4)
        ax.fill_between(subset['Tanggal'], subset['Volume_Masuk_Ton'], alpha=0.15, color='#1a6fc4')
        ax.set_title(f'Volume Masuk - {jenis_pilih[:40]}', fontsize=12, pad=12)
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Volume (Ton)')
        set_plot_style(fig, [ax])
        st.pyplot(fig)

        st.markdown('<div class="section-header">Total Volume per Bulan</div>', unsafe_allow_html=True)
        df['BulanTahun'] = df['Tanggal'].dt.to_period('M').astype(str)
        bulanan = df.groupby('BulanTahun')['Volume_Masuk_Ton'].sum().reset_index()

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.bar(bulanan['BulanTahun'], bulanan['Volume_Masuk_Ton'], color='#0a9e70', alpha=0.8, width=0.6)
        ax2.set_title('Total Volume Limbah B3 per Bulan', fontsize=12, pad=12)
        ax2.set_xlabel('Bulan')
        ax2.set_ylabel('Total Volume (Ton)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        set_plot_style(fig2, [ax2])
        st.pyplot(fig2)

# ─── EVALUASI MODEL ──────────────────────────────────────────
elif menu == "Evaluasi Model":
    st.markdown('<div class="page-title">Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Performa file model (.joblib) hasil ekstraksi Google Colab</div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.markdown('<div class="info-box">Upload data terlebih dahulu di halaman Beranda.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state['df']

        st.markdown("""
        <div class="info-box">
            <strong>Informasi Sistem:</strong> Model dimuat langsung dari berkas fisik <code>model_rf_limbah.joblib</code>. 
            Fitur waktu dan sisa TPS diproses secara dinamis menyelaraskan struktur latih Colab.
        </div>
        """, unsafe_allow_html=True)

        try:
            with st.spinner("Memuat file model dan komponen pendukung..."):
                model, le_jenis, le_sumber, fitur = muat_model_dan_encoder()
                
                df_clean = df.copy()
                df_clean = df_clean.sort_values(['Jenis_Limbah_B3', 'Tanggal']).reset_index(drop=True)
                
                # Ekstraksi fitur waktu
                df_clean['Bulan']            = df_clean['Tanggal'].dt.month
                df_clean['Tahun']            = df_clean['Tanggal'].dt.year
                df_clean['Kuartal']          = df_clean['Tanggal'].dt.quarter
                df_clean['Hari_dalam_Bulan'] = df_clean['Tanggal'].dt.day
                
                # Melakukan transformasi data menggunakan encoder joblib
                df_clean['Jenis_Encoded']    = le_jenis.transform(df_clean['Jenis_Limbah_B3'])
                df_clean['Sumber_Encoded']   = le_sumber.transform(df_clean['Sumber'])
                
                sisa_terakhir = df_clean.groupby('Jenis_Limbah_B3')['Sisa_di_TPS_Ton'].last()
                
                X_eval = df_clean[fitur]
                y_eval = df_clean['Volume_Masuk_Ton']
                
                _, X_test, _, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)

                # Amankan ke session state
                st.session_state['model']         = model
                st.session_state['le_jenis']      = le_jenis
                st.session_state['le_sumber']     = le_sumber
                st.session_state['fitur']         = fitur
                st.session_state['sisa_terakhir'] = sisa_terakhir

            # --- KALKULASI UKURAN AKURASI ---
            mae  = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2   = r2_score(y_test, y_pred)

            if r2 >= 0.8:
                badge = '<span class="badge badge-green">Model Baik</span>'
            elif r2 >= 0.6:
                badge = '<span class="badge badge-yellow">Model Cukup</span>'
            else:
                badge = '<span class="badge badge-red">Model Kurang Baik</span>'

            st.markdown('<div class="section-header">Hasil Evaluasi Akurasi</div>', unsafe_allow_html=True)
            st.markdown(badge, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card metric-good"><div class="metric-label">R² Score</div><div class="metric-value">{r2:.4f}</div><div class="metric-sub">Koefisien Determinasi</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card metric-info"><div class="metric-label">MAE</div><div class="metric-value">{mae:.4f}</div><div class="metric-sub">Ton · Mean Absolute Error</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card metric-warn"><div class="metric-label">RMSE</div><div class="metric-value">{rmse:.4f}</div><div class="metric-sub">Ton · Root Mean Squared Error</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Aktual vs Prediksi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(range(len(y_test)), y_test.values, label='Aktual', color='#1a6fc4', linewidth=1.5, marker='o', markersize=3)
            ax.plot(range(len(y_pred)), y_pred, label='Prediksi', color='#e03c31', linewidth=1.5, linestyle='--', marker='x', markersize=3)
            ax.set_title('Perbandingan Nilai Aktual vs Prediksi Model Terpilih', fontsize=12, pad=12)
            ax.set_xlabel('Indeks Data Uji')
            ax.set_ylabel('Volume (Ton)')
            ax.legend(fontsize=9)
            set_plot_style(fig, [ax])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Gagal memetakan model! Pesan kesalahan: {str(e)}")

# ─── FORECASTING ─────────────────────────────────────────────
elif menu == "Forecasting":
    st.markdown('<div class="page-title">Forecasting Volume Limbah B3</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Prediksi volume limbah B3 periode mendatang menggunakan model joblib resmi</div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.markdown('<div class="info-box">Upload data terlebih dahulu di halaman Beranda.</div>', unsafe_allow_html=True)
    elif 'model' not in st.session_state:
        st.markdown('<div class="info-box">Buka halaman Evaluasi Model terlebih dahulu untuk memicu sistem pembacaan joblib.</div>', unsafe_allow_html=True)
    else:
        df            = st.session_state['df']
        model         = st.session_state['model']
        le_jenis      = st.session_state['le_jenis']
        le_sumber     = st.session_state['le_sumber']
        fitur         = st.session_state['fitur']
        sisa_terakhir = st.session_state['sisa_terakhir']

        col_set, _ = st.columns([1, 2])
        with col_set:
            n_bulan = st.slider("Jumlah Bulan Forecasting", min_value=1, max_value=12, value=3)
            if st.button("Jalankan Forecasting"):
                st.session_state['run_forecast'] = True
                st.session_state['n_bulan'] = n_bulan

        if st.session_state.get('run_forecast'):
            n_bulan = st.session_state['n_bulan']
            tanggal_terakhir = df['Tanggal'].max()
            periode = pd.date_range(start=tanggal_terakhir + pd.DateOffset(months=1), periods=n_bulan, freq='MS')

            jenis_list  = df['Jenis_Limbah_B3'].unique()
            sumber_mode = df.groupby('Jenis_Limbah_B3')['Sumber'].agg(lambda x: x.mode()[0])

            sisa_current = sisa_terakhir.copy()
            rows = []
            for tgl in periode:
                for jenis in jenis_list:
                    rows.append({
                        'Tanggal'         : tgl,
                        'Jenis_Limbah_B3' : jenis,
                        'Sumber'          : sumber_mode[jenis],
                        'Bulan'           : tgl.month,
                        'Tahun'           : tgl.year,
                        'Kuartal'         : (tgl.month - 1) // 3 + 1,
                        'Hari_dalam_Bulan': 15,
                        'Sisa_di_TPS_Ton' : sisa_current[jenis],
                    })

            df_fc = pd.DataFrame(rows)
            df_fc['Jenis_Encoded']  = le_jenis.transform(df_fc['Jenis_Limbah_B3'])
            df_fc['Sumber_Encoded'] = le_sumber.transform(df_fc['Sumber'])
            df_fc['Volume_Prediksi_Ton'] = model.predict(df_fc[fitur]).round(4)

            st.markdown('<div class="section-header">Ringkasan Total per Bulan</div>', unsafe_allow_html=True)
            ringkasan = df_fc.groupby(df_fc['Tanggal'].dt.strftime('%B %Y'))['Volume_Prediksi_Ton'].sum().round(4).reset_index()
            ringkasan.columns = ['Periode', 'Total Volume Prediksi (Ton)']

            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, row in ringkasan.iterrows():
                with cols[i % 3]:
                    st.markdown(f'<div class="metric-card metric-good"><div class="metric-label">{row["Periode"]}</div><div class="metric-value">{row["Total Volume Prediksi (Ton)"]:.4f}</div><div class="metric-sub">ton total semua jenis</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Detail per Jenis Limbah</div>', unsafe_allow_html=True)
            tabel = df_fc[['Tanggal','Jenis_Limbah_B3','Volume_Prediksi_Ton']].copy()
            tabel['Tanggal'] = tabel['Tanggal'].dt.strftime('%B %Y')
            tabel.columns = ['Periode','Jenis Limbah B3','Volume Prediksi (Ton)']
            st.dataframe(tabel, use_container_width=True, hide_index=True)

            st.markdown('<div class="section-header">Grafik Forecasting</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(11, 4))
            colors = ['#1a6fc4','#0a9e70','#e09b0a','#e03c31','#7c4dcc']
            for i, jenis in enumerate(jenis_list):
                sub = df_fc[df_fc['Jenis_Limbah_B3'] == jenis]
                ax.plot(sub['Tanggal'], sub['Volume_Prediksi_Ton'], marker='o', linewidth=1.8, markersize=6, label=jenis[:25], color=colors[i % len(colors)])
            ax.set_title(f'Forecasting Volume Limbah B3 - {n_bulan} Bulan ke Depan', fontsize=12, pad=12)
            ax.set_xlabel('Periode')
            ax.set_ylabel('Volume Prediksi (Ton)')
            ax.legend(fontsize=8)
            set_plot_style(fig, [ax])
            st.pyplot(fig)

            st.markdown('<div class="section-header">Unduh Hasil</div>', unsafe_allow_html=True)
            csv = tabel.to_csv(index=False).encode('utf-8')
            st.download_button(label="Unduh Hasil Forecasting (CSV)", data=csv, file_name="hasil_forecasting.csv", mime="text/csv")

# ─── TENTANG SISTEM ──────────────────────────────────────────
elif menu == "Tentang Sistem":
    st.markdown('<div class="page-title">Tentang Sistem</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Informasi sistem prediksi volume limbah B3</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Sistem ini disinkronisasikan langsung dengan struktur latih model fisik (.joblib) dari Google Colab terbaru Anda untuk menjaga akurasi skripsi tetap konsisten.
    </div>
    """, unsafe_allow_html=True)
