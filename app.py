import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─── KONFIGURASI HALAMAN ─────────────────────────────────────
st.set_page_config(
    page_title="Sistem Prediksi Limbah B3 - PT Sawit",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS KUSTOM ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        color: #e8e8e8;
    }

    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7d8590;
        margin-bottom: 6px;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 500;
        color: #e6edf3;
        line-height: 1.2;
    }

    .metric-sub {
        font-size: 12px;
        color: #7d8590;
        margin-top: 4px;
    }

    .metric-good { border-left: 3px solid #3fb950; }
    .metric-warn { border-left: 3px solid #d29922; }
    .metric-info { border-left: 3px solid #388bfd; }
    .metric-muted { border-left: 3px solid #484f58; }

    .section-header {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: #7d8590;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }

    .page-title {
        font-size: 24px;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.02em;
    }

    .page-sub {
        font-size: 14px;
        color: #7d8590;
        margin-top: 4px;
        margin-bottom: 24px;
    }

    .badge {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.05em;
    }

    .badge-green { background: #1a3a1e; color: #3fb950; border: 1px solid #3fb950; }
    .badge-yellow { background: #2d2008; color: #d29922; border: 1px solid #d29922; }
    .badge-red { background: #3d1a1a; color: #f85149; border: 1px solid #f85149; }

    .info-box {
        background: #161b22;
        border: 1px solid #21262d;
        border-left: 3px solid #388bfd;
        border-radius: 6px;
        padding: 14px 16px;
        font-size: 13px;
        color: #adbac7;
        margin: 12px 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #21262d;
        border-radius: 8px;
    }

    .stButton > button {
        background: #238636;
        color: #ffffff;
        border: 1px solid #2ea043;
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 20px;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background: #2ea043;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7d8590;
    }

    hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ─── FUNGSI BANTUAN ───────────────────────────────────────────
def set_plot_style(fig, ax_list):
    fig.patch.set_facecolor('#0f1117')
    for ax in ax_list:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#7d8590', labelsize=9)
        ax.xaxis.label.set_color('#7d8590')
        ax.yaxis.label.set_color('#7d8590')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#21262d')
    return fig

@st.cache_data
def muat_data(file):
    df = pd.read_csv(file, parse_dates=['Tanggal'])
    return df

@st.cache_resource
def latih_model(df):
    le_jenis  = LabelEncoder()
    le_sumber = LabelEncoder()
    df = df.copy()
    df['Jenis_Encoded']  = le_jenis.fit_transform(df['Jenis_Limbah_B3'])
    df['Sumber_Encoded'] = le_sumber.fit_transform(df['Sumber'])

    fitur = ['Bulan','Tahun','Kuartal','Hari_dalam_Bulan',
             'Jenis_Encoded','Sumber_Encoded','Sisa_di_TPS_Ton']

    X = df[fitur]
    y = df['Volume_Masuk_Ton']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=15,
        min_samples_split=5, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, le_jenis, le_sumber, fitur, X_test, y_test, y_pred

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Sistem Prediksi Limbah B3")
    st.markdown("**PT Sawit** · Teknik Informatika")
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
        'Menggunakan Algoritma Random Forest</div>',
        unsafe_allow_html=True
    )

# ─── BERANDA ─────────────────────────────────────────────────
if menu == "Beranda":
    st.markdown('<div class="page-title">Sistem Prediksi Volume Limbah B3</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">PT Sawit — Upload data untuk memulai analisis dan prediksi</div>', unsafe_allow_html=True)

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

        # Distribusi per jenis
        st.markdown('<div class="section-header">Distribusi Jenis Limbah B3</div>', unsafe_allow_html=True)
        dist = df.groupby('Jenis_Limbah_B3')['Volume_Masuk_Ton'].agg(['sum','mean','count']).round(4)
        dist.columns = ['Total Volume (Ton)', 'Rata-rata (Ton)', 'Jumlah Transaksi']
        st.dataframe(dist, use_container_width=True)

        # Grafik tren
        st.markdown('<div class="section-header">Tren Volume per Jenis Limbah</div>', unsafe_allow_html=True)
        jenis_pilih = st.selectbox(
            "Pilih Jenis Limbah",
            options=df['Jenis_Limbah_B3'].unique()
        )

        subset = df[df['Jenis_Limbah_B3'] == jenis_pilih].sort_values('Tanggal')
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(subset['Tanggal'], subset['Volume_Masuk_Ton'],
                color='#388bfd', linewidth=1.5, marker='o', markersize=4)
        ax.fill_between(subset['Tanggal'], subset['Volume_Masuk_Ton'],
                        alpha=0.15, color='#388bfd')
        ax.set_title(f'Volume Masuk - {jenis_pilih[:40]}', fontsize=12, pad=12)
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Volume (Ton)')
        set_plot_style(fig, [ax])
        st.pyplot(fig)

        # Volume per bulan
        st.markdown('<div class="section-header">Total Volume per Bulan</div>', unsafe_allow_html=True)
        df['BulanTahun'] = df['Tanggal'].dt.to_period('M').astype(str)
        bulanan = df.groupby('BulanTahun')['Volume_Masuk_Ton'].sum().reset_index()

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.bar(bulanan['BulanTahun'], bulanan['Volume_Masuk_Ton'],
                color='#3fb950', alpha=0.8, width=0.6)
        ax2.set_title('Total Volume Limbah B3 per Bulan', fontsize=12, pad=12)
        ax2.set_xlabel('Bulan')
        ax2.set_ylabel('Total Volume (Ton)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        set_plot_style(fig2, [ax2])
        st.pyplot(fig2)

# ─── EVALUASI MODEL ──────────────────────────────────────────
elif menu == "Evaluasi Model":
    st.markdown('<div class="page-title">Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Performa algoritma Random Forest pada data uji</div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.markdown('<div class="info-box">Upload data terlebih dahulu di halaman Beranda.</div>', unsafe_allow_html=True)
    else:
        df = st.session_state['df']

        with st.spinner("Melatih model Random Forest..."):
            model, le_jenis, le_sumber, fitur, X_test, y_test, y_pred = latih_model(df)
            st.session_state['model']    = model
            st.session_state['le_jenis'] = le_jenis
            st.session_state['le_sumber']= le_sumber
            st.session_state['fitur']    = fitur

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        if r2 >= 0.8:
            badge = '<span class="badge badge-green">Model Baik</span>'
        elif r2 >= 0.6:
            badge = '<span class="badge badge-yellow">Model Cukup</span>'
        else:
            badge = '<span class="badge badge-red">Model Kurang Baik</span>'

        st.markdown('<div class="section-header">Hasil Evaluasi</div>', unsafe_allow_html=True)
        st.markdown(badge, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-good">
                <div class="metric-label">R² Score</div>
                <div class="metric-value">{r2:.4f}</div>
                <div class="metric-sub">Koefisien Determinasi</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-info">
                <div class="metric-label">MAE</div>
                <div class="metric-value">{mae:.4f}</div>
                <div class="metric-sub">Ton · Mean Absolute Error</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-warn">
                <div class="metric-label">RMSE</div>
                <div class="metric-value">{rmse:.4f}</div>
                <div class="metric-sub">Ton · Root Mean Squared Error</div>
            </div>""", unsafe_allow_html=True)

        # Grafik aktual vs prediksi
        st.markdown('<div class="section-header">Aktual vs Prediksi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(range(len(y_test)), y_test.values,
                label='Aktual', color='#388bfd', linewidth=1.5, marker='o', markersize=3)
        ax.plot(range(len(y_pred)), y_pred,
                label='Prediksi', color='#f85149', linewidth=1.5,
                linestyle='--', marker='x', markersize=3)
        ax.set_title('Perbandingan Nilai Aktual vs Prediksi', fontsize=12, pad=12)
        ax.set_xlabel('Indeks Data Uji')
        ax.set_ylabel('Volume (Ton)')
        ax.legend(fontsize=9)
        set_plot_style(fig, [ax])
        st.pyplot(fig)

        # Scatter plot
        st.markdown('<div class="section-header">Scatter Plot</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.scatter(y_test, y_pred, alpha=0.6, color='#388bfd',
                    edgecolors='#21262d', linewidth=0.5, s=40)
        ax2.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', linewidth=1.5, label='Garis Ideal')
        ax2.set_xlabel('Nilai Aktual (Ton)')
        ax2.set_ylabel('Nilai Prediksi (Ton)')
        ax2.set_title(f'Scatter Plot (R² = {r2:.4f})', fontsize=12, pad=12)
        ax2.legend(fontsize=9)
        set_plot_style(fig2, [ax2])
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.pyplot(fig2)

        # Feature importance
        with col_b:
            st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=fitur).sort_values(ascending=True)
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            ax3.barh(feat_imp.index, feat_imp.values, color='#388bfd', alpha=0.85)
            ax3.set_title('Tingkat Kepentingan Fitur', fontsize=11, pad=10)
            ax3.set_xlabel('Importance Score')
            set_plot_style(fig3, [ax3])
            st.pyplot(fig3)

# ─── FORECASTING ─────────────────────────────────────────────
elif menu == "Forecasting":
    st.markdown('<div class="page-title">Forecasting Volume Limbah B3</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Prediksi volume limbah B3 untuk periode mendatang</div>', unsafe_allow_html=True)

    if 'df' not in st.session_state:
        st.markdown('<div class="info-box">Upload data terlebih dahulu di halaman Beranda.</div>', unsafe_allow_html=True)
    elif 'model' not in st.session_state:
        st.markdown('<div class="info-box">Jalankan Evaluasi Model terlebih dahulu.</div>', unsafe_allow_html=True)
    else:
        df       = st.session_state['df']
        model    = st.session_state['model']
        le_jenis = st.session_state['le_jenis']
        le_sumber= st.session_state['le_sumber']
        fitur    = st.session_state['fitur']

        col_set, _ = st.columns([1, 2])
        with col_set:
            n_bulan = st.slider("Jumlah Bulan Forecasting", min_value=1, max_value=12, value=3)
            if st.button("Jalankan Forecasting"):
                st.session_state['run_forecast'] = True
                st.session_state['n_bulan'] = n_bulan

        if st.session_state.get('run_forecast'):
            n_bulan = st.session_state['n_bulan']
            tanggal_terakhir = df['Tanggal'].max()
            periode = pd.date_range(
                start=tanggal_terakhir + pd.DateOffset(months=1),
                periods=n_bulan, freq='MS'
            )

            jenis_list  = df['Jenis_Limbah_B3'].unique()
            sumber_mode = df.groupby('Jenis_Limbah_B3')['Sumber'].agg(lambda x: x.mode()[0])
            sisa_mean   = df.groupby('Jenis_Limbah_B3')['Sisa_di_TPS_Ton'].mean()

            rows = []
            for tgl, jenis in itertools.product(periode, jenis_list):
                rows.append({
                    'Tanggal'         : tgl,
                    'Jenis_Limbah_B3' : jenis,
                    'Sumber'          : sumber_mode[jenis],
                    'Bulan'           : tgl.month,
                    'Tahun'           : tgl.year,
                    'Kuartal'         : (tgl.month - 1) // 3 + 1,
                    'Hari_dalam_Bulan': 15,
                    'Sisa_di_TPS_Ton' : sisa_mean[jenis],
                })

            df_fc = pd.DataFrame(rows)
            df_fc['Jenis_Encoded']  = le_jenis.transform(df_fc['Jenis_Limbah_B3'])
            df_fc['Sumber_Encoded'] = le_sumber.transform(df_fc['Sumber'])
            df_fc['Volume_Prediksi_Ton'] = model.predict(df_fc[fitur]).round(4)

            # Ringkasan per bulan
            st.markdown('<div class="section-header">Ringkasan Total per Bulan</div>', unsafe_allow_html=True)
            ringkasan = df_fc.groupby(df_fc['Tanggal'].dt.strftime('%B %Y'))['Volume_Prediksi_Ton'].sum().round(4).reset_index()
            ringkasan.columns = ['Periode', 'Total Volume Prediksi (Ton)']

            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]
            for i, row in ringkasan.iterrows():
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="metric-card metric-good">
                        <div class="metric-label">{row['Periode']}</div>
                        <div class="metric-value">{row['Total Volume Prediksi (Ton)']:.4f}</div>
                        <div class="metric-sub">ton total semua jenis</div>
                    </div>""", unsafe_allow_html=True)

            # Tabel detail
            st.markdown('<div class="section-header">Detail per Jenis Limbah</div>', unsafe_allow_html=True)
            tabel = df_fc[['Tanggal','Jenis_Limbah_B3','Volume_Prediksi_Ton']].copy()
            tabel['Tanggal'] = tabel['Tanggal'].dt.strftime('%B %Y')
            tabel.columns = ['Periode','Jenis Limbah B3','Volume Prediksi (Ton)']
            st.dataframe(tabel, use_container_width=True, hide_index=True)

            # Grafik forecasting
            st.markdown('<div class="section-header">Grafik Forecasting</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(11, 4))
            colors = ['#388bfd','#3fb950','#d29922','#f85149','#a371f7']
            for i, jenis in enumerate(jenis_list):
                sub = df_fc[df_fc['Jenis_Limbah_B3'] == jenis]
                ax.plot(sub['Tanggal'], sub['Volume_Prediksi_Ton'],
                        marker='o', linewidth=1.8, markersize=6,
                        label=jenis[:25], color=colors[i % len(colors)])
            ax.set_title(f'Forecasting Volume Limbah B3 - {n_bulan} Bulan ke Depan', fontsize=12, pad=12)
            ax.set_xlabel('Periode')
            ax.set_ylabel('Volume Prediksi (Ton)')
            ax.legend(fontsize=8)
            set_plot_style(fig, [ax])
            st.pyplot(fig)

            # Download
            st.markdown('<div class="section-header">Unduh Hasil</div>', unsafe_allow_html=True)
            csv = tabel.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Forecasting (CSV)",
                data=csv,
                file_name="hasil_forecasting.csv",
                mime="text/csv"
            )

# ─── TENTANG SISTEM ──────────────────────────────────────────
elif menu == "Tentang Sistem":
    st.markdown('<div class="page-title">Tentang Sistem</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Informasi sistem prediksi volume limbah B3</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Sistem ini dikembangkan sebagai bagian dari penelitian skripsi Teknik Informatika
        dengan judul <strong>"Prediksi Volume Limbah B3 Menggunakan Algoritma Random Forest"</strong>
        pada operasional pabrik kelapa sawit PT Sawit.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Algoritma</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card metric-info">
        <div class="metric-label">Metode Prediksi</div>
        <div class="metric-value" style="font-size:18px;">Random Forest Regressor</div>
        <div class="metric-sub">n_estimators=200 · max_depth=15 · random_state=42</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Fitur yang Digunakan</div>', unsafe_allow_html=True)
    fitur_info = {
        'Bulan': 'Bulan pencatatan limbah (1-12)',
        'Tahun': 'Tahun pencatatan limbah',
        'Kuartal': 'Kuartal dalam tahun (1-4)',
        'Hari_dalam_Bulan': 'Hari ke berapa dalam bulan',
        'Jenis_Encoded': 'Jenis limbah B3 (encoded)',
        'Sumber_Encoded': 'Sumber penghasil limbah (encoded)',
        'Sisa_di_TPS_Ton': 'Sisa limbah di TPS sebelumnya (ton)',
    }
    df_fitur = pd.DataFrame(fitur_info.items(), columns=['Fitur', 'Keterangan'])
    st.dataframe(df_fitur, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Cara Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        1. Upload file <strong>data_limbah_b3_augmented.csv</strong> di halaman Beranda<br>
        2. Buka halaman <strong>Eksplorasi Data</strong> untuk melihat pola data<br>
        3. Buka halaman <strong>Evaluasi Model</strong> untuk melihat akurasi model<br>
        4. Buka halaman <strong>Forecasting</strong> untuk prediksi volume ke depan<br>
        5. Unduh hasil forecasting dalam format CSV
    </div>
    """, unsafe_allow_html=True)
