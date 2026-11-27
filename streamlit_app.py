import streamlit as st
import pandas as pd
import pickle
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="Dashboard Analitik DJPb",
    layout="wide",
)


@st.cache_resource
def load_prediction_model():
    model_path = Path(__file__).resolve().parent / "models" / "Best_model.pkcls"
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)

st.title("Dashboard Analitik DJPb-rindwa adhi")

# Sidebar Filter
st.sidebar.header("Menu Filter")
opsi = st.sidebar.selectbox("Pilih Model:", ["Klasifikasi", "Regresi"])

data = pd.read_csv("data/02_realisasi_anggaran_klasifikasi.csv")

data_visualisasi = data.copy()
kementerian = st.sidebar.multiselect(
    "Kementerian",
    options=sorted(data["nama_kementerian"].dropna().unique()),
    placeholder="Semua kementerian",
)
provinsi = st.sidebar.multiselect(
    "Provinsi",
    options=sorted(data["provinsi"].dropna().unique()),
    placeholder="Semua provinsi",
)
tipe_satker_pilihan = st.sidebar.multiselect(
    "Tipe Satker",
    options=sorted(data["tipe_satker"].dropna().unique()),
    placeholder="Semua tipe satker",
)
status_realisasi = st.sidebar.selectbox(
    "Status Realisasi ≥95%",
    ["Semua", *sorted(data["realisasi_tercapai_95persen"].dropna().unique())],
)
rentang_skor = st.sidebar.slider(
    "Rentang Skor IKPA",
    min_value=float(data["skor_ikpa"].min()),
    max_value=float(data["skor_ikpa"].max()),
    value=(float(data["skor_ikpa"].min()), float(data["skor_ikpa"].max())),
    step=1.0,
)

if kementerian:
    data_visualisasi = data_visualisasi[
        data_visualisasi["nama_kementerian"].isin(kementerian)
    ]
if provinsi:
    data_visualisasi = data_visualisasi[data_visualisasi["provinsi"].isin(provinsi)]
if tipe_satker_pilihan:
    data_visualisasi = data_visualisasi[
        data_visualisasi["tipe_satker"].isin(tipe_satker_pilihan)
    ]
if status_realisasi != "Semua":
    data_visualisasi = data_visualisasi[
        data_visualisasi["realisasi_tercapai_95persen"] == status_realisasi
    ]
data_visualisasi = data_visualisasi[
    data_visualisasi["skor_ikpa"].between(*rentang_skor)
]
st.sidebar.caption(f"Menampilkan {len(data_visualisasi)} dari {len(data)} satker")

tab_prediksi, tab_visualisasi = st.tabs(["Menu Prediksi", "Visualisasi"])

with tab_prediksi:
    st.header("Prediksi Realisasi 95%")
    with st.form("form_prediksi"):
        st.markdown("Masukkan data berikut untuk menyiapkan prediksi.")

        col1, col2 = st.columns(2)
        with col1:
            jumlah_spm = st.number_input(
                "Jumlah SPM",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                format="%d",
            )
            revisi_dipa = st.number_input(
                "Revisi DIPA",
                min_value=0,
                max_value=5,
                value=1,
                step=1,
                format="%d",
            )
            deviasi_rpd_persen = st.number_input(
                "Deviasi RPD (%)",
                min_value=0.00,
                max_value=30.00,
                value=5.50,
                format="%.2f",
            )
        with col2:
            skor_ikpa = st.slider(
                "Skor IKPA",
                min_value=70.0,
                max_value=100.0,
                value=85.0,
                step=0.01,
                format="%.2f",
            )
            tipe_satker = st.selectbox(
                "Tipe Satker",
                ["Dekonsentrasi", "Kantor Daerah", "Kantor Pusat", "Tugas Pembantuan"],
            )
        submit_prediksi = st.form_submit_button("Prediksi")

    if submit_prediksi:
        model = load_prediction_model()
        input_values = {
            "jumlah_spm": jumlah_spm,
            "revisi_dipa": revisi_dipa,
            "deviasi_rpd_persen": deviasi_rpd_persen,
            "skor_ikpa": skor_ikpa,
            **{
                f"tipe_satker={tipe}": float(tipe_satker == tipe)
                for tipe in [
                    "Dekonsentrasi",
                    "Kantor Daerah",
                    "Kantor Pusat",
                    "Tugas Pembantuan",
                ]
            },
        }
        feature_names = [variable.name for variable in model.domain.attributes]
        missing_features = [name for name in feature_names if name not in input_values]
        if missing_features:
            st.error(
                "Fitur model tidak tersedia pada formulir: "
                + ", ".join(missing_features)
            )
            st.stop()

        input_row = [[input_values[name] for name in feature_names]]
        prediction_index = int(model.skl_model.predict(input_row)[0])
        target_labels = model.domain.class_vars[0].values
        if not 0 <= prediction_index < len(target_labels):
            st.error("Model mengembalikan kelas prediksi yang tidak dikenal.")
            st.stop()
        probabilities = model.skl_model.predict_proba(input_row)[0]
        probability_by_label = {
            target_labels[int(class_index)]: float(probability)
            for class_index, probability in zip(
                model.skl_model.classes_, probabilities
            )
        }

        with st.container(border=True):
            st.subheader("Hasil Prediksi")
            result_col, no_col, yes_col = st.columns([1.3, 1, 1])
            with result_col:
                st.metric(
                    "Prediksi Realisasi ≥95%",
                    target_labels[prediction_index],
                )
            with no_col:
                st.metric("Probabilitas Tidak", f"{probability_by_label['Tidak']:.1%}")
            with yes_col:
                st.metric("Probabilitas Ya", f"{probability_by_label['Ya']:.1%}")

with tab_visualisasi:
    jenis_visualisasi = st.selectbox(
        "Pilih jenis visualisasi:",
        [
            "Bubble Plot",
            "Scatter Plot",
            "Diagram Batang",
            "Histogram",
            "Box Plot",
        ],
    )
    st.header(jenis_visualisasi)

    if data_visualisasi.empty:
        st.info("Tidak ada satker yang cocok dengan filter yang dipilih.")
    else:
        chart = alt.Chart(data_visualisasi)
        if jenis_visualisasi == "Bubble Plot":
            chart = chart.mark_circle(opacity=0.7, size=100).encode(
                x=alt.X(
                    "skor_ikpa:Q",
                    title="Skor IKPA",
                    scale=alt.Scale(domain=[70, 100]),
                ),
                y=alt.Y("deviasi_rpd_persen:Q", title="Deviasi RPD (%)"),
                color=alt.Color(
                    "realisasi_tercapai_95persen:N",
                    title="Realisasi Tercapai 95%",
                    scale=alt.Scale(
                        domain=["Tidak", "Ya"],
                        range=["#F97316", "#2563EB"],
                    ),
                ),
                tooltip=[
                    "kode_satker",
                    "tipe_satker",
                    "skor_ikpa",
                    "deviasi_rpd_persen",
                    "realisasi_tercapai_95persen",
                ],
            )
        elif jenis_visualisasi == "Scatter Plot":
            chart = chart.mark_circle(opacity=0.7, size=70).encode(
                x=alt.X("skor_ikpa:Q", title="Skor IKPA"),
                y=alt.Y("deviasi_rpd_persen:Q", title="Deviasi RPD (%)"),
                color=alt.Color(
                    "realisasi_tercapai_95persen:N",
                    title="Realisasi Tercapai 95%",
                    scale=alt.Scale(
                        domain=["Tidak", "Ya"],
                        range=["#F97316", "#2563EB"],
                    ),
                ),
                tooltip=["kode_satker", "skor_ikpa", "deviasi_rpd_persen"],
            )
        elif jenis_visualisasi == "Diagram Batang":
            chart = chart.mark_bar().encode(
                x=alt.X("tipe_satker:N", title="Tipe Satker", sort="-y"),
                y=alt.Y("mean(skor_ikpa):Q", title="Rata-rata Skor IKPA"),
                color=alt.Color(
                    "tipe_satker:N",
                    title="Tipe Satker",
                    scale=alt.Scale(scheme="blues"),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("tipe_satker:N", title="Tipe Satker"),
                    alt.Tooltip("mean(skor_ikpa):Q", title="Rata-rata Skor IKPA"),
                ],
            )
        elif jenis_visualisasi == "Histogram":
            chart = chart.mark_bar().encode(
                x=alt.X("skor_ikpa:Q", bin=alt.Bin(maxbins=20), title="Skor IKPA"),
                y=alt.Y("count():Q", title="Jumlah Satker"),
                tooltip=[alt.Tooltip("count():Q", title="Jumlah Satker")],
            )
        else:
            chart = chart.mark_boxplot().encode(
                x=alt.X("tipe_satker:N", title="Tipe Satker"),
                y=alt.Y("deviasi_rpd_persen:Q", title="Deviasi RPD (%)"),
            )

        chart = chart.properties(width=900, height=550).interactive()
        st.altair_chart(chart, width=900)