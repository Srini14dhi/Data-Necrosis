import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# --------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Data Necrosis Analyzer",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 DATA NECROSIS ANALYZER")
st.markdown("Visualizing the decay of untouched retail data")

# --------------- SIDEBAR -------------------- #
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", [
    "Upload & Preview",
    "Decay Analysis",
    "Survival & Churn",
    "Dormancy Forecast",
    "Anomaly Detection",
    "Dashboard Integration"
])

# --------------- SESSION STORAGE ------------- #
if "df" not in st.session_state:
    st.session_state.df = None


# ----------------- UPLOAD PAGE --------------- #
if page == "Upload & Preview":
    st.header("1️⃣ Upload Transaction Data")
    uploaded_file = st.file_uploader("Upload UCI Retail Dataset CSV", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df

        st.success("Data uploaded successfully!")
        st.dataframe(df.head(20))

        st.markdown("### Dataset Summary")
        st.write(df.describe())


# --------- DECAY SCORE COMPUTATION ---------- #
def compute_decay(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%m/%d/%Y %H:%M')
    last_date = df['InvoiceDate'].max()
    df['InactivityDays'] = (last_date - df['InvoiceDate']).dt.days
    df['DecayScore'] = np.log1p(df['InactivityDays'])  # simple scoring
    return df


if page == "Decay Analysis":
    st.header("2️⃣ Data Decay Analysis")

    if st.session_state.df is None:
        st.warning("Upload data in the first tab.")
    else:
        df = compute_decay(st.session_state.df)
        st.session_state.df = df  

        st.success("Decay scores computed!")
        
        st.dataframe(df[['InvoiceDate', 'StockCode', 'Description', 'InactivityDays', 'DecayScore']].head())


        st.line_chart(df.groupby('InactivityDays')['DecayScore'].mean())

        st.markdown("Older transactions = Higher decay score")


# --------- SURVIVAL ANALYSIS (K-M) ----------- #
if page == "Survival & Churn":
    st.header("3️⃣ Customer Survival & Churn")

    if st.session_state.df is None:
        st.warning("Upload data first")
    else:
        st.info("Kaplan–Meier model visualization will be integrated here.")
        st.line_chart(st.session_state.df['DecayScore'][:200])


# --------- FORECASTING ---------------------- #
if page == "Dormancy Forecast":
    st.header("4️⃣ Forecast Dormancy Trends")
    st.info("ARIMA / Prophet / LSTM chart goes here")

    if st.session_state.df is not None:
        decay_ts = st.session_state.df.groupby("InvoiceDate")["DecayScore"].mean()
        st.line_chart(decay_ts)


# ---------- ANOMALY DETECTION --------------- #
if page == "Anomaly Detection":
    st.header("5️⃣ Detect Anomalous Dormancy")
    st.info("Isolation Forest results go here")
    if st.session_state.df is not None:
        st.bar_chart(st.session_state.df['DecayScore'].sample(100))


# ---------- POWER BI INTEGRATION ------------ #
if page == "Dashboard Integration":
    st.header("6️⃣ Retail Data Necrosis Dashboard")

    st.info("This embeds your Power BI dashboard 👇")

    # Replace YOUR_EMBED_LINK with your actual published Power BI link
    powerbi_url = st.text_input("Enter Power BI Embed URL")

    if powerbi_url:
        st.components.v1.iframe(powerbi_url, width=1200, height=600)