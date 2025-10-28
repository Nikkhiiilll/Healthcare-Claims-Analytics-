import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Claims Analysis", layout="wide")
st.title("Healthcare Claims Analysis")

@st.cache_data
def generate_claims(n=50000, seed=42):
    np.random.seed(seed)
    claim_types = ['Inpatient','Outpatient','Pharmacy']
    payer = ['Aetna','United','Cigna','BlueCross']
    dates = pd.date_range('2023-01-01','2024-12-31')
    df = pd.DataFrame({
        'claim_id': np.arange(n),
        'date': np.random.choice(dates, n),
        'patient_age': np.random.randint(0,100, n),
        'claim_amount': np.round(np.random.exponential(3000, n) + 50, 2),
        'claim_type': np.random.choice(claim_types, n, p=[0.4,0.5,0.1]),
        'payer': np.random.choice(payer, n)
    })
    # Inject some anomalies
    idx = np.random.choice(n, int(0.005*n), replace=False)
    df.loc[idx, 'claim_amount'] *= 20
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df

df = generate_claims(50000)

with st.sidebar:
    st.header("Filters")
    payer = st.multiselect("Payer", df['payer'].unique(), default=df['payer'].unique())
    ctype = st.multiselect("Claim Type", df['claim_type'].unique(), default=df['claim_type'].unique())
    date_range = st.date_input("Date range", [df['date'].min(), df['date'].max()])

mask = (df['payer'].isin(payer)) & (df['claim_type'].isin(ctype)) & (df['date']>=pd.to_datetime(date_range[0])) & (df['date']<=pd.to_datetime(date_range[1]))
view = df[mask]

st.markdown("## KPIs")
col1,col2,col3,col4 = st.columns(4)
col1.metric("Total Claims", f"{len(view):,}")
col2.metric("Total Amount", f"₹{int(view['claim_amount'].sum()):,}")
col3.metric("Average Claim", f"₹{int(view['claim_amount'].mean()):,}")
col4.metric("Median Claim", f"₹{int(view['claim_amount'].median()):,}")

#Trend
tr = view.groupby('month', as_index=False)['claim_amount'].sum()
fig1 = px.line(tr, x='month', y='claim_amount', title='Monthly Claim Amount', markers=True)
st.plotly_chart(fig1, use_container_width=True)

#Payer comparison
pay = view.groupby('payer', as_index=False)['claim_amount'].sum()
fig2 = px.bar(pay, x='payer', y='claim_amount', title='Claim Amount by Payer')
st.plotly_chart(fig2, use_container_width=True)

#Anomaly detection
st.markdown("## Anomaly Detection (Isolation Forest)")
sample = view[['claim_amount','patient_age']].dropna().sample(min(5000, len(view)), random_state=42)
iso = IsolationForest(contamination=0.005, random_state=42)
iso.fit(sample)
sample['anomaly'] = iso.predict(sample)
outliers = sample[sample['anomaly']==-1]
st.write(f"Detected {len(outliers)} anomalous claims in the sample")

fig3 = px.scatter(sample, x='patient_age', y='claim_amount', color=sample['anomaly'].map({1:'normal',-1:'anomaly'}), title='Anomalies: claim_amount vs patient_age')
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### Sample flagged anomalies")
st.dataframe(outliers.head(10))

