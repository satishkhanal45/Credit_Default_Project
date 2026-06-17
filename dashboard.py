import streamlit as st
import requests
import pandas as pd

#  API_URL = "http://localhost:8001"    

TOP_FEATURES            = ['BILL_AMT3', 'PAY_AMT1', 'BILL_AMT1', 'AGE', 'LIMIT_BAL']
TOP_ENGINEERED_FEATURES = ['ZERO_PAY_COUNT', 'AVG_PAID', 'MAX_BILL', 'LIMIT_PER_AGE', 'TOTAL_PAID']

FULL_NAMES = {
    'LIMIT_BAL':  'Credit Limit Balance',
    'SEX':        'Gender',
    'EDUCATION':  'Education Level',
    'MARRIAGE':   'Marital Status',
    'AGE':        'Age',
    'PAY_0':      'Repayment Status — September',
    'PAY_2':      'Repayment Status — August',
    'PAY_3':      'Repayment Status — July',
    'PAY_4':      'Repayment Status — June',
    'PAY_5':      'Repayment Status — May',
    'PAY_6':      'Repayment Status — April',
    'BILL_AMT1':  'Bill Statement — September',
    'BILL_AMT2':  'Bill Statement — August',
    'BILL_AMT3':  'Bill Statement — July',
    'BILL_AMT4':  'Bill Statement — June',
    'BILL_AMT5':  'Bill Statement — May',
    'BILL_AMT6':  'Bill Statement — April',
    'PAY_AMT1':   'Previous Payment — September',
    'PAY_AMT2':   'Previous Payment — August',
    'PAY_AMT3':   'Previous Payment — July',
    'PAY_AMT4':   'Previous Payment — June',
    'PAY_AMT5':   'Previous Payment — May',
    'PAY_AMT6':   'Previous Payment — April',
    'ZERO_PAY_COUNT': 'Months With Zero Payment',
    'AVG_PAID':       'Average Monthly Payment',
    'MAX_BILL':       'Maximum Bill Amount',
    'LIMIT_PER_AGE':  'Credit Limit Per Age',
    'TOTAL_PAID':     'Total Amount Paid (6 months)',
}

def field_label(field_name):
    full = FULL_NAMES.get(field_name, field_name)
    if field_name in TOP_FEATURES:
        return f'🔴 {full}'
    return full

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(page_title="Credit Default Predictor", page_icon="💳", layout="wide")
st.title("💳 Credit Card Default Prediction")

st.markdown("""
    <style>
    div[data-testid="stNumberInput"]:has(label:contains("🔴")),
    div[data-testid="stSlider"]:has(label:contains("🔴")) {
        border-left: 4px solid #e74c3c;
        padding-left: 8px;
        background-color: #f8f8f8;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#c0392b; padding:14px 18px; border-radius:6px; margin-bottom:16px;">
    <span style="color:#ffffff; font-size:16px; font-weight:bold;">
        🔴 Fields marked with 🔴 have the highest impact on the prediction result.
        Pay close attention to these values before predicting.
    </span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        st.success("✅ API is Online") if r.status_code == 200 else st.error("API Error")
    except:
        st.error("❌ Cannot reach API — start FastAPI first")

    st.markdown("---")
    st.markdown("### 🔴 High Impact Features")

    st.markdown("**Original Input Features:**")
    for i, feat in enumerate(TOP_FEATURES, 1):
        st.markdown(
            f'<div style="background:#ffe0e0; padding:6px 10px; border-left:4px solid #e74c3c;'
            f'border-radius:4px; margin-bottom:6px;">'
            f'<b>{i}. {FULL_NAMES.get(feat, feat)}</b></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Auto-calculated Features:**")
    for i, feat in enumerate(TOP_ENGINEERED_FEATURES, 1):
        st.markdown(
            f'<div style="background:#fff0e0; padding:6px 10px; border-left:4px solid #e67e22;'
            f'border-radius:4px; margin-bottom:6px;">'
            f'<b>{i}. {FULL_NAMES.get(feat, feat)}</b></div>',
            unsafe_allow_html=True
        )

# ──────────────────────────────────────────────
# Confusion matrix presets
# TP = real defaulter, model says default   → clearly risky values
# TN = real non-defaulter, model says safe  → clearly safe values
# FP = real non-defaulter, model says default → looks risky but actually safe
# FN = real defaulter, model says safe      → looks safe but actually will default
# ──────────────────────────────────────────────
PRESETS = {
    "✅ True Positive (TP)": {
        "label": "Actual defaulter — model correctly predicted DEFAULT (confidence: 98.1%)",
        "color": "#1a7a2e",
        "bg":    "#d4edda",
        "values": {
            "LIMIT_BAL": 90000,  "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
            "PAY_0": 8,  "PAY_2": 7,  "PAY_3": 6,  "PAY_4": 5,  "PAY_5": 4,  "PAY_6": 3,
            "BILL_AMT1": 112662, "BILL_AMT2": 111077, "BILL_AMT3": 108539,
            "BILL_AMT4": 106001, "BILL_AMT5": 103816, "BILL_AMT6": 101878,
            "PAY_AMT1": 0,       "PAY_AMT2": 0,       "PAY_AMT3": 0,
            "PAY_AMT4": 0,       "PAY_AMT5": 0,       "PAY_AMT6": 0,
        }
    },
    "🔵 True Negative (TN)": {
        "label": "Actual non-defaulter — model correctly predicted NO DEFAULT (confidence: 99.97%)",
        "color": "#1a4a7a",
        "bg":    "#d0e4f7",
        "values": {
            "LIMIT_BAL": 510000, "SEX": 2, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 30,
            "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": 0,  "PAY_6": 0,
            "BILL_AMT1": 71121,   "BILL_AMT2": 481382, "BILL_AMT3": 559712,
            "BILL_AMT4": 163628,  "BILL_AMT5": 117475, "BILL_AMT6": 116656,
            "PAY_AMT1": 493358,   "PAY_AMT2": 1227082, "PAY_AMT3": 164577,
            "PAY_AMT4": 510,      "PAY_AMT5": 6239,    "PAY_AMT6": 4550,
        }
    },
    "⚠️ False Positive (FP)": {
        "label": "Actual non-defaulter — model wrongly predicted DEFAULT (confidence: 95.4%)",
        "color": "#7a4a00",
        "bg":    "#fff3cd",
        "values": {
            "LIMIT_BAL": 30000,  "SEX": 1, "EDUCATION": 3, "MARRIAGE": 2, "AGE": 46,
            "PAY_0": 6,  "PAY_2": 5,  "PAY_3": 4,  "PAY_4": 3,  "PAY_5": 2,  "PAY_6": 0,
            "BILL_AMT1": 33134,  "BILL_AMT2": 32351,  "BILL_AMT3": 31397,
            "BILL_AMT4": 30321,  "BILL_AMT5": 30000,  "BILL_AMT6": 28961,
            "PAY_AMT1": 0,       "PAY_AMT2": 0,       "PAY_AMT3": 0,
            "PAY_AMT4": 0,       "PAY_AMT5": 5000,    "PAY_AMT6": 0,
        }
    },
    "❌ False Negative (FN)": {
        "label": "Actual defaulter — model wrongly predicted NO DEFAULT (confidence: 92.8% safe)",
        "color": "#7a1a1a",
        "bg":    "#f8d7da",
        "values": {
            "LIMIT_BAL": 500000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 36,
            "PAY_0": -2, "PAY_2": -2, "PAY_3": -2, "PAY_4": -2, "PAY_5": -2, "PAY_6": -2,
            "BILL_AMT1": 45106,  "BILL_AMT2": 81264,  "BILL_AMT3": 18122,
            "BILL_AMT4": 27229,  "BILL_AMT5": 21462,  "BILL_AMT6": 27911,
            "PAY_AMT1": 81690,   "PAY_AMT2": 18225,   "PAY_AMT3": 27365,
            "PAY_AMT4": 21570,   "PAY_AMT5": 28050,   "PAY_AMT6": 17397,
        }
    },
}
# ──────────────────────────────────────────────
# Preset buttons
# ──────────────────────────────────────────────
st.subheader("🎯 Confusion Matrix Presets — click to auto-fill")

st.markdown("""
<div style="background:#f0f0f0; border-radius:8px; padding:12px 16px; margin-bottom:12px; font-size:13px; color:#444;">
    <b>TP</b> = model predicted default, customer actually defaults &nbsp;|&nbsp;
    <b>TN</b> = model predicted safe, customer actually is safe &nbsp;|&nbsp;
    <b>FP</b> = model predicted default, but customer is actually safe &nbsp;|&nbsp;
    <b>FN</b> = model predicted safe, but customer actually defaults
</div>
""", unsafe_allow_html=True)

btn_cols = st.columns(4)
selected_preset = None
selected_meta   = None

for i, (name, meta) in enumerate(PRESETS.items()):
    if btn_cols[i].button(name, use_container_width=True):
        selected_preset = meta["values"]
        selected_meta   = meta

if selected_preset:
    st.session_state.update(selected_preset)

# Show explanation banner for whichever preset is active
if selected_meta:
    st.markdown(
        f'<div style="background:{selected_meta["bg"]}; border-left:5px solid {selected_meta["color"]}; '
        f'padding:10px 16px; border-radius:4px; margin-bottom:8px; color:{selected_meta["color"]}; font-weight:600;">'
        f'{selected_meta["label"]}</div>',
        unsafe_allow_html=True
    )

def get(key, default):
    return st.session_state.get(key, default)

# ──────────────────────────────────────────────
# Input fields
# ──────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    LIMIT_BAL = st.number_input(
        field_label('LIMIT_BAL'),
        min_value=0, value=get('LIMIT_BAL', 50000), step=5000
    )
    SEX = st.selectbox(
        field_label('SEX'), [1, 2],
        index=[1,2].index(get('SEX', 1)),
        format_func=lambda x: "Male" if x==1 else "Female"
    )
    EDUCATION = st.selectbox(
        field_label('EDUCATION'), [1, 2, 3, 4],
        index=[1,2,3,4].index(get('EDUCATION', 2)),
        format_func=lambda x: {1:"Graduate School", 2:"University",
                                3:"High School",    4:"Other"}[x]
    )
    MARRIAGE = st.selectbox(
        field_label('MARRIAGE'), [1, 2, 3],
        index=[1,2,3].index(get('MARRIAGE', 1)),
        format_func=lambda x: {1:"Married", 2:"Single", 3:"Other"}[x]
    )
    AGE = st.number_input(
        field_label('AGE'),
        min_value=18, max_value=100, value=get('AGE', 30)
    )

with col2:
    st.markdown("**Repayment Status** &nbsp; *(-2 = No consumption / No balance &nbsp;|&nbsp; −1 = Paid on time &nbsp;|&nbsp; 0=Revolving credit &nbsp;|&nbsp; 1–9 = Months delayed)*",
                unsafe_allow_html=True)
    PAY_0 = st.slider(field_label('PAY_0'), -2, 9, value=get('PAY_0', 0))
    PAY_2 = st.slider(field_label('PAY_2'), -2, 9, value=get('PAY_2', 0))
    PAY_3 = st.slider(field_label('PAY_3'), -2, 9, value=get('PAY_3', 0))
    PAY_4 = st.slider(field_label('PAY_4'), -2, 9, value=get('PAY_4', 0))
    PAY_5 = st.slider(field_label('PAY_5'), -2, 9, value=get('PAY_5', 0))
    PAY_6 = st.slider(field_label('PAY_6'), -2, 9, value=get('PAY_6', 0))

with col3:
    st.markdown("**Bill Statement Amounts (NT$)**")
    BILL_AMT1 = st.number_input(field_label('BILL_AMT1'), value=get('BILL_AMT1', 20000), step=1000)
    BILL_AMT2 = st.number_input(field_label('BILL_AMT2'), value=get('BILL_AMT2', 18000), step=1000)
    BILL_AMT3 = st.number_input(field_label('BILL_AMT3'), value=get('BILL_AMT3', 15000), step=1000)
    BILL_AMT4 = st.number_input(field_label('BILL_AMT4'), value=get('BILL_AMT4', 12000), step=1000)
    BILL_AMT5 = st.number_input(field_label('BILL_AMT5'), value=get('BILL_AMT5', 10000), step=1000)
    BILL_AMT6 = st.number_input(field_label('BILL_AMT6'), value=get('BILL_AMT6', 8000),  step=1000)

st.markdown("**Previous Payment Amounts (NT$)**")
pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
PAY_AMT1 = pc1.number_input(field_label('PAY_AMT1'), value=get('PAY_AMT1', 2000), step=500)
PAY_AMT2 = pc2.number_input(field_label('PAY_AMT2'), value=get('PAY_AMT2', 2000), step=500)
PAY_AMT3 = pc3.number_input(field_label('PAY_AMT3'), value=get('PAY_AMT3', 1500), step=500)
PAY_AMT4 = pc4.number_input(field_label('PAY_AMT4'), value=get('PAY_AMT4', 1500), step=500)
PAY_AMT5 = pc5.number_input(field_label('PAY_AMT5'), value=get('PAY_AMT5', 1000), step=500)
PAY_AMT6 = pc6.number_input(field_label('PAY_AMT6'), value=get('PAY_AMT6', 1000), step=500)


# Predict button

st.markdown("---")
if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):
    payload = dict(
        LIMIT_BAL=LIMIT_BAL, SEX=SEX,       EDUCATION=EDUCATION,
        MARRIAGE=MARRIAGE,   AGE=AGE,
        PAY_0=PAY_0, PAY_2=PAY_2, PAY_3=PAY_3,
        PAY_4=PAY_4, PAY_5=PAY_5, PAY_6=PAY_6,
        BILL_AMT1=BILL_AMT1, BILL_AMT2=BILL_AMT2, BILL_AMT3=BILL_AMT3,
        BILL_AMT4=BILL_AMT4, BILL_AMT5=BILL_AMT5, BILL_AMT6=BILL_AMT6,
        PAY_AMT1=PAY_AMT1,   PAY_AMT2=PAY_AMT2,   PAY_AMT3=PAY_AMT3,
        PAY_AMT4=PAY_AMT4,   PAY_AMT5=PAY_AMT5,   PAY_AMT6=PAY_AMT6,
    )

    try:
        result     = requests.post(f"{API_URL}/predict", json=payload).json()
        prob       = result["default_probability"]
        pred       = result["prediction"]
        label_text = result["label"]

        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"⚠️  {label_text}  —  Probability: {prob*100:.1f}%")
        else:
            st.success(f"✅  {label_text}  —  Probability: {prob*100:.1f}%")

        chart_data = pd.DataFrame(
            {"Probability": [1-prob, prob]},
            index=["Will Not Default", "Will Default"]
        )
        st.bar_chart(chart_data)

        if prob < 0.3:
            st.info("🟢 Low Risk — Customer is unlikely to default")
        elif prob < 0.6:
            st.warning("🟡 Moderate Risk — Monitor this account")
        else:
            st.error("🔴 High Risk — Default is very likely")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API — make sure FastAPI is running on port 8001")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")