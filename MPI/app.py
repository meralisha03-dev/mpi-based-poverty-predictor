import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="India MPI Poverty Risk Predictor",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Deep navy + saffron theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2a3b 60%, #162032 100%);
    color: #e8edf2;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #112240 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #ccd6f6 !important; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #112240 0%, #0d2137 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(255,153,51,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #FF9933;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.95rem;
    color: #8892b0;
    margin: 0;
    letter-spacing: 0.04em;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,153,51,0.15);
    border: 1px solid rgba(255,153,51,0.4);
    color: #FF9933;
    font-size: 0.75rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}
.metric-card {
    background: linear-gradient(135deg, #112240, #0d2137);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #FF9933;
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #8892b0;
    margin-top: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: #ccd6f6;
    border-left: 3px solid #FF9933;
    padding-left: 12px;
    margin: 28px 0 16px 0;
}

/* Result cards */
.result-low {
    background: linear-gradient(135deg, #0d3322, #0a2218);
    border: 1px solid #1a6640;
    border-left: 5px solid #27ae60;
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
}
.result-medium {
    background: linear-gradient(135deg, #332900, #261f00);
    border: 1px solid #665200;
    border-left: 5px solid #f39c12;
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
}
.result-high {
    background: linear-gradient(135deg, #330d0d, #200808);
    border: 1px solid #661a1a;
    border-left: 5px solid #e74c3c;
    border-radius: 12px;
    padding: 22px 26px;
    text-align: center;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
}
.result-desc {
    font-size: 0.85rem;
    color: #8892b0;
    margin: 0;
}

/* Solution cards */
.solution-card {
    background: rgba(17, 34, 64, 0.7);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.solution-icon { font-size: 1.4rem; margin-top: 2px; }
.solution-text { font-size: 0.88rem; color: #ccd6f6; line-height: 1.5; }
.solution-title { font-weight: 600; color: #FF9933; margin-bottom: 3px; }

/* Info box */
.info-box {
    background: rgba(255,153,51,0.08);
    border: 1px solid rgba(255,153,51,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #ccd6f6;
    margin-bottom: 18px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: #4a5568;
    font-size: 0.78rem;
    border-top: 1px solid #1e3a5f;
    margin-top: 40px;
}

/* Streamlit overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: #ccd6f6 !important; font-size: 0.85rem !important; }
.stButton > button {
    background: linear-gradient(135deg, #FF9933, #e67e00);
    color: #0d1b2a !important;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 14px 36px;
    width: 100%;
    cursor: pointer;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
div[data-testid="stDataFrame"] { background: #112240 !important; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  LOAD DATA & TRAIN MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("ABA FINAL MPI DATASET.csv")
    return df

@st.cache_resource
def train_model(df):
    drop_cols = [c for c in ['State', 'Region',
                              'Poverty_Risk_Category',
                              'Poverty_Risk_Label'] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['Poverty_Risk_Label']
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        random_state=42, class_weight='balanced')
    model.fit(X, y)
    return model, list(X.columns)

df_original = load_data()
model, feature_cols = train_model(df_original)

INDICATOR_COLS = [
    'Nutrition_Deprived', 'Child_Mortality_Deprived',
    'Maternal_Health_Deprived', 'Years_of_Schooling_Deprived',
    'School_Attendance_Deprived', 'Cooking_Fuel_Deprived',
    'Sanitation_Deprived', 'Drinking_Water_Deprived',
    'Electricity_Deprived', 'Housing_Deprived',
    'Assets_Deprived', 'Bank_Account_Deprived'
]
INDICATOR_COLS = [c for c in INDICATOR_COLS if c in df_original.columns]

SOLUTIONS = {
    'Nutrition_Deprived':             ('🥗', 'Nutrition', 'Strengthen POSHAN Abhiyaan, mid-day meal schemes, and anganwadi nutrition support programs.'),
    'Child_Mortality_Deprived':       ('👶', 'Child Health', 'Expand immunisation drives, improve ASHA worker coverage, and strengthen primary healthcare.'),
    'Maternal_Health_Deprived':       ('🏥', 'Maternal Health', 'Improve JSY/JSSK maternity schemes and increase institutional delivery infrastructure.'),
    'Years_of_Schooling_Deprived':    ('📘', 'Education Access', 'Increase school enrolment through Samagra Shiksha and reduce dropout rates especially for girls.'),
    'School_Attendance_Deprived':     ('🎒', 'School Attendance', 'Introduce conditional cash transfers and mid-day meals to incentivise regular attendance.'),
    'Cooking_Fuel_Deprived':          ('🔥', 'Clean Cooking Fuel', 'Accelerate PM Ujjwala Yojana LPG connections and promote solar cooking solutions.'),
    'Sanitation_Deprived':            ('🚽', 'Sanitation', 'Strengthen Swachh Bharat Mission Phase II — build and maintain community toilets.'),
    'Drinking_Water_Deprived':        ('💧', 'Safe Drinking Water', 'Expedite Jal Jeevan Mission tap connections and water quality testing infrastructure.'),
    'Electricity_Deprived':           ('⚡', 'Electricity Access', 'Expand DDUGJY/Saubhagya scheme coverage in unelectrified hamlets and tribal areas.'),
    'Housing_Deprived':               ('🏠', 'Housing', 'Scale PM Awas Yojana (Gramin & Urban) for pucca housing for homeless and kaccha-house households.'),
    'Assets_Deprived':                ('📦', 'Household Assets', 'Promote MSME credit access and rural livelihood asset distribution under NRLM.'),
    'Bank_Account_Deprived':          ('🏦', 'Financial Inclusion', 'Drive Jan Dhan Yojana saturation and BC (Banking Correspondent) network expansion in rural areas.'),
}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 10px 0;'>
        <div style='font-family:Playfair Display,serif;font-size:1.1rem;color:#FF9933;font-weight:700;'>
            🇮🇳 MPI Predictor
        </div>
        <div style='font-size:0.75rem;color:#4a5568;margin-top:4px;'>Applied Business Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📌 Navigation**")
    page = st.radio("Navigation", ["🏠 Dashboard", "🔮 Predict Risk", "📊 Compare States", "ℹ️ About"], label_visibility="collapsed")
    st.markdown("---")

    st.markdown("**📋 Dataset Info**")
    st.markdown(f"<div style='font-size:0.8rem;color:#8892b0;'>States/UTs: <b style='color:#FF9933'>{len(df_original)}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.8rem;color:#8892b0;'>Features: <b style='color:#FF9933'>{len(feature_cols)}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.8rem;color:#8892b0;'>Source: <b style='color:#FF9933'>NITI Aayog 2023</b></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#4a5568;'>Thiagarajar School of Management<br>ABA Final Project — 2025</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HERO BANNER (all pages)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">NITI AAYOG MPI 2023 · MACHINE LEARNING · INDIA</div>
    <div class="hero-title">Multidimensional Poverty Risk Predictor</div>
    <p class="hero-sub">Identifying deprivation drivers across Indian States & Union Territories using Random Forest Classification</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    # Top metrics
    low_c  = len(df_original[df_original['Poverty_Risk_Category'] == 'Low'])
    med_c  = len(df_original[df_original['Poverty_Risk_Category'] == 'Medium'])
    high_c = len(df_original[df_original['Poverty_Risk_Category'] == 'High'])
    avg_mpi = df_original['MPI_Value'].mean()

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card"><div class="val">{len(df_original)}</div><div class="lbl">States & UTs</div></div>
        <div class="metric-card"><div class="val" style="color:#e74c3c">{high_c}</div><div class="lbl">High Risk States</div></div>
        <div class="metric-card"><div class="val" style="color:#f39c12">{med_c}</div><div class="lbl">Medium Risk States</div></div>
        <div class="metric-card"><div class="val" style="color:#27ae60">{low_c}</div><div class="lbl">Low Risk States</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">MPI Value by State</div>', unsafe_allow_html=True)
        df_sorted = df_original.sort_values('MPI_Value', ascending=True)
        color_map = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
        fig1 = px.bar(df_sorted, x='MPI_Value', y='State', orientation='h',
                      color='Poverty_Risk_Category',
                      color_discrete_map=color_map,
                      labels={'MPI_Value': 'MPI Value', 'State': ''},
                      height=680)
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            legend_title_text='Risk Category',
            xaxis=dict(gridcolor='#1e3a5f'),
            yaxis=dict(gridcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=10, t=10, b=10),
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig1, width="stretch")

    with col2:
        st.markdown('<div class="section-header">Risk Category Distribution</div>', unsafe_allow_html=True)
        cat_counts = df_original['Poverty_Risk_Category'].value_counts()
        fig2 = px.pie(values=cat_counts.values, names=cat_counts.index,
                      color=cat_counts.index,
                      color_discrete_map=color_map,
                      hole=0.5)
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig2, width="stretch")

        st.markdown('<div class="section-header">Region-wise Avg MPI</div>', unsafe_allow_html=True)
        if 'Region' in df_original.columns:
            reg_mpi = df_original.groupby('Region')['MPI_Value'].mean().sort_values(ascending=False).reset_index()
            fig3 = px.bar(reg_mpi, x='Region', y='MPI_Value',
                          color='MPI_Value', color_continuous_scale='YlOrRd',
                          labels={'MPI_Value': 'Avg MPI'}, height=260)
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                yaxis=dict(gridcolor='#1e3a5f'),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=10)
            )
            st.plotly_chart(fig3, width="stretch")

    # Heatmap
    st.markdown('<div class="section-header">Deprivation Heatmap — All 12 Indicators</div>', unsafe_allow_html=True)
    heat_df = df_original.set_index('State')[INDICATOR_COLS]
    short_labels = [c.replace('_Deprived', '').replace('_', ' ') for c in INDICATOR_COLS]
    fig4 = px.imshow(heat_df.values,
                     x=short_labels,
                     y=heat_df.index.tolist(),
                     color_continuous_scale='YlOrRd',
                     aspect='auto',
                     labels=dict(color='Deprivation %'))
    fig4.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        height=700,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(tickangle=-35)
    )
    st.plotly_chart(fig4, width="stretch")


# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICT RISK
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Predict Risk":

    col_left, col_right = st.columns([1, 1.6], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Select State</div>', unsafe_allow_html=True)
        state = st.selectbox("Select State", df_original['State'].sort_values(), label_visibility="collapsed")
        selected_row = df_original[df_original['State'] == state].iloc[0]

        st.markdown('<div class="section-header">Adjust Indicators</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">💡 Sliders are pre-filled with the selected state\'s actual values. Modify to simulate policy scenarios.</div>', unsafe_allow_html=True)

        inputs = {}
        for col in feature_cols:
            default_val = float(selected_row[col]) if col in selected_row.index else 0.0
            max_val = 100.0 if col not in ['MPI_Value', 'HDI_Proxy'] else 1.0
            label = col.replace('_Deprived', '').replace('_', ' ')
            inputs[col] = st.slider(label, 0.0, max_val, default_val, step=0.1, key=col)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict Poverty Risk", key="predict")

    with col_right:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Always show current state info
        st.markdown('<div class="section-header">State Overview</div>', unsafe_allow_html=True)
        actual_cat = selected_row.get('Poverty_Risk_Category', 'N/A')
        actual_mpi = selected_row.get('MPI_Value', 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("MPI Value", f"{actual_mpi:.3f}")
        m2.metric("Headcount Ratio", f"{selected_row.get('Headcount_Ratio_H', 0):.1f}%")
        m3.metric("Current Category", actual_cat)

        # Radar chart of indicators
        st.markdown('<div class="section-header">Indicator Radar</div>', unsafe_allow_html=True)
        ind_vals = [inputs.get(c, 0) for c in INDICATOR_COLS]
        ind_labels = [c.replace('_Deprived','').replace('_',' ') for c in INDICATOR_COLS]
        fig_radar = go.Figure(go.Scatterpolar(
            r=ind_vals + [ind_vals[0]],
            theta=ind_labels + [ind_labels[0]],
            fill='toself',
            fillcolor='rgba(255,153,51,0.15)',
            line=dict(color='#FF9933', width=2),
            name=state
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='#1e3a5f', color='#8892b0'),
                angularaxis=dict(gridcolor='#1e3a5f', color='#8892b0')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6',
            showlegend=False,
            height=340,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig_radar, width="stretch")

        if predict_btn:
            st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

            if prediction == 0:
                st.markdown("""
                <div class="result-low">
                    <div style="font-size:2.5rem">🟢</div>
                    <div class="result-title" style="color:#27ae60">Low Poverty Risk</div>
                    <p class="result-desc">This state shows relatively good human development outcomes.</p>
                </div>""", unsafe_allow_html=True)
            elif prediction == 1:
                st.markdown("""
                <div class="result-medium">
                    <div style="font-size:2.5rem">🟡</div>
                    <div class="result-title" style="color:#f39c12">Medium Poverty Risk</div>
                    <p class="result-desc">Moderate deprivation — targeted interventions can make a significant impact.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-high">
                    <div style="font-size:2.5rem">🔴</div>
                    <div class="result-title" style="color:#e74c3c">High Poverty Risk</div>
                    <p class="result-desc">Severe multidimensional deprivation — urgent policy action required.</p>
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            st.markdown("<br>", unsafe_allow_html=True)
            labels = ['Low Risk', 'Medium Risk', 'High Risk']
            colors = ['#27ae60', '#f39c12', '#e74c3c']
            fig_prob = go.Figure(go.Bar(
                x=labels, y=[p*100 for p in proba],
                marker_color=colors,
                text=[f'{p*100:.1f}%' for p in proba],
                textposition='outside',
                textfont=dict(color='#ccd6f6')
            ))
            fig_prob.update_layout(
                title=dict(text='Prediction Confidence', font=dict(color='#ccd6f6', size=13)),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#ccd6f6',
                yaxis=dict(range=[0, 110], gridcolor='#1e3a5f', ticksuffix='%'),
                xaxis=dict(gridcolor='rgba(0,0,0,0)'),
                height=260,
                margin=dict(l=0, r=0, t=40, b=10)
            )
            st.plotly_chart(fig_prob, width="stretch")

            # Top 3 contributing factors
            st.markdown('<div class="section-header">Top Deprivation Drivers</div>', unsafe_allow_html=True)
            ind_input = {c: inputs.get(c, 0) for c in INDICATOR_COLS}
            top3 = sorted(ind_input.items(), key=lambda x: x[1], reverse=True)[:3]

            for factor, value in top3:
                sol = SOLUTIONS.get(factor, ('📊', factor, 'Focus on improving this indicator through targeted government schemes.'))
                st.markdown(f"""
                <div class="solution-card">
                    <div class="solution-icon">{sol[0]}</div>
                    <div class="solution-text">
                        <div class="solution-title">{sol[1]} — {value:.1f}% deprived</div>
                        {sol[2]}
                    </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: COMPARE STATES
# ══════════════════════════════════════════════════════════════
elif page == "📊 Compare States":

    st.markdown('<div class="section-header">Compare Two States Side by Side</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    state1 = c1.selectbox("State 1", df_original['State'].sort_values(), index=0)
    state2 = c2.selectbox("State 2", df_original['State'].sort_values(), index=5)

    r1 = df_original[df_original['State'] == state1].iloc[0]
    r2 = df_original[df_original['State'] == state2].iloc[0]

    # Key metrics comparison
    metrics = ['MPI_Value', 'Headcount_Ratio_H', 'Intensity_A', 'HDI_Proxy', 'Avg_Deprivation_Score']
    metrics = [m for m in metrics if m in df_original.columns]

    st.markdown('<div class="section-header">Key Metrics Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        v1 = r1.get(m, 0)
        v2 = r2.get(m, 0)
        delta = round(float(v2) - float(v1), 3)
        cols[i].metric(m.replace('_', ' '), f"{v1:.3f}", f"vs {v2:.3f}", delta_color="inverse")

    # Radar comparison
    st.markdown('<div class="section-header">Indicator Radar Comparison</div>', unsafe_allow_html=True)
    ind1 = [float(r1.get(c, 0)) for c in INDICATOR_COLS]
    ind2 = [float(r2.get(c, 0)) for c in INDICATOR_COLS]
    ind_labels = [c.replace('_Deprived', '').replace('_', ' ') for c in INDICATOR_COLS]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatterpolar(
        r=ind1 + [ind1[0]], theta=ind_labels + [ind_labels[0]],
        fill='toself', name=state1,
        fillcolor='rgba(255,153,51,0.15)',
        line=dict(color='#FF9933', width=2)))
    fig_comp.add_trace(go.Scatterpolar(
        r=ind2 + [ind2[0]], theta=ind_labels + [ind_labels[0]],
        fill='toself', name=state2,
        fillcolor='rgba(52,152,219,0.15)',
        line=dict(color='#3498db', width=2)))
    fig_comp.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0,100], gridcolor='#1e3a5f', color='#8892b0'),
            angularaxis=dict(gridcolor='#1e3a5f', color='#8892b0')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=480,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_comp, width="stretch")

    # Bar comparison
    st.markdown('<div class="section-header">Indicator-by-Indicator Bar Chart</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame({
        'Indicator': ind_labels,
        state1: ind1,
        state2: ind2
    })
    fig_bar = px.bar(comp_df, x='Indicator', y=[state1, state2],
                     barmode='group',
                     color_discrete_sequence=['#FF9933', '#3498db'])
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        xaxis=dict(tickangle=-35, gridcolor='rgba(0,0,0,0)'),
        yaxis=dict(gridcolor='#1e3a5f'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=380,
        margin=dict(l=0, r=0, t=10, b=80)
    )
    st.plotly_chart(fig_bar, width="stretch")


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>Title:</b> Predicting Multidimensional Poverty Vulnerability Across Indian States Using Machine Learning<br><br>
    <b>Course:</b> Applied Business Analytics (ABA) — Thiagarajar School of Management, Madurai<br><br>
    <b>Dataset:</b> NITI Aayog National MPI 2023 — 36 States & UTs × 12 Deprivation Indicators<br><br>
    <b>Models Used:</b> Random Forest Classifier (200 trees) + K-Means Clustering (K=3)<br><br>
    <b>Objective:</b> Move beyond descriptive MPI reporting to a predictive, policy-prescriptive analytics framework.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Importances (Model Explainability)</div>', unsafe_allow_html=True)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    imp_sorted = importances.sort_values(ascending=True).tail(12)
    short = [c.replace('_Deprived','').replace('_',' ') for c in imp_sorted.index]
    fig_imp = px.bar(x=imp_sorted.values, y=short, orientation='h',
                     color=imp_sorted.values, color_continuous_scale='YlOrRd',
                     labels={'x': 'Importance Score', 'y': ''})
    fig_imp.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ccd6f6',
        coloraxis_showscale=False,
        xaxis=dict(gridcolor='#1e3a5f'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        height=420,
        margin=dict(l=0, r=0, t=10, b=10)
    )
    st.plotly_chart(fig_imp, width="stretch")

    st.markdown('<div class="section-header">MPI Indicator Definitions</div>', unsafe_allow_html=True)
    defs = {
        'Nutrition': 'Any adult/child in household is undernourished',
        'Child Mortality': 'Any child under 18 has died in the family',
        'Maternal Health': 'Births without skilled attendance or ANC',
        'Years of Schooling': 'No household member completed 6 years of schooling',
        'School Attendance': 'Any school-age child not attending school',
        'Cooking Fuel': 'Household uses solid/polluting cooking fuel',
        'Sanitation': 'No access to improved sanitation facility',
        'Drinking Water': 'No access to safe drinking water within 30 min',
        'Electricity': 'Household has no electricity connection',
        'Housing': 'House has inadequate floor/roof/wall materials',
        'Assets': 'Household lacks basic assets (TV, phone, etc.)',
        'Bank Account': 'No household member has a bank/financial account'
    }
    for ind, defn in defs.items():
        st.markdown(f"<div class='solution-card'><div class='solution-text'><b style='color:#FF9933'>{ind}:</b> {defn}</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    ABA Final Project · Thiagarajar School of Management · NITI Aayog MPI 2023 · Random Forest Classifier<br>
    Built with Streamlit · For academic and policy research purposes only
</div>
""", unsafe_allow_html=True)
