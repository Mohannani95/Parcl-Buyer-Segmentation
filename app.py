# ============================================================
# PARCL REAL ESTATE — BUYER SEGMENTATION DASHBOARD
# app.py | Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# (Must be the very first Streamlit command!)
# ============================================================
st.set_page_config(
    page_title   = "Parcl Buyer Intelligence",
    page_icon    = "🏠",
    layout       = "wide",           # Use full screen width
    initial_sidebar_state = "expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# Makes the dashboard look professional
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8F9FA; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A237E;
        color: white;
    }

    /* Title styling */
    h1 { color: #1A237E; }
    h2 { color: #283593; }
    h3 { color: #3949AB; }

    /* Segment color badges */
    .badge-c1 { background:#2196F3; color:white;
                padding:3px 10px; border-radius:12px; }
    .badge-c2 { background:#4CAF50; color:white;
                padding:3px 10px; border-radius:12px; }
    .badge-c3 { background:#FF9800; color:white;
                padding:3px 10px; border-radius:12px; }
    .badge-c4 { background:#9C27B0; color:white;
                padding:3px 10px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data   # Cache = load once, reuse every time (faster!)
def load_data():
    df = pd.read_csv('buyers_segmented.csv')
    return df

df = load_data()

# Segment color mapping
SEGMENT_COLORS = {
    'C1 - Global Investors'  : '#2196F3',
    'C2 - First-Time Buyers' : '#4CAF50',
    'C3 - Corporate Buyers'  : '#FF9800',
    'C4 - Luxury Investors'  : '#9C27B0'
}

SEGMENT_ORDER = [
    'C1 - Global Investors',
    'C2 - First-Time Buyers',
    'C3 - Corporate Buyers',
    'C4 - Luxury Investors'
]

# ============================================================
# SIDEBAR — FILTERS
# ============================================================
st.sidebar.image(
    "https://parcl.co/favicon.ico",
    width=50
)
st.sidebar.title("🏠 Parcl Analytics")
st.sidebar.markdown("**Buyer Segmentation Intelligence**")
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Buyers")

# Filter 1: Buyer Segment
all_segments = ['All Segments'] + SEGMENT_ORDER
selected_segment = st.sidebar.selectbox(
    "Buyer Segment",
    all_segments,
    help="Select a specific buyer segment to focus on"
)

# Filter 2: Country
all_countries = ['All Countries'] + sorted(df['country'].unique().tolist())
selected_country = st.sidebar.selectbox(
    "Country",
    all_countries,
    help="Filter by buyer's country of residence"
)

# Filter 3: Acquisition Purpose
all_purposes = ['All'] + sorted(df['acquisition_purpose'].unique().tolist())
selected_purpose = st.sidebar.selectbox(
    "Acquisition Purpose",
    all_purposes,
    help="Filter by reason for purchase"
)

# Filter 4: Client Type
all_types = ['All'] + sorted(df['client_type'].unique().tolist())
selected_type = st.sidebar.selectbox(
    "Client Type",
    all_types,
    help="Filter by Individual or Company"
)

# Filter 5: Age Range Slider
min_age = int(df['age'].min())
max_age = int(df['age'].max())
age_range = st.sidebar.slider(
    "Age Range",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age),
    help="Filter buyers by age"
)

st.sidebar.markdown("---")

# ── Apply all filters ──────────────────────────────────────
filtered = df.copy()

if selected_segment != 'All Segments':
    filtered = filtered[filtered['buyer_segment'] == selected_segment]
if selected_country != 'All Countries':
    filtered = filtered[filtered['country'] == selected_country]
if selected_purpose != 'All':
    filtered = filtered[filtered['acquisition_purpose'] == selected_purpose]
if selected_type != 'All':
    filtered = filtered[filtered['client_type'] == selected_type]

filtered = filtered[
    (filtered['age'] >= age_range[0]) &
    (filtered['age'] <= age_range[1])
]

# Show filter result count in sidebar
st.sidebar.info(f"📊 Showing **{len(filtered):,}** of **{len(df):,}** buyers")

# ============================================================
# MAIN HEADER
# ============================================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🏠")
with col_title:
    st.title("Parcl Real Estate — Buyer Intelligence Dashboard")

st.markdown(
    "**AI-Powered Buyer Segmentation** | "
    "ML K-Means Clustering | "
    f"Dataset: {len(df):,} Buyers | "
    "4 Distinct Segments Identified"
)
st.markdown("---")

# ============================================================
# SECTION 1 — KPI METRICS
# ============================================================
st.subheader("📊 Key Performance Indicators")

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric(
        label    = "Total Buyers",
        value    = f"{len(filtered):,}",
        delta    = f"{len(filtered)/len(df)*100:.1f}% of total"
    )
with k2:
    avg_price = filtered['avg_sale_price'].mean()
    st.metric(
        label = "Avg Property Price",
        value = f"${avg_price:,.0f}"
    )
with k3:
    inv_pct = (filtered['acquisition_purpose'] == 'Investment').mean() * 100
    st.metric(
        label = "Investment Purpose",
        value = f"{inv_pct:.1f}%"
    )
with k4:
    loan_pct = (filtered['loan_applied'] == 'Yes').mean() * 100
    st.metric(
        label = "Loan Rate",
        value = f"{loan_pct:.1f}%"
    )
with k5:
    avg_sat = filtered['satisfaction_score'].mean()
    st.metric(
        label = "Avg Satisfaction",
        value = f"{avg_sat:.2f}/5"
    )
with k6:
    avg_props = filtered['num_properties'].mean()
    st.metric(
        label = "Avg Properties",
        value = f"{avg_props:.1f}"
    )

st.markdown("---")

# ============================================================
# SECTION 2 — BUYER SEGMENT OVERVIEW
# ============================================================
st.subheader("🎯 Buyer Segmentation Overview")

col1, col2 = st.columns(2)

with col1:
    # Pie chart
    seg_counts = filtered['buyer_segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']

    fig_pie = px.pie(
        seg_counts,
        values = 'Count',
        names  = 'Segment',
        title  = 'Buyer Segment Distribution',
        color  = 'Segment',
        color_discrete_map = SEGMENT_COLORS,
        hole   = 0.3   # Donut style
    )
    fig_pie.update_traces(
        textposition = 'inside',
        textinfo     = 'percent+label',
        pull         = [0.05] * len(seg_counts)
    )
    fig_pie.update_layout(
        showlegend   = True,
        height       = 420,
        margin       = dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Avg spend by segment
    spend_seg = filtered.groupby('buyer_segment')['avg_sale_price'] \
                        .mean().reset_index()
    spend_seg.columns = ['Segment', 'Avg Price']
    spend_seg = spend_seg.sort_values('Avg Price', ascending=True)

    fig_spend = px.bar(
        spend_seg,
        x           = 'Avg Price',
        y           = 'Segment',
        orientation = 'h',
        title       = 'Average Property Price by Segment',
        color       = 'Segment',
        color_discrete_map = SEGMENT_COLORS,
        text        = 'Avg Price'
    )
    fig_spend.update_traces(
        texttemplate = '$%{text:,.0f}',
        textposition = 'outside'
    )
    fig_spend.update_layout(
        showlegend = False,
        height     = 420,
        xaxis_title = 'Average Sale Price ($)',
        yaxis_title = '',
        margin      = dict(t=40, b=20, l=20, r=80)
    )
    st.plotly_chart(fig_spend, use_container_width=True)

st.markdown("---")

# ============================================================
# SECTION 3 — INVESTOR BEHAVIOR DASHBOARD
# ============================================================
st.subheader("💼 Investor Behavior Analysis")

col1, col2 = st.columns(2)

with col1:
    # Loan rate by segment
    loan_data = filtered.groupby('buyer_segment')['loan_applied'] \
                        .apply(lambda x: (x=='Yes').mean()*100) \
                        .reset_index()
    loan_data.columns = ['Segment', 'Loan Rate (%)']

    fig_loan = px.bar(
        loan_data,
        x     = 'Segment',
        y     = 'Loan Rate (%)',
        title = 'Loan Application Rate by Segment',
        color = 'Segment',
        color_discrete_map = SEGMENT_COLORS,
        text  = 'Loan Rate (%)'
    )
    fig_loan.update_traces(
        texttemplate = '%{text:.1f}%',
        textposition = 'outside'
    )
    fig_loan.update_layout(
        showlegend  = False,
        height      = 400,
        xaxis_title = '',
        yaxis_title = 'Loan Rate (%)',
        margin      = dict(t=40, b=20)
    )
    st.plotly_chart(fig_loan, use_container_width=True)

with col2:
    # Investment vs Home grouped bar
    purpose_data = filtered.groupby(
        ['buyer_segment', 'acquisition_purpose']
    ).size().reset_index(name='Count')

    fig_purpose = px.bar(
        purpose_data,
        x        = 'buyer_segment',
        y        = 'Count',
        color    = 'acquisition_purpose',
        barmode  = 'group',
        title    = 'Investment vs Home Purchase by Segment',
        color_discrete_map = {
            'Investment' : '#FF6B6B',
            'Home'       : '#4ECDC4'
        },
        text     = 'Count'
    )
    fig_purpose.update_traces(textposition='outside')
    fig_purpose.update_layout(
        height      = 400,
        xaxis_title = '',
        yaxis_title = 'Number of Buyers',
        legend_title = 'Purpose',
        margin      = dict(t=40, b=20)
    )
    st.plotly_chart(fig_purpose, use_container_width=True)

st.markdown("---")

# ============================================================
# SECTION 4 — GEOGRAPHIC ANALYSIS
# ============================================================
st.subheader("🌍 Geographic Buyer Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    geo_data = filtered.groupby(
        ['country', 'buyer_segment']
    ).size().reset_index(name='Count')

    fig_geo = px.bar(
        geo_data,
        x        = 'country',
        y        = 'Count',
        color    = 'buyer_segment',
        barmode  = 'stack',
        title    = 'Buyer Segments by Country',
        color_discrete_map = SEGMENT_COLORS
    )
    fig_geo.update_layout(
        height      = 420,
        xaxis_title = 'Country',
        yaxis_title = 'Number of Buyers',
        legend_title = 'Segment',
        xaxis_tickangle = -30,
        margin      = dict(t=40, b=60)
    )
    st.plotly_chart(fig_geo, use_container_width=True)

with col2:
    # Top country stats
    st.markdown("**Top 5 Countries by Buyer Count**")
    top_countries = filtered['country'].value_counts().head(5)

    for i, (country, count) in enumerate(top_countries.items(), 1):
        pct = count / len(filtered) * 100
        st.markdown(f"**{i}. {country}**")
        st.progress(pct/100)
        st.caption(f"{count} buyers ({pct:.1f}%)")

st.markdown("---")

# ============================================================
# SECTION 5 — SEGMENT INSIGHTS PANEL
# ============================================================
st.subheader("🔍 Segment Insights Panel")
st.markdown("Click on each segment to expand detailed statistics")

for segment in SEGMENT_ORDER:
    seg_data = filtered[filtered['buyer_segment'] == segment]

    if len(seg_data) == 0:
        continue  # Skip if no buyers in this segment after filtering

    color = SEGMENT_COLORS[segment]

    with st.expander(f"📋 {segment} — {len(seg_data):,} buyers"):

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Total Buyers",    f"{len(seg_data):,}")
            st.metric("Avg Age",         f"{seg_data['age'].mean():.0f} yrs")

        with m2:
            st.metric("Avg Satisfaction", f"{seg_data['satisfaction_score'].mean():.2f}/5")
            inv_r = (seg_data['acquisition_purpose']=='Investment').mean()*100
            st.metric("Investment Rate",  f"{inv_r:.1f}%")

        with m3:
            st.metric("Avg Property Price", f"${seg_data['avg_sale_price'].mean():,.0f}")
            loan_r = (seg_data['loan_applied']=='Yes').mean()*100
            st.metric("Loan Rate",          f"{loan_r:.1f}%")

        with m4:
            st.metric("Avg Properties",    f"{seg_data['num_properties'].mean():.1f}")
            st.metric("Avg Total Spend",   f"${seg_data['total_spend'].mean():,.0f}")

        # Top countries
        st.markdown("**Top Countries:**")
        top_c = seg_data['country'].value_counts().head(5).reset_index()
        top_c.columns = ['Country', 'Buyers']

        fig_c = px.bar(
            top_c,
            x='Country', y='Buyers',
            color_discrete_sequence=[color],
            height=250
        )
        fig_c.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                             showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

st.markdown("---")

# ============================================================
# SECTION 6 — SATISFACTION ANALYSIS
# ============================================================
st.subheader("⭐ Customer Satisfaction Analysis")

col1, col2 = st.columns(2)

with col1:
    # Box plot of satisfaction by segment
    fig_box = px.box(
        filtered,
        x     = 'buyer_segment',
        y     = 'satisfaction_score',
        color = 'buyer_segment',
        title = 'Satisfaction Score Distribution',
        color_discrete_map = SEGMENT_COLORS
    )
    fig_box.update_layout(
        showlegend  = False,
        height      = 380,
        xaxis_title = '',
        yaxis_title = 'Satisfaction Score (1-5)',
        margin      = dict(t=40, b=20)
    )
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    # Referral channel breakdown
    ref_data = filtered.groupby(
        ['buyer_segment','referral_channel']
    ).size().reset_index(name='Count')

    fig_ref = px.bar(
        ref_data,
        x        = 'buyer_segment',
        y        = 'Count',
        color    = 'referral_channel',
        barmode  = 'group',
        title    = 'Referral Channel by Segment',
        color_discrete_sequence = ['#FF9800','#9C27B0','#2196F3']
    )
    fig_ref.update_layout(
        height      = 380,
        xaxis_title = '',
        yaxis_title = 'Number of Buyers',
        legend_title = 'Channel',
        margin      = dict(t=40, b=20)
    )
    st.plotly_chart(fig_ref, use_container_width=True)

st.markdown("---")

# ============================================================
# SECTION 7 — RAW DATA TABLE
# ============================================================
st.subheader("📄 Buyer Data Table")

# Column selector
display_cols = st.multiselect(
    "Select columns to display:",
    options = ['client_id','client_type','first_name','last_name',
               'gender','age','country','region',
               'acquisition_purpose','loan_applied',
               'satisfaction_score','num_properties',
               'avg_sale_price','total_spend','buyer_segment'],
    default = ['client_id','client_type','age','country',
               'acquisition_purpose','loan_applied',
               'satisfaction_score','avg_sale_price','buyer_segment']
)

# Row limit selector
n_rows = st.slider("Number of rows to display:", 10, 200, 50)

if display_cols:
    st.dataframe(
        filtered[display_cols].head(n_rows),
        use_container_width=True,
        height=400
    )

    # Download button
    csv_data = filtered[display_cols].to_csv(index=False)
    st.download_button(
        label     = "⬇️ Download Filtered Data as CSV",
        data      = csv_data,
        file_name = "parcl_filtered_buyers.csv",
        mime      = "text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:12px;'>
    🏠 Parcl Real Estate Buyer Intelligence Dashboard |
    Built with Python & Streamlit |
    ML-Based K-Means Clustering (K=4) |
    Dataset: 2,000 Buyers × 10,000 Properties
    </div>
    """,
    unsafe_allow_html=True
)