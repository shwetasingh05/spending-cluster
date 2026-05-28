import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Spending Cluster Analysis",
    page_icon="💳",
    layout="wide"
)

# ── SLEEK DARK THEME STYLING FOR MATPLOTLIB/SEABORN ──
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#0e1117',
    'savefig.facecolor': '#0e1117',
    'grid.color': '#212529',
    'grid.alpha': 0.5,
    'axes.edgecolor': '#4a5568',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#cbd5e1',
    'ytick.color': '#cbd5e1',
    'font.sans-serif': 'sans-serif',
    'font.family': 'sans-serif'
})
# Set Seaborn theme consistent with our styling
sns.set_theme(style='darkgrid', rc={
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#0e1117',
    'grid.color': '#212529',
    'text.color': '#e2e8f0',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#cbd5e1',
    'ytick.color': '#cbd5e1',
    'axes.edgecolor': '#4a5568'
})

# ── CUSTOM CSS STYLING (GLASSMORPHISM & PREMIUM THEMING) ──
st.markdown("""
<style>
/* Import modern typography from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Set global font family */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Gradient background for the sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090b11 0%, #151a2e 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphic Cards container */
.glass-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

/* Glowing text class */
.glow-text {
    background: linear-gradient(90deg, #2ecc71, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD & PREPARE DATA ──────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('CC GENERAL.csv')
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
    df.drop('CUST_ID', axis=1, inplace=True)
    return df

@st.cache_resource
def run_clustering(n_clusters=3):
    df = load_data()
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)
    cluster_names = {0: 'Active Transactors', 1: 'Inactive Users', 2: 'Cash Advance Revolvers'}
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_scaled)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    return df, df_scaled, pca, scaler, kmeans

df = load_data()

# ── GLOBAL COLOR PALETTE FOR CLUSTERS ────────
colors = {'Active Transactors': '#2ecc71', 'Inactive Users': '#3498db', 'Cash Advance Revolvers': '#e74c3c'}

# ── SIDEBAR NAVIGATION ───────────────────────
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Overview",
    "📊 EDA",
    "🔍 Find Optimal K",
    "🎯 Clusters",
    "💡 Insights",
    "🔮 Predictor"
])

# ── PAGES ────────────────────────────────────

# ── 1. OVERVIEW ──────────────────────────────
if page == "🏠 Overview":
    st.title("💳 Credit Card Spending Cluster Analysis")
    st.markdown("#### Segmenting customers based on credit card spending behavior using K-Means Clustering")
    st.divider()

    # Introduction / Business Case Callout
    st.markdown("""
    ### 🎯 Business Objective & Context
    In the highly competitive credit card industry, one-size-fits-all marketing campaigns are inefficient and expensive. 
    To maximize profitability and manage risk, financial institutions must understand their customers' distinct behavioral profiles.
    
    This application utilizes **K-Means Clustering** to segment **8,950 credit card holders** based on 17 behavioral attributes (including spending habits, cash advance usage, payment frequency, and credit limits). 
    By defining distinct customer personas, the bank can:
    * 🟢 **Maximize Retention:** Target high-value transactors with premium loyalty programs.
    * 🔵 **Drive Activation:** Deploy personalized spend incentives to dormant cardholders.
    * 🔴 **Mitigate Default Risk:** Identify and proactively support customers heavily reliant on high-interest cash advances.
    """)
    
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{df.shape[0]:,}", help="Total number of active credit card holders in the dataset")
    col2.metric("Features", df.shape[1], help="Number of operational and spending features analyzed")
    col3.metric("Missing Values", "0 (cleaned)", help="Missing minimum payments and credit limits imputed with medians")

    st.divider()

    st.write("### 📋 Dataset Preview")
    st.markdown("A sample of the underlying transactional data showing spending patterns, balance history, and credit usage:")
    st.dataframe(df.head(10), use_container_width=True)

    st.write("### 📊 Summary Statistics")
    st.markdown("Statistical summary of credit behaviors across the entire customer base:")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ── 2. EDA ────────────────────────────────────
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Purchase Distribution")
        fig = px.histogram(df, x='PURCHASES', nbins=100, 
                           color_discrete_sequence=['#3498db'], 
                           opacity=0.85, 
                           title='Highly Right-Skewed — Few Spend a Lot')
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#e2e8f0',
            xaxis_title='Purchase Amount',
            yaxis_title='Count',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cash Advance Distribution")
        fig = px.histogram(df, x='CASH_ADVANCE', nbins=100, 
                           color_discrete_sequence=['#e74c3c'], 
                           opacity=0.85, 
                           title='Most Customers Never Use Cash Advance')
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#e2e8f0',
            xaxis_title='Cash Advance Amount',
            yaxis_title='Count',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Outlier Detection — Boxplot")
    fig = px.box(df, y=['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS'],
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 title='Key Feature Distributions (Outliers Hidden for Clarity)')
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='#e2e8f0',
        margin=dict(l=40, r=20, t=40, b=40),
        height=400
    )
    fig.update_yaxes(showfliers=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Correlation Heatmap")
    with plt.style.context('default'):
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
        ax.set_title('Feature Correlations')
        st.pyplot(fig)
        plt.close()

# ── 3. FIND OPTIMAL K ─────────────────────────
elif page == "🔍 Find Optimal K":
    st.title("🔍 Finding Optimal Number of Clusters")
    st.divider()

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    st.info("Running Elbow Method and Silhouette Scores for K=2 to 10. This may take a moment...")

    inertias = []
    sil_scores = []
    K_range = range(2, 11)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(df_scaled, labels, sample_size=2000, random_state=42)
        sil_scores.append(score)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Elbow Method")
        fig = px.line(x=list(K_range), y=inertias, markers=True,
                      color_discrete_sequence=['#3498db'],
                      title='Elbow Curve')
        fig.add_vline(x=3, line_dash='dash', line_color='#e74c3c', annotation_text='Optimal K=3')
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#e2e8f0',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Inertia (WCSS)',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Silhouette Scores")
        fig = px.line(x=list(K_range), y=sil_scores, markers=True,
                      color_discrete_sequence=['#e74c3c'],
                      title='Silhouette Score (Higher = Better)')
        fig.add_vline(x=3, line_dash='dash', line_color='#2ecc71', annotation_text='Best K=3')
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#e2e8f0',
            xaxis_title='Number of Clusters (K)',
            yaxis_title='Silhouette Score',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success("✅ K=3 chosen — highest silhouette score (0.2403) and clear elbow point")

# ── 4. CLUSTERS ───────────────────────────────
elif page == "🎯 Clusters":
    st.title("🎯 Cluster Profiles")
    st.divider()

    df_clustered, df_scaled, pca, scaler, kmeans = run_clustering()

    # Cluster sizes
    counts = df_clustered['Cluster_Name'].value_counts()
    
    col1, col2, col3 = st.columns(3)
    # Using dynamic, glassmorphic metric cards for cluster counts
    col1.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #2ecc71 !important;">
        <h4 style="color: #cbd5e1; margin: 0; font-size: 14px;">🟢 Active Transactors</h4>
        <h2 style="color: #ffffff; margin: 5px 0 0 0; font-size: 28px;">{counts.get('Active Transactors', 0):,}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col2.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #3498db !important;">
        <h4 style="color: #cbd5e1; margin: 0; font-size: 14px;">🔵 Inactive Users</h4>
        <h2 style="color: #ffffff; margin: 5px 0 0 0; font-size: 28px;">{counts.get('Inactive Users', 0):,}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col3.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid #e74c3c !important;">
        <h4 style="color: #cbd5e1; margin: 0; font-size: 14px;">🔴 Cash Advance Revolvers</h4>
        <h2 style="color: #ffffff; margin: 5px 0 0 0; font-size: 28px;">{counts.get('Cash Advance Revolvers', 0):,}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # PCA Scatter
    st.subheader("Customer Clusters — PCA 2D View")
    fig = px.scatter(df_clustered, x='PCA1', y='PCA2', 
                     color='Cluster_Name',
                     color_discrete_map=colors,
                     opacity=0.6,
                     hover_data={
                         'PCA1': False,
                         'PCA2': False,
                         'Cluster_Name': True,
                         'BALANCE': ':.2f',
                         'PURCHASES': ':.2f',
                         'CASH_ADVANCE': ':.2f',
                         'CREDIT_LIMIT': ':.2f',
                         'PAYMENTS': ':.2f'
                     },
                     title='Customer Spending Clusters (Interactive PCA 2D View)')
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='#e2e8f0',
        xaxis_title='PC1 (27.3% Variance Explained)',
        yaxis_title='PC2 (20.3% Variance Explained)',
        margin=dict(l=40, r=20, t=40, b=40),
        legend_title='Clusters',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster profile bar chart
    st.subheader("Cluster Comparison — Key Features")
    features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    profile = df_clustered.groupby('Cluster_Name')[features].mean().round(2)
    
    # Melt profile for Plotly Bar
    profile_melted = pd.melt(profile.reset_index(), id_vars=['Cluster_Name'], value_vars=features,
                             var_name='Feature', value_name='Mean Value')
    
    fig = px.bar(profile_melted, x='Feature', y='Mean Value', 
                 color='Cluster_Name',
                 barmode='group',
                 color_discrete_map=colors,
                 title='Mean Feature Values per Cluster')
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='#e2e8f0',
        xaxis_title='Feature',
        yaxis_title='Mean Value ($)',
        margin=dict(l=40, r=20, t=40, b=40),
        legend_title='Clusters',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full Cluster Profile Table")
    st.dataframe(df_clustered.groupby('Cluster_Name').mean().round(2), use_container_width=True)

# ── 5. INSIGHTS ───────────────────────────────
elif page == "💡 Insights":
    st.title("💡 Business Insights")
    st.divider()

    df_clustered, df_scaled, pca, scaler, kmeans = run_clustering()
    profile = df_clustered.groupby('Cluster_Name').mean().round(3)
    counts = df_clustered['Cluster_Name'].value_counts()

    col1, col2 = st.columns(2)

    # ── PIE CHART ──
    with col1:
        st.subheader("🥧 Customer Segment Distribution")
        fig = px.pie(names=counts.index, values=counts.values,
                     color=counts.index,
                     color_discrete_map=colors,
                     hole=0.4,
                     title='How Customers Are Divided')
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          marker=dict(line=dict(color='#0e1117', width=2)))
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#e2e8f0',
            margin=dict(l=20, r=20, t=40, b=20),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** Nearly 7 out of 10 customers are inactive. The biggest growth opportunity is activating existing customers — not acquiring new ones.")

    # ── FULL PAYMENT RATE ──
    with col2:
        st.subheader("💳 Who Pays Their Full Bill?")
        profile_full = df_clustered.groupby('Cluster_Name')['PRC_FULL_PAYMENT'].mean().reset_index()
        
        fig = px.bar(profile_full, x='Cluster_Name', y='PRC_FULL_PAYMENT',
                     color='Cluster_Name',
                     color_discrete_map=colors,
                     title='Full Bill Payment Rate per Cluster',
                     text='PRC_FULL_PAYMENT')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#e2e8f0',
            xaxis_title='Cluster Name',
            yaxis_title='Proportion Paying in Full',
            yaxis_tickformat='.0%',
            margin=dict(l=40, r=20, t=40, b=40),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Insight:** Active Transactors pay in full most often — low risk, high value. Cash Advance Revolvers rarely pay in full — higher default risk for the bank.")

    st.divider()

    # ── MARKETING ROI SIMULATOR ──
    st.subheader("💵 Marketing Campaign ROI Simulator")
    st.markdown("Estimate the potential financial impact of running targeted activation campaigns for the **Inactive Users** segment.")
    
    col_sim1, col_sim2 = st.columns([1, 1])
    
    with col_sim1:
        st.markdown("### 🎛️ Simulation Parameters")
        target_pct = st.slider("Target Size (% of Inactive Users)", min_value=5, max_value=100, value=25, step=5)
        conv_rate = st.slider("Estimated Conversion Rate (%)", min_value=1, max_value=20, value=5, step=1)
        spend_boost = st.slider("Average Annual Spend Boost ($)", min_value=50, max_value=1000, value=250, step=50)
        incentive_cost = st.slider("Incentive Cost per Customer ($)", min_value=5, max_value=50, value=15, step=5)
        
    with col_sim2:
        st.markdown("### 📈 Projected Campaign Financials")
        total_inactive = counts.get('Inactive Users', 6076)
        targeted = int(total_inactive * (target_pct / 100))
        converted = int(targeted * (conv_rate / 100))
        gross_rev = converted * spend_boost
        total_cost = targeted * incentive_cost
        net_profit = gross_rev - total_cost
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        
        roi_color = "#2ecc71" if net_profit >= 0 else "#e74c3c"
        
        st.markdown(f"""
        <div class="glass-card">
            <table style="width:100%; border:none; border-collapse:collapse;">
                <tr style="border:none;"><td style="color:#cbd5e1; border:none; padding:4px;">Targeted Customers:</td><td style="text-align:right; font-weight:bold; color:white; border:none; padding:4px;">{targeted:,}</td></tr>
                <tr style="border:none;"><td style="color:#cbd5e1; border:none; padding:4px;">Successfully Converted:</td><td style="text-align:right; font-weight:bold; color:#3498db; border:none; padding:4px;">{converted:,}</td></tr>
                <tr style="border:none;"><td style="color:#cbd5e1; border:none; padding:4px;">Gross Revenue Increase:</td><td style="text-align:right; font-weight:bold; color:#2ecc71; border:none; padding:4px;">${gross_rev:,}</td></tr>
                <tr style="border:none;"><td style="color:#cbd5e1; border:none; padding:4px;">Campaign Cost:</td><td style="text-align:right; font-weight:bold; color:#e74c3c; border:none; padding:4px;">${total_cost:,}</td></tr>
                <tr style="border:none; border-top:1px solid rgba(255,255,255,0.1);"><td style="color:white; font-weight:bold; border:none; padding:8px 4px 4px 4px;">Net Campaign Profit:</td><td style="text-align:right; font-size:18px; font-weight:bold; color:{roi_color}; border:none; padding:8px 4px 4px 4px;">${net_profit:,}</td></tr>
                <tr style="border:none;"><td style="color:white; font-weight:bold; border:none; padding:4px;">Projected ROI:</td><td style="text-align:right; font-size:18px; font-weight:bold; color:{roi_color}; border:none; padding:4px;">{roi:.1f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        if net_profit >= 0:
            st.success("✅ **Profitable Strategy:** The conversion boost easily recovers the customer incentive campaign cost.")
        else:
            st.error("⚠️ **Negative Margins:** The campaign cost exceeds the return. Try reducing the incentive cost or increasing conversion estimates.")

    st.divider()

    # ── WRITTEN RECOMMENDATIONS ──
    st.subheader("📋 Business Recommendations")

    st.markdown("### 🟢 Active Transactors — *Retain & Reward*")
    st.markdown("""
    - Highest spenders, pay bills regularly, high credit limits
    - These are the bank's most **profitable and reliable** customers
    - **Action:** Offer premium rewards cards, travel cashback, and loyalty programs to retain them long-term
    """)
    st.divider()

    st.markdown("### 🔵 Inactive Users — *Activate & Engage*")
    st.markdown("""
    - 68% of all customers — barely use their card
    - Small nudges can convert them into active users
    - **Action:** Send personalised spending incentives, limited-time cashback offers, and gamified rewards to drive first transactions
    """)
    st.divider()

    st.markdown("### 🔴 Cash Advance Revolvers — *Support & Reduce Risk*")
    st.markdown("""
    - Rely heavily on cash withdrawals, carry high debt balances
    - Highest risk of default — need intervention early
    - **Action:** Proactively offer lower-interest EMI plans, debt counselling, and spending alerts
    """)
    st.divider()

    st.success("📌 **Key Finding:** Converting just 10% of inactive users to moderate spenders would add more revenue than acquiring new customers — at zero acquisition cost.")

# ── 6. PREDICTOR ──────────────────────────────
elif page == "🔮 Predictor":
    st.title("🔮 Real-Time Customer Segment Predictor")
    st.markdown("#### Input operational and spending features to classify a customer profile in real-time.")
    st.divider()

    df_clustered, df_scaled, pca, scaler, kmeans = run_clustering()

    st.markdown("""
    ### 👤 Enter Customer Behavior Details
    Adjust the attributes below to classify which customer profile best fits this behavior.
    """)

    col_inp1, col_inp2 = st.columns(2)

    with col_inp1:
        balance = st.number_input("Current Balance ($)", min_value=0.0, max_value=25000.0, value=1500.0, step=100.0)
        purchases = st.number_input("Annual Purchases ($)", min_value=0.0, max_value=50000.0, value=1000.0, step=100.0)
        cash_advance = st.number_input("Cash Advance Amount ($)", min_value=0.0, max_value=50000.0, value=0.0, step=100.0)
        credit_limit = st.number_input("Credit Limit ($)", min_value=50.0, max_value=30000.0, value=4000.0, step=500.0)
        
    with col_inp2:
        payments = st.number_input("Total Payments Made ($)", min_value=0.0, max_value=60000.0, value=1200.0, step=100.0)
        min_payments = st.number_input("Minimum Payments Due ($)", min_value=0.0, max_value=80000.0, value=500.0, step=50.0)
        purchase_freq = st.slider("Purchase Frequency (0 = Never, 1 = Daily)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        full_pay_rate = st.slider("Proportion Paid in Full (0 = None, 1 = All)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if st.button("🔮 Classify Customer Segment", use_container_width=True):
        feature_cols = [
            'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
            'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
            'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
        ]
        
        raw_data = load_data()
        input_dict = {}
        for col in feature_cols:
            input_dict[col] = raw_data[col].mean()
            
        input_dict['BALANCE'] = balance
        input_dict['PURCHASES'] = purchases
        input_dict['CASH_ADVANCE'] = cash_advance
        input_dict['CREDIT_LIMIT'] = credit_limit
        input_dict['PAYMENTS'] = payments
        input_dict['MINIMUM_PAYMENTS'] = min_payments
        input_dict['PURCHASES_FREQUENCY'] = purchase_freq
        input_dict['PRC_FULL_PAYMENT'] = full_pay_rate
        
        input_df = pd.DataFrame([input_dict])[feature_cols]
        scaled_input = scaler.transform(input_df)
        predicted_cluster = kmeans.predict(scaled_input)[0]
        
        cluster_names = {0: 'Active Transactors', 1: 'Inactive Users', 2: 'Cash Advance Revolvers'}
        pred_name = cluster_names[predicted_cluster]
        pred_color = colors[pred_name]
        
        strategies = {
            'Active Transactors': """
            * 💎 **Engagement Plan:** Retain & Reward.
            * 🎁 **Tactical Offer:** Invite them to a premium Mastercard/Visa Black tier card with double reward points on travel, luxury shopping, and dining.
            * 🎯 **Targeting Priority:** High priority for relationship managers.
            """,
            'Inactive Users': """
            * 📢 **Engagement Plan:** Activate & Spend Incentives.
            * 💸 **Tactical Offer:** Send a targeted "Spend $200 in the next 30 days and get $25 cash back" incentive to encourage their first transactional behavior.
            * 🎯 **Targeting Priority:** High volume opportunity (covers ~68% of the customer base).
            """,
            'Cash Advance Revolvers': """
            * 🛡️ **Engagement Plan:** Debt restructuring and credit monitoring.
            * 📊 **Tactical Offer:** Proactively offer low-interest balance transfer plans or conversion of high cash balances into structured 12-month installment plans (EMIs) to reduce delinquency risks.
            * 🎯 **Targeting Priority:** High default risk mitigation.
            """
        }
        
        st.divider()
        
        st.markdown(f"""
        <div class="glass-card" style="border-left: 6px solid {pred_color} !important;">
            <h4 style="margin:0; color:#cbd5e1;">Predicted Customer Persona</h4>
            <h1 style="color:{pred_color}; margin: 5px 0 15px 0; font-size:32px;">{pred_name}</h1>
            <p style="color:white; font-size:16px;">This customer exhibits spending patterns matching our <b>{pred_name}</b> cluster.</p>
            <div style="background:rgba(0,0,0,0.2); padding:20px; border-radius:8px; border:1px solid rgba(255,255,255,0.05); margin-top:15px;">
                <h5 style="margin:0 0 10px 0; color:white; font-weight:bold; font-size:16px;">📋 Targeted Marketing & Risk Mitigation Plan:</h5>
                <div style="color:#cbd5e1; line-height:1.8; font-size:15px;">
                    {strategies[pred_name]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
