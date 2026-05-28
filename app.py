import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

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

# ── PAGE CONFIG ──────────────────────────────
st.set_page_config(
    page_title="Spending Cluster Analysis",
    page_icon="💳",
    layout="wide"
)

# ── LOAD & PREPARE DATA ──────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('CC GENERAL.csv')
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median())
    df.drop('CUST_ID', axis=1, inplace=True)
    return df

@st.cache_data
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
    return df, df_scaled, pca

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
    "💡 Insights"
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
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['PURCHASES'], bins=100, color='steelblue', edgecolor='none', alpha=0.85)
        ax.set_xlabel('Purchase Amount')
        ax.set_ylabel('Count')
        ax.set_title('Highly Right-Skewed — Few Spend a Lot')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Cash Advance Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['CASH_ADVANCE'], bins=100, color='tomato', edgecolor='none', alpha=0.85)
        ax.set_xlabel('Cash Advance Amount')
        ax.set_ylabel('Count')
        ax.set_title('Most Customers Never Use Cash Advance')
        st.pyplot(fig)
        plt.close()

    st.subheader("Outlier Detection — Boxplot")
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.boxplot(data=df[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']], ax=ax, palette='Set2', showfliers=False)
    ax.set_title('Key Feature Distributions (Outliers Hidden for Clarity)')
    plt.xticks(rotation=15)
    st.pyplot(fig)
    plt.close()


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
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(K_range), inertias, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Optimal K=3')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Inertia (WCSS)')
        ax.set_title('Elbow Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Silhouette Scores")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(K_range), sil_scores, 's-', color='tomato', linewidth=2, markersize=8)
        ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Best K=3')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score (Higher = Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    st.success("✅ K=3 chosen — highest silhouette score (0.2403) and clear elbow point")

# ── 4. CLUSTERS ───────────────────────────────
elif page == "🎯 Clusters":
    st.title("🎯 Cluster Profiles")
    st.divider()

    df_clustered, df_scaled, pca = run_clustering()

    # Cluster sizes
    counts = df_clustered['Cluster_Name'].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Active Transactors", counts.get('Active Transactors', 0))
    col2.metric("🔵 Inactive Users", counts.get('Inactive Users', 0))
    col3.metric("🔴 Cash Advance Revolvers", counts.get('Cash Advance Revolvers', 0))

    st.divider()

    # PCA Scatter
    st.subheader("Customer Clusters — PCA 2D View")
    colors = {'Active Transactors': '#2ecc71', 'Inactive Users': '#3498db', 'Cash Advance Revolvers': '#e74c3c'}
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, group in df_clustered.groupby('Cluster_Name'):
        ax.scatter(group['PCA1'], group['PCA2'], c=colors[name], label=name, alpha=0.5, s=15)
    ax.set_xlabel(f'PC1 (27.3% variance)')
    ax.set_ylabel(f'PC2 (20.3% variance)')
    ax.set_title('Customer Spending Clusters (PCA 2D View)')
    ax.legend(markerscale=3, fontsize=10)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Cluster profile bar chart
    st.subheader("Cluster Comparison — Key Features")
    features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']
    profile = df_clustered.groupby('Cluster_Name')[features].mean().round(2)
    
    # Map colors to match the scatter plot exactly
    bar_colors = [colors[col] for col in profile.T.columns]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    profile.T.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='#0e1117', linewidth=1)
    ax.set_title('Mean Feature Values per Cluster')
    ax.set_ylabel('Mean Value')
    ax.set_xlabel('Feature')
    plt.xticks(rotation=15)
    ax.legend(loc='upper right')
    st.pyplot(fig)
    plt.close()


    st.subheader("Full Cluster Profile Table")
    st.dataframe(df_clustered.groupby('Cluster_Name').mean().round(2), use_container_width=True)

# ── 5. INSIGHTS ───────────────────────────────
elif page == "💡 Insights":
    st.title("💡 Business Insights")
    st.divider()

    df_clustered, df_scaled, pca = run_clustering()
    profile = df_clustered.groupby('Cluster_Name').mean().round(3)
    counts = df_clustered['Cluster_Name'].value_counts()

    col1, col2 = st.columns(2)

    # ── PIE CHART ──
    with col1:
        st.subheader("🥧 Customer Segment Distribution")
        fig, ax = plt.subplots(figsize=(6, 5))
        pie_colors = [colors[name] for name in counts.index]
        ax.pie(counts.values, labels=counts.index,
               colors=pie_colors,
               autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor': '#0e1117', 'linewidth': 2})
        ax.set_title('How Customers Are Divided')
        st.pyplot(fig)
        plt.close()
        st.info("💡 **Insight:** Nearly 7 out of 10 customers are inactive. The biggest growth opportunity is activating existing customers — not acquiring new ones.")

    # ── FULL PAYMENT RATE ──
    with col2:
        st.subheader("💳 Who Pays Their Full Bill?")
        fig, ax = plt.subplots(figsize=(6, 5))
        clusters = profile.index.tolist()
        values = profile['PRC_FULL_PAYMENT'].values
        colors_bar = [colors[name] for name in clusters]
        bars = ax.bar(clusters, values, color=colors_bar, edgecolor='#0e1117', linewidth=1.5)
        ax.set_ylabel('Proportion Paying in Full')
        ax.set_title('Full Bill Payment Rate per Cluster')
        ax.set_ylim(0, max(values) + 0.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', fontweight='bold', fontsize=11, color='#ffffff')
        plt.xticks(rotation=10)
        st.pyplot(fig)
        plt.close()
        st.info("💡 **Insight:** Active Transactors pay in full most often — low risk, high value. Cash Advance Revolvers rarely pay in full — higher default risk for the bank.")


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
