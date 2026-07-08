# 💳 Credit Card Customer Spending Cluster Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://share.streamlit.io/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive, premium analytics dashboard built using **Streamlit** and **scikit-learn** to segment credit card customers based on transactional behavior. This project applies **K-Means Clustering** and **Principal Component Analysis (PCA)** to uncover distinct spending patterns and deliver actionable, data-driven business recommendations.

---

## 🎯 Business Context & Value
In the modern financial sector, one-size-fits-all marketing is inefficient. To maximize customer lifetime value (CLV) and manage credit risk, banks must build precise customer profiles. 

This model segments **8,950 cardholders** using 17 behavioral attributes into **three highly distinct cohorts**:
1. 🟢 **Active Transactors (20.0%):** High spenders with massive credit limits who pay their bills in full regularly. (Retain & reward with premium perks).
2. 🔵 **Inactive Users (67.9%):** The vast majority who carry balances but rarely transact. (Activate using personalized incentives at zero acquisition cost).
3. 🔴 **Cash Advance Revolvers (12.1%):** Heavily dependent on cash advances, carrying high revolving balances with very low payment-in-full rates. (Mitigate default risk through early debt restructuring and spending alerts).

---

## 🚀 Interactive App Features
* **🏠 Executive Overview:** Complete business case overview, metric highlights, and full descriptive statistics.
* **📊 Exploratory Data Analysis (EDA):** Sleek, right-skewed distributions of purchases, cash advance usage, and feature correlation matrix.
* **🔍 Optimal K-Selection:** High-fidelity visual evidence using the **Elbow Method (WCSS)** and **Silhouette Analysis** (K=3 chosen as mathematically optimal).
* **🎯 Segment Profiles:** Highly interactive 2D cluster visualization using **PCA** and deep dive feature-by-feature comparisons.
* **💡 Business Insights & Action Items:** Custom pie distributions, payment rates, and structured action items tailored for product teams.

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spending-cluster.git
cd spending-cluster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

---

## Open In The Browser 
https://spending-cluster.streamlit.app/

## 🛠️ Tech Stack & Methods
* **Dashboard:** Streamlit (Custom Dark Theme UI)
* **Visualizations:** Matplotlib, Seaborn (Matched to dark theme context)
* **Data Preparation:** Scikit-Learn `StandardScaler`
* **Segmentation Algorithm:** `KMeans` (Clustering)
* **Dimensionality Reduction:** `PCA` (2 Component Projection)

---

## 📈 Key Mathematical Insights
* **Silhouette Score:** `0.240` (K=3) — signifying robust cluster boundaries.
* **PCA Variance Explained:** PC1 (27.3%) + PC2 (20.3%) = **47.6%** total explained variance in 2D space.
