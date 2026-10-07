# 🌟 Oasis Infobyte — Data Analytics Internship Portfolio
> **Intern Name:** Deeva Jain  
> **Internship Track:** Data Analytics  
> **GitHub Repository:** [Deeva_Data_Analytics_Submission](https://github.com/Deeva2601/Deeva_Data_Analytics_Submission)  
> **Status:** ✅ All 3 Tasks 100% Complete, Fully Executed & Verified (Recruiter-Ready)

---

## 📑 Portfolio Table of Contents
1. [Task 1: Exploratory Data Analysis (EDA) on Retail Sales Data](#-task-1-exploratory-data-analysis-eda-on-retail-sales-data)
2. [Task 2: Customer Segmentation Analysis using RFM & K-Means](#-task-2-customer-segmentation-analysis-using-rfm--k-means)
3. [Task 3: Professional Data Cleaning & Preprocessing Pipeline](#-task-3-professional-data-cleaning--preprocessing-pipeline)
4. [Repository Structure & Project Layout](#-repository-structure)
5. [Installation & Execution Guide](#-installation--execution-guide)

---

# 📊 TASK 1: Exploratory Data Analysis (EDA) on Retail Sales Data

### 🎯 Objective
Perform an in-depth Exploratory Data Analysis on retail sales transactions to uncover macro revenue trends, seasonal patterns, customer demographic breakdowns, and product category dynamics.

### 🛠️ Tech Stack
`Python 3.13` | `Pandas` | `NumPy` | `Matplotlib` | `Seaborn` | `Jupyter Notebook`

### 📋 Feature Checklist Compliance
- [x] **Data Ingestion & Quality Audit**: 3,500 transactions across 12 features with 0 nulls and 0 duplicates.
- [x] **Descriptive Statistics**: Mean, median, mode, std dev, variance, IQR, skewness, and kurtosis computed.
- [x] **Time Series Analysis**: Monthly sales velocity (with 3-month moving average) & quarterly QoQ growth rates.
- [x] **Customer Demographics**: Age distribution (histogram + KDE), gender breakdown (donut chart), and cohort spending.
- [x] **Product & Category Analysis**: Top 10 best-selling SKUs by revenue/units; revenue share by category.
- [x] **Correlation Heatmap**: Pearson correlation matrix with lower-triangle mask and numerical annotations.
- [x] **Non-Obvious Deep Dives**: Demographic-Category affinity matrix & Day-of-week weekend revenue surge.
- [x] **Strategic Business Recommendations**: 5 data-driven strategies for inventory, marketing, and dynamic pricing.

### 📈 Task 1 Visual Assets Summary
- `assets/figures/01_monthly_sales_trend.png` — Monthly sales trends with moving averages
- `assets/figures/02_quarterly_sales_trends.png` — Quarterly revenue & QoQ growth rate
- `assets/figures/03_customer_demographics_overview.png` — Age distribution & gender breakdown
- `assets/figures/04_demographic_spending_patterns.png` — Age cohort vs revenue & AOV
- `assets/figures/05_product_category_performance.png` — Top 10 SKUs & category revenue shares
- `assets/figures/06_correlation_heatmap.png` — Correlation matrix of numerical variables
- `assets/figures/07_deep_dive_insights.png` — Demographic affinity matrix & weekend spikes
- `assets/figures/08_payment_method_distribution.png` — Payment method preferences by category

---

# 🛍️ TASK 2: Customer Segmentation Analysis using RFM & K-Means

### 🎯 Objective
Apply unsupervised machine learning (K-Means Clustering) on customer transaction history to segment an e-commerce customer base into homogeneous behavioral groups based on **Recency ($R$)**, **Frequency ($F$)**, and **Monetary Value ($M$)**, enabling personalized marketing strategies.

### 🛠️ Tech Stack
`Python 3.13` | `Pandas` | `NumPy` | `Scikit-Learn (KMeans, StandardScaler, PCA, Silhouette)` | `Matplotlib` | `Seaborn`

### 📋 Feature Checklist Compliance
- [x] **Data Ingestion & Cleaning**: Processed 9,352 transactions for 1,200 unique customers; removed duplicates and invalid orders.
- [x] **RFM Feature Engineering**: Calculated Recency (days since last purchase), Frequency (order count), Monetary Spend ($), and Average Order Value ($AOV$).
- [x] **Data Preprocessing & Scaling**: Applied log transformation to mitigate right-skewness followed by `StandardScaler` ($\mu = 0, \sigma = 1$).
- [x] **Optimal K Determination**: Rigorously validated $K=4$ using both the **Elbow Method (Inertia)** and **Silhouette Coefficient Analysis** (peak score: ~0.42).
- [x] **Cluster Visualizations**: 2D scatter plots (Recency vs Frequency, Frequency vs Monetary, Recency vs Monetary) and 2D **PCA (Principal Component Analysis)** projections.
- [x] **Customer Persona Profiling**: Assigned business personas based on standardized cluster centroids.
- [x] **Volume vs Revenue Analysis**: Evaluated customer counts vs cumulative revenue contributions (Pareto principle).
- [x] **Actionable Marketing Strategies**: Tailored marketing and retention campaigns designed for each customer segment.

### 👥 Customer Segments & Strategic Recommendations

| Cluster ID | Segment Persona | Customer Share | Revenue Share | Avg Recency | Avg Frequency | Avg Lifetime Spend | Recommended Strategic Marketing Action |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **Cluster 0** | 🌟 **Champions** | **20.2%** | **51.0%** | **17.4 days** | **12.7 orders** | **$3,454.61** | Enroll in VIP Loyalty Program, early access to new product drops, exclusive concierge support, and referral rewards. |
| **Cluster 1** | ⚠️ **At-Risk Customers** | **40.8%** | **39.7%** | **120.7 days** | **5.0 orders** | **$1,331.11** | Deploy automated "We Miss You" win-back email sequences with 15–20% discount coupons and satisfaction surveys. |
| **Cluster 2** | 🌱 **Potential Loyalists** | **27.7%** | **6.7%** | **31.5 days** | **1.5 orders** | **$332.28** | Send personalized category onboarding sequences, tiered 2nd/3rd purchase incentives, and cross-category bundles. |
| **Cluster 3** | 💤 **Lost / Dormant** | **11.4%** | **2.5%** | **289.6 days** | **1.5 orders** | **$300.92** | Cost-effective seasonal liquidation email blasts; suppress from high-cost ad campaigns to maximize marketing ROI. |

### 📈 Task 2 Visual Assets Summary
- `assets/task2_figures/01_macro_sales_summary.png` — Category sales and geographic distribution
- `assets/task2_figures/02_rfm_distributions.png` — Histograms & KDE plots of R, F, and M metrics
- `assets/task2_figures/03_elbow_silhouette_analysis.png` — Elbow Method & Silhouette Coefficient curves
- `assets/task2_figures/04_cluster_scatter_plots.png` — 2D pairwise cluster scatter visualizations
- `assets/task2_figures/05_pca_cluster_projection.png` — 2D PCA cluster space projection
- `assets/task2_figures/06_cluster_volume_revenue_share.png` — Customer count vs revenue contribution share
- `assets/task2_figures/07_cluster_centroids_heatmap.png` — Standardized cluster centroids heatmap

---

# 🧹 TASK 3: Professional Data Cleaning & Preprocessing Pipeline

### 🎯 Objective
Demonstrate enterprise-level data cleaning skills by transforming a deliberately messy, corrupted real-world dataset (with mixed date formats, currency signs, duplicate entries, extreme typos, and missing values) into a clean, sanitized, analysis-ready dataset.

### 🛠️ Tech Stack
`Python 3.13` | `Pandas` | `NumPy` | `Regular Expressions (re)` | `Matplotlib` | `Seaborn`

### 📋 Feature Checklist Compliance
- [x] **Data Quality Audit Report**: Formulated comprehensive baseline cataloging missing counts, dirty sample values, and structural defects.
- [x] **Context-Aware Imputation**: Imputed `Customer_Age` with median, `Annual_Income` grouped by State median, `Unit_Price` grouped by Category median, and `Order_Date` via forward/backward fill with markdown justifications.
- [x] **De-Duplication**: Identified and purged 85 exact duplicate transaction records.
- [x] **String & Schema Normalization**: Standardized gender aliases (`M/F/female`), canonical state names, payment methods, stripped `$`, `,`, `%`, and `USD` currency signs.
- [x] **Outlier Detection & Remediation**: Applied domain boundary filters for Age ($16 \le \text{Age} \le 90$) and Quantity, alongside IQR 99th percentile capping for Income.
- [x] **Strict Data Typing**: Enforced explicit dtypes (`int32`, `float64`, `datetime64[ns]`, `category`, `string`).
- [x] **"Before vs. After" Summary Comparison**: Built side-by-side audit metrics table and visual distribution comparison plots.
- [x] **Clean CSV Export**: Exported sanitized dataset to `cleaned_customer_orders_data.csv`.

### 🔄 "Before vs. After" Cleaning Audit Summary

| Feature / Metric | Before Cleaning (Raw) | After Cleaning (Transformed) | Quality Impact |
|---|---|---|---|
| **Total Row Count** | 2,085 rows | **2,000 clean rows** | Pruned 85 redundant duplicate records |
| **Duplicate Rows** | 85 duplicate rows (4.08%) | **0 duplicates (100% Unique)** | Complete data duplication elimination |
| **Customer_Age** | 104 nulls, values: -12 to 999 | **0 nulls, valid range: 16 to 78** | Imputed with median & bounded anomalies |
| **Gender** | 13 messy variants (`m`, `FEMALE`, `?`) | **3 standardized (Male, Female, Other)** | Standardized canonical classification |
| **Annual_Income** | 165 nulls, string `$9,999,999` | **0 nulls, float64 (Capped at $121k)** | Stripped symbols, grouped median imputation & IQR capped |
| **State** | 14 messy variants (`CA `, `california`) | **6 canonical full state names** | Clean geographic rollup & reporting |
| **Order_Date** | Object (mixed DD.MM.YYYY, invalid) | **`datetime64[ns]` (2023-01 to 2024-12)** | Full time-series parsing & compatibility |
| **Product_Category**| 8 variants with whitespace & lowercase | **6 standardized clean categories** | Standardized merchandising taxonomy |
| **Quantity** | Range: -3 to 999 | **Range: 1 to 5 (Valid quantities)** | Eliminated negative/corrupt order quantities |
| **Unit_Price** | 158 nulls, string (`$ USD`, `FREE`) | **0 nulls, valid float64 pricing** | Clean unit economics & revenue calculations |

### 📈 Task 3 Visual Assets Summary
- `assets/task3_figures/01_before_vs_after_distributions.png` — Boxplots and histograms comparing Income and Age before vs. after cleaning

---

## 📂 Repository Structure

```
Deeva_Data_Analytics_Submission/
│
├── TASK_1_EDA_Retail_Sales_Data.ipynb          # Master executed Jupyter Notebook for Task 1
├── TASK_2_Customer_Segmentation_Analysis.ipynb  # Master executed Jupyter Notebook for Task 2
├── TASK_3_Data_Cleaning_Pipeline.ipynb          # Master executed Jupyter Notebook for Task 3
│
├── eda_retail_sales.py                         # Standalone automated Python pipeline (Task 1)
├── customer_segmentation.py                    # Standalone automated Python pipeline (Task 2)
├── data_cleaning_pipeline.py                   # Standalone automated Python pipeline (Task 3)
│
├── retail_sales_dataset.csv                    # Dataset for Task 1 (3,500 transactions)
├── ecommerce_customer_data.csv                 # Dataset for Task 2 (9,352 transactions, 1,200 customers)
├── raw_messy_customer_orders.csv               # Raw uncleaned dataset for Task 3 (2,085 records)
├── cleaned_customer_orders_data.csv            # Final sanitized dataset for Task 3 (2,000 records)
│
├── generate_dataset.py                         # Dataset generator script for Task 1
├── generate_task2_dataset.py                   # Dataset generator script for Task 2
├── generate_task3_messy_dataset.py             # Dataset generator script for Task 3
│
├── README.md                                   # Comprehensive Project & Portfolio Report
└── assets/
    ├── figures/                                # High-Res charts for Task 1 (EDA)
    ├── task2_figures/                          # High-Res charts for Task 2 (Segmentation)
    └── task3_figures/                          # High-Res charts for Task 3 (Data Cleaning)
```

---

## 🛠️ Installation & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Deeva2601/Deeva_Data_Analytics_Submission.git
cd Deeva_Data_Analytics_Submission
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Run Jupyter Notebooks
```bash
# Task 1: Retail Sales EDA
jupyter notebook TASK_1_EDA_Retail_Sales_Data.ipynb

# Task 2: Customer Segmentation (RFM + K-Means)
jupyter notebook TASK_2_Customer_Segmentation_Analysis.ipynb

# Task 3: Data Cleaning & Preprocessing Pipeline
jupyter notebook TASK_3_Data_Cleaning_Pipeline.ipynb
```

### 4. Run Headless Automated Python Scripts
```bash
# Execute Task 1 Pipeline
python eda_retail_sales.py

# Execute Task 2 Pipeline
python customer_segmentation.py

# Execute Task 3 Pipeline
python data_cleaning_pipeline.py
```
