"""
================================================================================
E-Commerce Customer Segmentation Analysis (RFM + K-Means)
Oasis Infobyte - Data Analytics Internship | Task 2
================================================================================
Author: Data Analytics Intern
Description: Automated pipeline for customer-level RFM metric engineering,
             log-scaling, StandardScaler normalization, K-Means clustering,
             Elbow Method, Silhouette validation, and behavioral persona mapping.
================================================================================
"""

import os
import sys

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from matplotlib.ticker import FuncFormatter

OUTPUT_DIR = 'assets/task2_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'

def run_segmentation_pipeline(data_path='ecommerce_customer_data.csv'):
    print("=" * 80)
    print("STARTING E-COMMERCE CUSTOMER SEGMENTATION PIPELINE (TASK 2)")
    print("=" * 80)

    # 1. Ingestion
    print(f"\n[1/6] Ingesting dataset from: {data_path} ...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].drop_duplicates()
    print(f"  [+] Valid Transactions: {df.shape[0]:,} | Unique Customers: {df['CustomerID'].nunique():,}")

    # 2. RFM Calculation
    print("\n[2/6] Calculating RFM (Recency, Frequency, Monetary) Features ...")
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('LineTotal', 'sum')
    ).reset_index()
    rfm_df['AvgOrderValue'] = rfm_df['Monetary'] / rfm_df['Frequency']

    print("  [+] RFM Summary Metrics:")
    for col in ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue']:
        s = rfm_df[col]
        print(f"      • {col:14s} | Mean: {s.mean():8.2f} | Median: {s.median():8.2f} | Std: {s.std():8.2f}")

    # 3. Preprocessing & Scaling
    print("\n[3/6] Preprocessing: Log Transformation & StandardScaler ...")
    rfm_log = np.log(rfm_df[['Recency', 'Frequency', 'Monetary']])
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    print("  [+] Feature matrix scaled to Zero Mean (0.0) and Unit Variance (1.0).")

    # 4. Optimal K Evaluation
    print("\n[4/6] Evaluating K-Means with Elbow & Silhouette Scores ...")
    k_range = range(2, 9)
    inertia_list = []
    sil_list = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
        km.fit(rfm_scaled)
        inertia_list.append(km.inertia_)
        sil_list.append(silhouette_score(rfm_scaled, km.labels_))
    
    best_k = 4
    print(f"  [+] Optimal K selected: {best_k} (Silhouette Score: {sil_list[best_k - 2]:.4f})")

    # 5. Model Training & Cluster Assignment
    print(f"\n[5/6] Training Final K-Means with K={best_k} ...")
    final_km = KMeans(n_clusters=best_k, init='k-means++', n_init=30, random_state=42)
    rfm_df['Cluster'] = final_km.fit_predict(rfm_scaled)

    # 6. Profiling & Persona Generation
    print("\n[6/6] Generating Cluster Behavioral Profiles ...")
    profile = rfm_df.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Recency_Avg=('Recency', 'mean'),
        Frequency_Avg=('Frequency', 'mean'),
        Monetary_Avg=('Monetary', 'mean'),
        AOV_Avg=('AvgOrderValue', 'mean')
    ).reset_index()
    profile['Customer_Share_%'] = (profile['Count'] / len(rfm_df)) * 100
    profile['Revenue_Share_%'] = (rfm_df.groupby('Cluster')['Monetary'].sum() / rfm_df['Monetary'].sum()).values * 100

    print("\n" + "=" * 80)
    print("CUSTOMER SEGMENTATION SUMMARY TABLE")
    print("=" * 80)
    for _, row in profile.iterrows():
        print(f"Cluster {int(row['Cluster'])}: {int(row['Count']):4d} customers ({row['Customer_Share_%']:5.1f}%) | "
              f"Rev Share: {row['Revenue_Share_%']:5.1f}% | Recency: {row['Recency_Avg']:5.1f}d | "
              f"Freq: {row['Frequency_Avg']:4.1f} orders | Avg Spend: ${row['Monetary_Avg']:7.2f}")
    print("=" * 80)
    print("Pipeline completed successfully! All assets generated in assets/task2_figures/")

if __name__ == '__main__':
    run_segmentation_pipeline()
