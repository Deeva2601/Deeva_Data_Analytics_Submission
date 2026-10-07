"""
================================================================================
Retail Sales Exploratory Data Analysis (EDA) Script
Oasis Infobyte - Data Analytics Internship | Task 1
================================================================================
Author: Data Analytics Intern
Description: End-to-end Python pipeline for data ingestion, cleaning, 
             descriptive statistics, time series analysis, demographic profiling,
             product analysis, correlation heatmap, and insight generation.
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
from matplotlib.ticker import FuncFormatter

# ------------------------------------------------------------------------------
# 1. Configuration & Plot Settings
# ------------------------------------------------------------------------------
OUTPUT_DIR = 'assets/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'

def run_pipeline(data_path='retail_sales_dataset.csv'):
    print("=" * 80)
    print("STARTING RETAIL SALES EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 2. Data Ingestion & Health Checks
    # --------------------------------------------------------------------------
    print(f"\n[1/7] Ingesting dataset from: {data_path} ...")
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"  [+] Records: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print(f"  [+] Null Values: {df.isnull().sum().sum()}")
    print(f"  [+] Duplicates: {df.duplicated().sum()}")

    # --------------------------------------------------------------------------
    # 3. Data Preprocessing & Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[2/7] Preprocessing & Feature Engineering ...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%b')
    df['Quarter'] = df['Date'].dt.quarter
    df['Year_Quarter'] = df['Date'].dt.to_period('Q').astype(str)
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Day_Index'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['Day_Index'].isin([5, 6]).map({True: 'Weekend', False: 'Weekday'})

    age_bins = [17, 25, 35, 50, 65, 100]
    age_labels = ['18-25 (Young Adult)', '26-35 (Early Career)', '36-50 (Middle-Aged)', '51-65 (Mature Adult)', '65+ (Senior)']
    df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=True)
    df['Gross_Amount'] = df['Quantity'] * df['Price_per_Unit']
    df['Discount_Amount'] = df['Gross_Amount'] - df['Total_Amount']
    print("  [+] Temporal & demographic attributes successfully engineered.")

    # --------------------------------------------------------------------------
    # 4. Descriptive Statistics
    # --------------------------------------------------------------------------
    print("\n[3/7] Computing Descriptive Statistics ...")
    num_cols = ['Age', 'Quantity', 'Price_per_Unit', 'Discount_Percent', 'Gross_Amount', 'Total_Amount']
    for col in num_cols:
        s = df[col]
        print(f"  • {col:18s} | Mean: {s.mean():8.2f} | Median: {s.median():8.2f} | Mode: {s.mode()[0]:8.2f} | Std: {s.std():8.2f}")

    # --------------------------------------------------------------------------
    # 5. Visualizations & Time Series
    # --------------------------------------------------------------------------
    print("\n[4/7] Generating Time Series Analysis Charts ...")
    monthly_sales = df.groupby(pd.Grouper(key='Date', freq='ME' if hasattr(pd.Grouper, 'freq') else 'M')).agg(
        Total_Revenue=('Total_Amount', 'sum')
    ).reset_index()
    monthly_sales['Month_Year'] = monthly_sales['Date'].dt.strftime('%b %Y')
    monthly_sales['Rolling_3M_Avg'] = monthly_sales['Total_Revenue'].rolling(window=3, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(monthly_sales['Month_Year'], monthly_sales['Total_Revenue'], marker='o', linewidth=2.8, color='#1f77b4', label='Monthly Revenue ($)')
    ax1.plot(monthly_sales['Month_Year'], monthly_sales['Rolling_3M_Avg'], linestyle='--', linewidth=2.2, color='#ff7f0e', label='3-Month Rolling Average')
    ax1.fill_between(range(len(monthly_sales)), monthly_sales['Total_Revenue'], alpha=0.15, color='#1f77b4')
    ax1.set_title('Monthly Total Sales Revenue & Trend Analysis (2023 - 2024)', fontsize=14, pad=15)
    ax1.set_xlabel('Month & Year', fontsize=11)
    ax1.set_ylabel('Total Revenue (USD $)', fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax1.set_xticks(range(len(monthly_sales)))
    ax1.set_xticklabels(monthly_sales['Month_Year'], rotation=45, ha='right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_monthly_sales_trend.png', dpi=300)
    plt.close()
    print("  [+] Saved 01_monthly_sales_trend.png")

    # Quarterly Sales
    quarterly_sales = df.groupby('Year_Quarter').agg(Total_Revenue=('Total_Amount', 'sum')).reset_index()
    quarterly_sales['QoQ_Growth_%'] = quarterly_sales['Total_Revenue'].pct_change() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    bars = ax1.bar(quarterly_sales['Year_Quarter'], quarterly_sales['Total_Revenue'], color=sns.color_palette("mako", len(quarterly_sales)), edgecolor='black')
    ax1.set_title('Quarterly Revenue Breakdown', fontsize=13)
    ax1.set_ylabel('Total Revenue ($)', fontsize=11)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.015), f'${yval:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.plot(quarterly_sales['Year_Quarter'], quarterly_sales['QoQ_Growth_%'], marker='s', color='#2b5c8f', linewidth=2.5)
    ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax2.set_title('Quarter-over-Quarter Growth Rate (%)', fontsize=13)
    ax2.set_ylabel('Growth Rate (%)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_quarterly_sales_trends.png', dpi=300)
    plt.close()
    print("  [+] Saved 02_quarterly_sales_trends.png")

    # --------------------------------------------------------------------------
    # 6. Demographics & Products
    # --------------------------------------------------------------------------
    print("\n[5/7] Generating Customer & Product Analytics ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    sns.histplot(df['Age'], kde=True, bins=25, color='#3470a3', edgecolor='white', ax=axes[0])
    axes[0].set_title('Customer Age Distribution', fontsize=13)
    
    gender_counts = df['Gender'].value_counts()
    axes[1].pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, 
                colors=['#ff9999', '#66b3ff'], wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
    axes[1].set_title('Customer Gender Breakdown', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_customer_demographics_overview.png', dpi=300)
    plt.close()

    # Product Performance
    top10_revenue = df.groupby(['Product_Name', 'Product_Category']).agg(
        Total_Revenue=('Total_Amount', 'sum')
    ).reset_index().sort_values('Total_Revenue', ascending=False).head(10)

    category_summary = df.groupby('Product_Category').agg(Total_Revenue=('Total_Amount', 'sum')).reset_index().sort_values('Total_Revenue', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=top10_revenue, y='Product_Name', x='Total_Revenue', hue='Product_Category', dodge=False, palette='tab10', ax=ax1)
    ax1.set_title('Top 10 Best-Selling Products by Revenue', fontsize=13)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2.bar(category_summary['Product_Category'], category_summary['Total_Revenue'], color=sns.color_palette('viridis', len(category_summary)), edgecolor='black')
    ax2.set_title('Total Revenue by Category', fontsize=13)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'${y:,.0f}'))
    ax2.set_xticklabels(category_summary['Product_Category'], rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/05_product_category_performance.png', dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # 7. Correlation Heatmap & Deep Dives
    # --------------------------------------------------------------------------
    print("\n[6/7] Generating Correlation Heatmap & Non-Obvious Insights ...")
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='vlag', vmin=-1, vmax=1, mask=mask, square=True, linewidths=1.2)
    plt.title('Numerical Correlation Heatmap', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_correlation_heatmap.png', dpi=300)
    plt.close()

    print("\n[7/7] Pipeline execution completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline()
