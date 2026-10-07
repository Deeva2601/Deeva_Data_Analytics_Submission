"""
================================================================================
Professional Data Cleaning & Preprocessing Pipeline
Oasis Infobyte - Data Analytics Internship | Task 3
================================================================================
Author: Data Analytics Intern
Description: Automated pipeline for raw data quality auditing, duplicate removal,
             string standardization, missing value imputation, IQR outlier treatment,
             and explicit schema type enforcement.
================================================================================
"""

import os
import sys
import re

# Ensure UTF-8 output handling on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = 'assets/task3_figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_cleaning_pipeline(raw_path='raw_messy_customer_orders.csv', clean_output_path='cleaned_customer_orders_data.csv'):
    print("=" * 80)
    print("STARTING ENTERPRISE DATA CLEANING PIPELINE (TASK 3)")
    print("=" * 80)

    # 1. Ingestion
    print(f"\n[1/6] Ingesting raw messy dataset: {raw_path} ...")
    if not os.path.exists(raw_path):
        print(f"Error: Raw dataset {raw_path} not found.")
        sys.exit(1)

    df_raw = pd.read_csv(raw_path)
    print(f"  [+] Raw Dimensions: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"  [+] Total Nulls Detected: {df_raw.isnull().sum().sum():,}")
    print(f"  [+] Exact Duplicate Rows: {df_raw.duplicated().sum():,}")

    # 2. Duplicate Removal
    print("\n[2/6] Purging duplicate records ...")
    df = df_raw.drop_duplicates().reset_index(drop=True)
    print(f"  [+] Rows remaining after de-duplication: {len(df):,}")

    # 3. String Standardization & Parsing
    print("\n[3/6] Standardizing text formatting, categories & numeric strings ...")
    
    # Gender
    def clean_gender(val):
        if pd.isnull(val):
            return 'Other / Unspecified'
        s = str(val).strip().lower()
        if s in ['m', 'male']:
            return 'Male'
        elif s in ['f', 'female']:
            return 'Female'
        return 'Other / Unspecified'
    df['Gender'] = df['Gender'].apply(clean_gender)

    # Category
    df['Product_Category'] = df['Product_Category'].astype(str).str.strip().str.title()

    # State
    state_mapping = {
        'ca': 'California', 'california': 'California',
        'ny': 'New York', 'new york': 'New York',
        'tx': 'Texas', 'texas': 'Texas',
        'fl': 'Florida', 'florida': 'Florida',
        'il': 'Illinois', 'illinois': 'Illinois',
        'wa': 'Washington', 'washington': 'Washington'
    }
    df['State'] = df['State'].apply(lambda x: state_mapping.get(str(x).strip().lower(), 'Unknown') if pd.notnull(x) else 'Unknown')

    # Payment Method
    payment_mapping = {
        'credit card': 'Credit Card', 'cc': 'Credit Card',
        'debit card': 'Debit Card', 'debit_card': 'Debit Card',
        'paypal': 'PayPal', 'cash': 'Cash', 'upi / wallet': 'UPI / Digital Wallet'
    }
    df['Payment_Method'] = df['Payment_Method'].apply(lambda x: payment_mapping.get(str(x).strip().lower(), 'Unknown') if pd.notnull(x) else 'Unknown')

    # Clean numeric strings (Income & Price)
    def clean_numeric_str(val):
        if pd.isnull(val):
            return np.nan
        s = str(val).strip().lower()
        if any(p in s for p in ['unknown', 'n/a', 'null', 'nan', '-', 'tbd', 'free', '?']):
            return np.nan
        s_cleaned = re.sub(r'[^\d.]', '', s)
        try:
            return float(s_cleaned) if s_cleaned else np.nan
        except ValueError:
            return np.nan

    df['Annual_Income'] = df['Annual_Income'].apply(clean_numeric_str)
    df['Unit_Price'] = df['Unit_Price'].apply(clean_numeric_str)

    # Discount
    def clean_discount(val):
        if pd.isnull(val):
            return 0.0
        s = str(val).strip()
        s_cleaned = re.sub(r'[^\d.]', '', s)
        try:
            f = float(s_cleaned) if s_cleaned else 0.0
            return f / 100.0 if f > 1.0 else f
        except ValueError:
            return 0.0

    df['Discount_Applied'] = df['Discount_Applied'].apply(clean_discount)
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='mixed', errors='coerce')
    print("  [+] String schemas and date coercion completed.")

    # 4. Missing Value Imputation
    print("\n[4/6] Imputing missing values with statistical justifications ...")
    age_median = df['Customer_Age'].median()
    df['Customer_Age'] = df['Customer_Age'].fillna(age_median)

    df['Annual_Income'] = df.groupby('State')['Annual_Income'].transform(lambda x: x.fillna(x.median()))
    df['Annual_Income'] = df['Annual_Income'].fillna(df['Annual_Income'].median())

    df['Unit_Price'] = df.groupby('Product_Category')['Unit_Price'].transform(lambda x: x.fillna(x.median()))
    df['Unit_Price'] = df['Unit_Price'].fillna(df['Unit_Price'].median())

    df['Order_Date'] = df['Order_Date'].ffill().bfill()
    print(f"  [+] Missing values after imputation: {df.isnull().sum().sum()}")

    # 5. Outlier Detection & Capping (IQR & Domain Bounds)
    print("\n[5/6] Remediating outliers & domain anomalies ...")
    df['Customer_Age'] = df['Customer_Age'].apply(lambda x: age_median if (x < 16 or x > 90) else x)
    qty_mode = df.loc[(df['Quantity'] > 0) & (df['Quantity'] <= 10), 'Quantity'].mode()[0]
    df['Quantity'] = df['Quantity'].apply(lambda x: qty_mode if (x <= 0 or x > 20) else x)

    # Income IQR Capping
    q1 = df['Annual_Income'].quantile(0.25)
    q3 = df['Annual_Income'].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + (1.5 * iqr)
    cap_val = df['Annual_Income'].quantile(0.99)
    df['Annual_Income'] = np.where(df['Annual_Income'] > upper_fence, cap_val, df['Annual_Income'])

    # Calculated Financial Columns
    df['Gross_Amount'] = df['Quantity'] * df['Unit_Price']
    df['Total_Amount'] = df['Gross_Amount'] * (1.0 - df['Discount_Applied'])

    # 6. Type Enforcement & Export
    print("\n[6/6] Enforcing explicit data types and exporting ...")
    df['Order_ID'] = df['Order_ID'].astype('string')
    df['Customer_ID'] = df['Customer_ID'].astype('string')
    df['Customer_Age'] = df['Customer_Age'].astype('int32')
    df['Gender'] = df['Gender'].astype('category')
    df['Annual_Income'] = df['Annual_Income'].astype('float64')
    df['State'] = df['State'].astype('category')
    df['Product_Category'] = df['Product_Category'].astype('category')
    df['Quantity'] = df['Quantity'].astype('int32')
    df['Unit_Price'] = df['Unit_Price'].astype('float64')
    df['Discount_Applied'] = df['Discount_Applied'].astype('float64')
    df['Gross_Amount'] = df['Gross_Amount'].astype('float64')
    df['Total_Amount'] = df['Total_Amount'].astype('float64')
    df['Payment_Method'] = df['Payment_Method'].astype('category')

    df.to_csv(clean_output_path, index=False)
    print(f"  [+] Cleaned dataset successfully exported to: {clean_output_path}")
    print(f"  [+] Final Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    run_cleaning_pipeline()
