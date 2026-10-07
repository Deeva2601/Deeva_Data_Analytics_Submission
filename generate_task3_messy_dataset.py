"""
Script to generate a realistic, deliberately messy real-world E-Commerce Customer & Orders dataset for Task 3: Data Cleaning.
Includes dirty formatting, mixed date formats, string currency signs, whitespace, typos, duplicates, missing values, and anomalies.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_messy_dataset(n_rows=2000, random_state=42):
    np.random.seed(random_state)
    
    genders = ['Male', 'male', 'M', 'MALE', 'Female', 'female', 'F', 'FEMALE', 'Other', 'unknown', np.nan, '?', 'n/a']
    raw_weights = [0.25, 0.15, 0.08, 0.04, 0.25, 0.12, 0.05, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01]
    gender_weights = [w / sum(raw_weights) for w in raw_weights]
    
    categories = [' Electronics ', 'clothing', 'CLOTHING', 'Beauty & Personal Care', 'Home & Kitchen', 'sports & outdoors', 'Books', 'electronics']
    
    payment_methods = ['Credit Card', 'credit card', 'CC', 'Debit Card', 'debit_card', 'PayPal', 'paypal', 'PAYPAL', 'Cash', 'cash', 'UPI / Wallet', np.nan]
    
    states = [' CA ', 'California', 'california', 'NY', 'New York', 'TX', 'Texas', 'texas', 'FL', 'Florida', 'IL', 'Illinois', 'WA', 'Washington', np.nan]
    
    records = []
    
    base_date = datetime(2023, 1, 1)
    
    for i in range(1, n_rows + 1):
        cust_id = f"CUST_{np.random.randint(1001, 1600)}"
        order_id = f"ORD-{10000 + i}"
        
        # Age: mostly 18-75, with some missing and some extreme outliers/negatives
        age_rnd = np.random.rand()
        if age_rnd < 0.05:
            age = np.nan
        elif age_rnd < 0.06:
            age = np.random.choice([-12, -5, 185, 250, 999])
        else:
            age = int(np.random.normal(38, 14))
            age = max(16, min(80, age))
            
        gender = np.random.choice(genders, p=gender_weights)
        
        # Annual Income with string formatting and missing
        income_rnd = np.random.rand()
        if income_rnd < 0.08:
            income_str = np.random.choice([np.nan, 'unknown', 'N/A', 'null', '$ - '])
        elif income_rnd < 0.09:
            income_str = "$9,999,999.00" # Extreme outlier typo
        else:
            income_val = round(np.random.normal(65000, 24000), 2)
            income_val = max(15000.0, income_val)
            # Add string dirty formats ($ and commas)
            fmt_style = np.random.choice(['dollar_comma', 'plain', 'dollar_space', 'comma_only'])
            if fmt_style == 'dollar_comma':
                income_str = f"${income_val:,.2f}"
            elif fmt_style == 'dollar_space':
                income_str = f"$ {income_val:.2f}"
            elif fmt_style == 'comma_only':
                income_str = f"{income_val:,.2f}"
            else:
                income_str = str(income_val)
                
        # Order Date with mixed date formats
        date_offset = np.random.randint(0, 700)
        dt = base_date + timedelta(days=date_offset)
        date_format = np.random.choice(['iso', 'us_slash', 'text_month', 'dot_format', 'missing'])
        if date_format == 'iso':
            order_date_str = dt.strftime('%Y-%m-%d')
        elif date_format == 'us_slash':
            order_date_str = dt.strftime('%m/%d/%Y')
        elif date_format == 'text_month':
            order_date_str = dt.strftime('%b %d, %Y')
        elif date_format == 'dot_format':
            order_date_str = dt.strftime('%d.%m.%Y')
        else:
            order_date_str = np.random.choice([np.nan, '2023-99-99', 'Invalid Date'])
            
        # Product Category
        category = np.random.choice(categories)
        
        # Order Quantity
        qty_rnd = np.random.rand()
        if qty_rnd < 0.03:
            qty = np.random.choice([-1, -3, 0, 999])
        else:
            qty = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))
            
        # Unit Price with symbols and dirty strings
        unit_price_val = round(np.random.uniform(15.0, 450.0), 2)
        price_fmt = np.random.choice(['clean', 'dollar', 'dirty_str', 'missing'])
        if price_fmt == 'clean':
            price_str = str(unit_price_val)
        elif price_fmt == 'dollar':
            price_str = f"${unit_price_val:.2f}"
        elif price_fmt == 'dirty_str':
            price_str = f" ${unit_price_val:.2f} USD "
        else:
            price_str = np.random.choice([np.nan, 'FREE', '0.00', 'TBD'])
            
        # Discount Percentage
        disc_rnd = np.random.rand()
        if disc_rnd < 0.10:
            disc_str = np.random.choice([np.nan, '0%', '10%', '15 %', '25%'])
        else:
            disc_str = f"{np.random.choice([0, 5, 10, 15, 20])}%"
            
        payment = np.random.choice(payment_methods)
        state = np.random.choice(states)
        
        records.append({
            'Order_ID': order_id,
            'Customer_ID': cust_id,
            'Customer_Age': age,
            'Gender': gender,
            'Annual_Income': income_str,
            'State': state,
            'Order_Date': order_date_str,
            'Product_Category': category,
            'Quantity': qty,
            'Unit_Price': price_str,
            'Discount_Applied': disc_str,
            'Payment_Method': payment
        })
        
    df = pd.DataFrame(records)
    
    # Inject ~85 duplicate rows intentionally
    duplicate_indices = np.random.choice(len(df), size=85, replace=False)
    duplicates_df = df.iloc[duplicate_indices].copy()
    df_messy = pd.concat([df, duplicates_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_messy

if __name__ == '__main__':
    df = generate_messy_dataset(n_rows=2000)
    output_file = 'raw_messy_customer_orders.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated deliberately messy dataset with {len(df):,} rows and {df.shape[1]} columns at {output_file}")
    print("\nSample messy rows:")
    print(df.head(6))
