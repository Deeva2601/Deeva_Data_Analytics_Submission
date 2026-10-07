"""
Generate realistic E-Commerce Online Retail Customer Transactions dataset for RFM Customer Segmentation.
Contains realistic customer purchase histories, timestamps, invoice amounts, quantities, and return statuses.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_ecommerce_rfm_data(n_records=5000, n_customers=1200, random_state=42):
    np.random.seed(random_state)
    
    # Customer IDs
    customer_ids = [f"CUST-{2000 + i}" for i in range(n_customers)]
    
    # Define distinct customer behavioral personas to produce meaningful clusters:
    # 1. High-Value Champions (Frequent, high spend, recent purchases) ~ 15%
    # 2. Loyal Customers (Moderate-high frequency, steady spend, fairly recent) ~ 25%
    # 3. Potential Loyalists / New Customers (Recent, 1-2 purchases, moderate spend) ~ 30%
    # 4. At-Risk / Hibernating (Infrequent, long days since last purchase, low/medium spend) ~ 20%
    # 5. Lost / Low-Value Churned (Very long recency, single low-value purchase) ~ 10%
    
    persona_assignments = np.random.choice(
        ['Champion', 'Loyal', 'Potential', 'At_Risk', 'Lost'],
        size=n_customers,
        p=[0.15, 0.25, 0.30, 0.20, 0.10]
    )
    
    cust_profiles = {}
    reference_date = datetime(2024, 12, 31)
    
    for cid, persona in zip(customer_ids, persona_assignments):
        if persona == 'Champion':
            n_orders = np.random.randint(8, 22)
            recency_days = np.random.randint(1, 30)
            avg_basket_val = np.random.uniform(250, 750)
        elif persona == 'Loyal':
            n_orders = np.random.randint(4, 9)
            recency_days = np.random.randint(15, 75)
            avg_basket_val = np.random.uniform(120, 350)
        elif persona == 'Potential':
            n_orders = np.random.randint(1, 3)
            recency_days = np.random.randint(5, 60)
            avg_basket_val = np.random.uniform(60, 200)
        elif persona == 'At_Risk':
            n_orders = np.random.randint(3, 7)
            recency_days = np.random.randint(120, 280)
            avg_basket_val = np.random.uniform(80, 220)
        else: # Lost
            n_orders = np.random.choice([1, 2], p=[0.8, 0.2])
            recency_days = np.random.randint(220, 365)
            avg_basket_val = np.random.uniform(25, 90)
            
        cust_profiles[cid] = {
            'persona': persona,
            'n_orders': n_orders,
            'last_purchase_day': recency_days,
            'avg_basket_val': avg_basket_val
        }
        
    records = []
    invoice_counter = 536000
    
    product_catalog = [
        ('PROD-101', 'Smart Wireless Headphones', 'Electronics', 120.0),
        ('PROD-102', 'Ultra HD Action Camera', 'Electronics', 280.0),
        ('PROD-103', 'Mechanical Gaming Keyboard', 'Electronics', 95.0),
        ('PROD-104', 'Ergonomic Office Chair', 'Furniture', 210.0),
        ('PROD-105', 'Stainless Steel Water Flask', 'Home', 25.0),
        ('PROD-106', 'Luxury Aromatherapy Diffuser', 'Home', 45.0),
        ('PROD-107', 'Organic Cotton Hooded Sweatshirt', 'Apparel', 65.0),
        ('PROD-108', 'Performance Running Shoes', 'Apparel', 115.0),
        ('PROD-109', 'Professional Chef Knife Set', 'Kitchen', 140.0),
        ('PROD-110', 'High-Speed Immersion Blender', 'Kitchen', 55.0)
    ]
    
    countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 'Australia']
    country_weights = [0.60, 0.15, 0.10, 0.05, 0.05, 0.05]
    
    for cid, profile in cust_profiles.items():
        n_orders = profile['n_orders']
        last_recency = profile['last_purchase_day']
        country = np.random.choice(countries, p=country_weights)
        
        # Distribute order dates chronologically leading up to the recency date
        order_recencies = sorted(
            [last_recency] + list(np.random.randint(last_recency + 1, 365, size=n_orders - 1)),
            reverse=True
        )
        
        for r_days in order_recencies:
            invoice_no = f"INV-{invoice_counter}"
            invoice_counter += 1
            inv_date = reference_date - timedelta(days=int(r_days))
            
            # Number of items in this invoice (1 to 4 line items)
            n_items = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            for _ in range(n_items):
                p_id, p_name, p_cat, base_p = product_catalog[np.random.randint(len(product_catalog))]
                qty = int(np.random.choice([1, 2, 3, 4], p=[0.65, 0.22, 0.09, 0.04]))
                unit_price = round(base_p * np.random.uniform(0.9, 1.1), 2)
                line_total = round(qty * unit_price, 2)
                
                records.append({
                    'InvoiceNo': invoice_no,
                    'StockCode': p_id,
                    'Description': p_name,
                    'Category': p_cat,
                    'Quantity': qty,
                    'InvoiceDate': inv_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'UnitPrice': unit_price,
                    'LineTotal': line_total,
                    'CustomerID': cid,
                    'Country': country
                })
                
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_ecommerce_rfm_data()
    output_file = 'ecommerce_customer_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df):,} transactions for {df['CustomerID'].nunique():,} unique customers in {output_file}")
    print(df.head())
