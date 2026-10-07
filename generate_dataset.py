"""
Script to generate a rich, realistic Retail Sales Dataset for EDA.
Spans 2023-01-01 to 2024-12-31 with realistic seasonal variations,
demographics, product categories, and transaction metrics.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_retail_data(n_records=3000, random_state=42):
    np.random.seed(random_state)
    
    # 1. Product Catalog Definition
    products_catalog = {
        'Electronics': [
            {'name': 'Smartphone Pro Max', 'base_price': 850, 'price_std': 50},
            {'name': 'Wireless Noise-Canceling Earbuds', 'base_price': 150, 'price_std': 20},
            {'name': 'Ultra HD 4K Smart TV', 'base_price': 650, 'price_std': 40},
            {'name': 'Gaming Laptop 16GB', 'base_price': 1200, 'price_std': 100},
            {'name': 'Fitness Smartwatch', 'base_price': 220, 'price_std': 25},
            {'name': 'Bluetooth Portable Speaker', 'base_price': 85, 'price_std': 10}
        ],
        'Clothing': [
            {'name': 'Classic Denim Jacket', 'base_price': 75, 'price_std': 8},
            {'name': 'Premium Cotton T-Shirt', 'base_price': 30, 'price_std': 5},
            {'name': 'Athletic Running Shoes', 'base_price': 110, 'price_std': 15},
            {'name': 'Formal Wool Blazer', 'base_price': 180, 'price_std': 20},
            {'name': 'Silk Evening Dress', 'base_price': 140, 'price_std': 15},
            {'name': 'Casual Chino Pants', 'base_price': 55, 'price_std': 7}
        ],
        'Beauty': [
            {'name': 'Anti-Aging Facial Serum', 'base_price': 65, 'price_std': 8},
            {'name': 'Luxury Floral Perfume', 'base_price': 125, 'price_std': 15},
            {'name': 'Hydrating Matte Lipstick', 'base_price': 28, 'price_std': 4},
            {'name': 'Ionic Hair Dryer Pro', 'base_price': 95, 'price_std': 10},
            {'name': 'Organic Sunscreen SPF 50', 'base_price': 35, 'price_std': 5},
            {'name': 'Exfoliating Face Scrub', 'base_price': 22, 'price_std': 3}
        ],
        'Home & Kitchen': [
            {'name': 'Digital Air Fryer XL', 'base_price': 130, 'price_std': 15},
            {'name': 'Espresso Coffee Machine', 'base_price': 280, 'price_std': 30},
            {'name': 'High-Speed Smoothie Blender', 'base_price': 90, 'price_std': 10},
            {'name': 'Non-Stick Cookware Set (10 Pc)', 'base_price': 175, 'price_std': 20},
            {'name': 'Robot Vacuum Cleaner', 'base_price': 320, 'price_std': 35},
            {'name': 'Cast Iron Dutch Oven', 'base_price': 80, 'price_std': 10}
        ],
        'Sports & Outdoors': [
            {'name': 'Eco-Friendly Yoga Mat', 'base_price': 45, 'price_std': 6},
            {'name': 'Adjustable Dumbbells Set', 'base_price': 190, 'price_std': 20},
            {'name': 'Waterproof Camping Tent', 'base_price': 210, 'price_std': 25},
            {'name': 'Insulated Thermal Water Bottle', 'base_price': 30, 'price_std': 4},
            {'name': 'High-Tension Resistance Bands', 'base_price': 25, 'price_std': 3},
            {'name': 'Mountain Trail Backpack', 'base_price': 95, 'price_std': 12}
        ]
    }
    
    categories = list(products_catalog.keys())
    cat_weights = [0.28, 0.24, 0.18, 0.16, 0.14]
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    days_span = (end_date - start_date).days
    
    records = []
    
    # Customer pool
    n_unique_customers = 850
    customer_ids = [f"CUST{i+1000:04d}" for i in range(n_unique_customers)]
    customer_genders = np.random.choice(['Female', 'Male'], size=n_unique_customers, p=[0.52, 0.48])
    customer_ages = np.random.normal(38, 13, size=n_unique_customers).astype(int)
    customer_ages = np.clip(customer_ages, 18, 70)
    
    cust_profile = {
        cid: {'gender': g, 'age': a} 
        for cid, g, a in zip(customer_ids, customer_genders, customer_ages)
    }
    
    payment_methods = ['Credit Card', 'Debit Card', 'UPI / Digital Wallet', 'Cash']
    payment_weights = [0.42, 0.23, 0.25, 0.10]
    
    for i in range(1, n_records + 1):
        # Transaction ID
        trans_id = f"TXN-{100000 + i}"
        
        # Pick customer
        cid = np.random.choice(customer_ids)
        gender = cust_profile[cid]['gender']
        age = cust_profile[cid]['age']
        
        # Generate date with realistic seasonal bias (higher sales in Nov-Dec and July)
        day_offset = np.random.randint(0, days_span + 1)
        tx_date = start_date + timedelta(days=day_offset)
        
        # Seasonality probability adjustment
        month = tx_date.month
        quarter = (month - 1) // 3 + 1
        
        # Category selection influenced slightly by demographic preferences
        if gender == 'Female' and age < 35:
            adj_weights = [0.22, 0.32, 0.28, 0.10, 0.08]
        elif gender == 'Male' and age < 35:
            adj_weights = [0.40, 0.20, 0.06, 0.12, 0.22]
        elif age >= 50:
            adj_weights = [0.20, 0.20, 0.14, 0.32, 0.14]
        else:
            adj_weights = cat_weights
            
        chosen_cat = np.random.choice(categories, p=adj_weights)
        prod_list = products_catalog[chosen_cat]
        prod_item = np.random.choice(prod_list)
        
        prod_name = prod_item['name']
        unit_price = round(max(10.0, np.random.normal(prod_item['base_price'], prod_item['price_std'])), 2)
        
        # Quantity purchased: mostly 1-3, occasionally 4-5
        quantity = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))
        
        # Holiday season discount (Nov, Dec, July)
        if month in [11, 12]:
            discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], p=[0.1, 0.15, 0.3, 0.25, 0.15, 0.05])
        elif month in [7]:
            discount_pct = np.random.choice([0, 5, 10, 15], p=[0.3, 0.3, 0.25, 0.15])
        else:
            discount_pct = np.random.choice([0, 5, 10], p=[0.6, 0.25, 0.15])
            
        gross_amount = round(quantity * unit_price, 2)
        discount_amount = round(gross_amount * (discount_pct / 100.0), 2)
        total_amount = round(gross_amount - discount_amount, 2)
        
        payment_method = np.random.choice(payment_methods, p=payment_weights)
        
        records.append({
            'Transaction_ID': trans_id,
            'Date': tx_date.strftime('%Y-%m-%d'),
            'Customer_ID': cid,
            'Gender': gender,
            'Age': age,
            'Product_Category': chosen_cat,
            'Product_Name': prod_name,
            'Quantity': quantity,
            'Price_per_Unit': unit_price,
            'Discount_Percent': discount_pct,
            'Total_Amount': total_amount,
            'Payment_Method': payment_method
        })
        
    df = pd.DataFrame(records)
    # Sort chronologically
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date_dt').reset_index(drop=True)
    df = df.drop(columns=['Date_dt'])
    
    # Recalculate Transaction_ID strictly sequentially
    df['Transaction_ID'] = [f"TXN-{100001 + idx}" for idx in range(len(df))]
    
    return df

if __name__ == '__main__':
    df = generate_retail_data(n_records=3500)
    output_path = 'retail_sales_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at {output_path} with {len(df)} records and {df.shape[1]} columns.")
    print("\nFirst 5 rows:")
    print(df.head())
