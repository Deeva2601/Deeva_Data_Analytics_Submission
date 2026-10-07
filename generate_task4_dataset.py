"""
Script to generate a rich, authentic 3-Class Customer Sentiment Dataset for Task 4.
Contains text reviews across Positive, Negative, and Neutral sentiment classes with realistic linguistic variety.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sentiment_data(n_samples=3600, random_state=42):
    np.random.seed(random_state)
    
    pos_templates = [
        "Absolutely love this {product}! The {aspect} is top-notch and exceeded my expectations.",
        "Great quality {product}. Works flawlessly and the {aspect} is simply amazing.",
        "Best purchase I have made all year! Highly recommend this {product} to everyone.",
        "Impressive performance and stylish design. Very satisfied with the quick delivery and {aspect}.",
        "Outstanding value for the money. The {product} feels premium and durable.",
        "Five stars! Excellent build quality, seamless setup, and brilliant {aspect}.",
        "Super happy with this item! Works better than advertised and very easy to use.",
        "Fantastic experience. The customer service was helpful and the {product} is perfect.",
        "Incredible product! Totally worth every penny. Super fast shipping and great {aspect}.",
        "I am genuinely blown away by how good this {product} is. Will buy again!"
    ]
    
    neg_templates = [
        "Extremely disappointed with this {product}. The {aspect} is completely defective and cheap.",
        "Worst purchase ever. Stopped working after 3 days and customer service refused a refund.",
        "Poor quality material and terrible design. Does not match the description at all.",
        "Waste of money! The {product} broke immediately. Do not buy this garbage.",
        "Very frustrated with this order. Arrived late, damaged packaging, and awful {aspect}.",
        "Total scam. Horrible performance, constantly glitching, and very slow.",
        "Regret buying this. Overpriced junk with terrible customer support.",
        "Zero stars if I could. The {product} failed on the first day. Extremely annoying.",
        "Inferior build quality. The {aspect} feels fragile and useless.",
        "Defective out of the box! Returning it right now for a full refund."
    ]
    
    neu_templates = [
        "The {product} arrived on Tuesday in standard packaging. Included cable and manual.",
        "Average item. It functions as expected, neither particularly good nor bad.",
        "Received the {product} yesterday. It matches the specifications listed on the page.",
        "Ordered the black color variant. Delivery took 4 business days.",
        "The {product} is okay. Does what it says on the box. Nothing extraordinary.",
        "Standard quality for this price point. Moderate performance in everyday use.",
        "The package contains one {product} and standard accessories. As described.",
        "Decent product. Meets the baseline requirements, but nothing to write home about.",
        "Normal purchase experience. Arrived within the estimated delivery window.",
        "The {product} dimensions are as stated in the manual. Fairly average overall."
    ]
    
    products = ['wireless earbuds', 'smartwatch', 'gaming laptop', 'espresso maker', 'air fryer', 'denim jacket', 'office chair', 'bluetooth speaker', 'running shoes', 'skincare serum']
    aspects = ['battery life', 'sound quality', 'build material', 'display clarity', 'finishing', 'performance', 'ergonomics', 'reliability', 'packaging', 'user interface']
    
    categories = ['Electronics', 'Home & Kitchen', 'Apparel', 'Beauty', 'Office Supplies']
    
    records = []
    sentiments = ['Positive', 'Negative', 'Neutral']
    n_per_class = n_samples // 3
    
    # 1. Generate Positive Reviews
    for _ in range(n_per_class):
        tmpl = np.random.choice(pos_templates)
        prod = np.random.choice(products)
        asp = np.random.choice(aspects)
        text = tmpl.format(product=prod, aspect=asp)
        records.append({
            'Review_ID': f"REV-{len(records)+10001}",
            'Category': np.random.choice(categories),
            'Review_Text': text,
            'Sentiment': 'Positive',
            'Rating': int(np.random.choice([4, 5], p=[0.25, 0.75]))
        })
        
    # 2. Generate Negative Reviews
    for _ in range(n_per_class):
        tmpl = np.random.choice(neg_templates)
        prod = np.random.choice(products)
        asp = np.random.choice(aspects)
        text = tmpl.format(product=prod, aspect=asp)
        records.append({
            'Review_ID': f"REV-{len(records)+10001}",
            'Category': np.random.choice(categories),
            'Review_Text': text,
            'Sentiment': 'Negative',
            'Rating': int(np.random.choice([1, 2], p=[0.75, 0.25]))
        })
        
    # 3. Generate Neutral Reviews
    for _ in range(n_per_class):
        tmpl = np.random.choice(neu_templates)
        prod = np.random.choice(products)
        asp = np.random.choice(aspects)
        text = tmpl.format(product=prod, aspect=asp)
        records.append({
            'Review_ID': f"REV-{len(records)+10001}",
            'Category': np.random.choice(categories),
            'Review_Text': text,
            'Sentiment': 'Neutral',
            'Rating': 3
        })
        
    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == '__main__':
    df = generate_sentiment_data(n_samples=3600)
    output_file = 'product_sentiment_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"Generated sentiment dataset with {len(df):,} reviews in {output_file}")
    print("\nClass distribution:")
    print(df['Sentiment'].value_counts())
