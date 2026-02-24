import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# ========================
# مسارات (عدّليها حسب مكان ملفاتك الفعلي)
# ========================
data_dir = r'E:\Projects\Travel_Recommender\data'  # المجلد الرئيسي
models_dir = os.path.join(data_dir, r'E:\Projects\Travel_Recommender\models')
os.makedirs(models_dir, exist_ok=True)

travel_path = os.path.join(data_dir, r'E:/Projects/Travel details dataset.csv')
listings_path = os.path.join(data_dir, r'E:/Projects/Airbnb Data/Listings.csv')  # لو في مجلد Airbnb Data، غيّري لـ r'E:\Projects\Airbnb Data\Listings.csv'

# ========================
# تحميل البيانات
# ========================
print("Loading data...")
travel_df = pd.read_csv(travel_path)
listings_df = pd.read_csv(listings_path, encoding='latin1', low_memory=False)

# ========================
# معالجة travel_df
# ========================
travel_df['city'] = travel_df['Destination'].str.split(',').str[0].str.lower().str.strip()
travel_df['room_type'] = travel_df['Accommodation type'].map({
    'Hotel': 'Entire home/apt',
    'Resort': 'Entire home/apt',
    'Villa': 'Entire home/apt',
    'Airbnb': 'Private room',
    'Hostel': 'Shared room'
}).fillna('Other')

travel_df['Accommodation cost'] = pd.to_numeric(travel_df['Accommodation cost'], errors='coerce')
travel_df['price_usd_daily'] = travel_df['Accommodation cost'] / travel_df['Duration (days)']
travel_df = travel_df.dropna(subset=['city', 'Traveler age', 'price_usd_daily'])

# إضافة صفوف وهمية لتحسين التنوع
dummy_rows = [
    {'city': 'paris', 'room_type': 'Entire home/apt', 'price_usd_daily': 100, 'Traveler age': 25, 'Traveler nationality': 'American'},
    {'city': 'rio de janeiro', 'room_type': 'Entire home/apt', 'price_usd_daily': 100, 'Traveler age': 30, 'Traveler nationality': 'Brazilian'},
    {'city': 'sydney', 'room_type': 'Entire home/apt', 'price_usd_daily': 150, 'Traveler age': 35, 'Traveler nationality': 'Australian'},
    {'city': 'bali', 'room_type': 'Private room', 'price_usd_daily': 50, 'Traveler age': 28, 'Traveler nationality': 'Indonesian'},
    {'city': 'auckland', 'room_type': 'Entire home/apt', 'price_usd_daily': 120, 'Traveler age': 32, 'Traveler nationality': 'New Zealander'}
]
travel_df = pd.concat([travel_df, pd.DataFrame(dummy_rows)], ignore_index=True)

print("Travel cities:", travel_df['city'].unique())
print("Travel room types:", travel_df['room_type'].unique())

# ========================
# معالجة listings_df
# ========================
listings_df['city'] = listings_df['host_location'].str.split(',').str[0].str.lower().str.strip().fillna('unknown')
listings_df['average_rating'] = listings_df['review_scores_rating'] / 20
listings_df['price_usd'] = listings_df['price']
listings_df['description'] = listings_df.get('summary', listings_df['name']).str.lower().str.strip().fillna('default listing')
listings_df['room_type'] = listings_df['room_type'].replace('Entire place', 'Entire home/apt')
listings_df = listings_df[listings_df['room_type'].isin(['Entire home/apt', 'Private room', 'Shared room'])]
listings_df = listings_df.dropna(subset=['city', 'price_usd', 'average_rating', 'description', 'room_type'])
listings_df = listings_df[listings_df['price_usd'] <= 1000]
listings_df = listings_df.head(50000)

print("Listings cities:", listings_df['city'].unique())
print("Listings room types:", listings_df['room_type'].unique())

# ========================
# الدمج
# ========================
merged_df = pd.merge(travel_df, listings_df, on=['city', 'room_type'], how='left')
merged_df = merged_df.groupby('city').head(500)

if merged_df.empty:
    print("Warning: Merge failed → using fallback from travel_df")
    final_df = travel_df[['city', 'room_type', 'price_usd_daily', 'Traveler age', 'Traveler nationality']].copy()
    final_df['description'] = 'default accommodation'
    final_df = final_df.rename(columns={'price_usd_daily': 'price_usd'})  # غيّر الاسم عشان يبقى موحد
    final_df['average_rating'] = 3.0
else:
    final_df = merged_df[['city', 'room_type', 'price_usd', 'average_rating', 'description']].copy()
    # لو فيه price_usd_daily في merged_df، نستخدمه كـ fallback للقيم الفاضية
    if 'price_usd_daily' in merged_df.columns:
        final_df['price_usd'] = final_df['price_usd'].fillna(merged_df['price_usd_daily'])
    final_df['average_rating'] = final_df['average_rating'].fillna(3.0)

#final_df = final_df.drop_duplicates().dropna(subset=['city', 'room_type', 'description'])
# Remove duplicates
final_df = final_df.drop_duplicates(subset=['description', 'city', 'room_type', 'price_usd', 'average_rating'], keep='first')
final_df = final_df.dropna(subset=['city', 'room_type', 'description'])

# أضيفي هنا:
final_df['average_rating'] = np.log1p(final_df['average_rating'])  # <--- السطر الجديد

# Debug بعد التحويل (اختياري، عشان تتأكدي)
print("Rating distribution after log transformation:")
print(final_df['average_rating'].describe())

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=50000, min_df=1)
final_df['text_features'] = final_df['description'] + ' ' + final_df['city'] + ' ' + final_df['room_type'] + ' ' + final_df['average_rating'].astype(str)
final_df = final_df.reset_index(drop=True)
item_vectors = vectorizer.fit_transform(final_df['text_features'])
print(f"Final shape after cleaning: {final_df.shape}")
print("City distribution:\n", final_df['city'].value_counts().head(10))
print("Average rating stats:\n", final_df['average_rating'].describe())

# ========================
# TF-IDF Vectorization
# ========================
vectorizer = TfidfVectorizer(stop_words='english', max_features=30000, min_df=1)
final_df['text_features'] = final_df['description'] + ' ' + final_df['city'] + ' ' + final_df['room_type']
item_vectors = vectorizer.fit_transform(final_df['text_features'])

# حفظ
joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
final_df.to_csv(os.path.join(models_dir, 'final_df.csv'), index=False)
np.save(os.path.join(models_dir, 'item_vectors.npy'), item_vectors.toarray())

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
print(f"Vectorizer saved → {os.path.join(models_dir, 'tfidf_vectorizer.joblib')}")
print(f"Final data saved → {os.path.join(models_dir, 'final_df.csv')}")
print(f"Item vectors shape: {item_vectors.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")