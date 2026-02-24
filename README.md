# WanderWise - Intelligent Travel Stay Recommender

WanderWise is a smart recommendation system that suggests personalized accommodations (hotels, apartments, etc.) based on your interests, budget, and nationality.

Built as part of a graduation project using TF-IDF + Cosine Similarity + budget filtering + Streamlit web interface.

## Features
- Text-based interest matching (e.g., "culture museum history", "beach relaxation", "adventure hiking")
- Daily budget filtering
- Personalized recommendations with city, room type, price, rating, and description
- Clean and modern Streamlit user interface

## Demo
(إذا رفعتي الواجهة على Streamlit Cloud، حطي الرابط هنا لاحقًا)
Live Demo: https://your-app-name.streamlit.app (to be added)

## Project Structure

├── app.py                  # Streamlit web interface
├── preprocessing.py        # Data cleaning & TF-IDF vectorizer
├── training_and_inference.py  # Neural model training (optional)
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── final_df.csv
│   ├── item_vectors.npy
│   └── neural_recommend_model.keras (optional)
├── requirements.txt
└── README.md


## How to Run Locally

### Prerequisites
- Python 3.9+
- Git (optional)

### Step-by-step

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/WanderWise.git
   cd WanderWise

Create & activate virtual environment (recommended)

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run app.py

The app will open in your browser at: http://localhost:8501

Requirements (requirements.txt content)

streamlit
pandas
numpy
scikit-learn
joblib

How to generate the models (if you want to rebuild)

Place your datasets in the appropriate folder:
Travel details dataset.csv
Listings.csv

Run

python preprocessing.py

(Optional) Train neural model

python training_and_inference.py

python training_and_inference.py

