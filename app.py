import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# ========================
# Paths (update if your models are saved elsewhere)
# ========================
MODELS_PATH = r'E:\Projects\Travel_Recommender\models'  # ← تأكدي إن ده المسار الصح

# ========================
# Load models and data (cached for fast loading)
# ========================
@st.cache_resource
def load_models():
    try:
        vectorizer = joblib.load(os.path.join(MODELS_PATH, 'tfidf_vectorizer.joblib'))
        final_df = pd.read_csv(os.path.join(MODELS_PATH, 'final_df.csv'))
        item_vectors = vectorizer.transform(final_df['text_features'])
        return vectorizer, final_df, item_vectors
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error(f"Check if files exist in: {MODELS_PATH}")
        st.error("Run preprocessing.py if files are missing.")
        return None, None, None

vectorizer, final_df, item_vectors = load_models()

if vectorizer is None or final_df is None or item_vectors is None:
    st.stop()

# ========================
# Recommendation function – fresh calculation every time
# ========================
def recommend_items(interests, budget, nationality, final_df, item_vectors, vectorizer, top_n=5):
    user_text = interests + ' ' + nationality
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, item_vectors).flatten() * 2.0  # Boost similarity weight
    
    df_copy = final_df.copy()
    df_copy['similarity'] = similarities
    
    # Show price statistics for debugging/understanding
    st.write(f"**Selected Budget:** ${budget}")
    st.write(f"**Total places before filter:** {len(df_copy)}")
    st.write(f"**Min price in data:** ${df_copy['price_usd'].min():.2f}")
    st.write(f"**Max price in data:** ${df_copy['price_usd'].max():.2f}")
    st.write(f"**Average price in data:** ${df_copy['price_usd'].mean():.2f}")
    
    # Budget filter
    filtered = df_copy[df_copy['price_usd'] <= budget]
    
    st.write(f"**Places after budget filter:** {len(filtered)}")
    
    if filtered.empty:
        st.warning("No places match your budget. Showing all results instead.")
        filtered = df_copy.copy()
    
    return filtered.sort_values('similarity', ascending=False).head(top_n)

# ========================
# Streamlit App – Modern Travel Style
# ========================
st.set_page_config(
    page_title="WanderWise – Smart Travel Recommendations",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for travel app feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { 
        background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)), 
                          url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?q=80&w=2073&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stButton > button { 
        background-color: #1e88e5; 
        color: white; 
        border-radius: 12px; 
        padding: 14px 28px; 
        font-weight: bold;
        font-size: 18px;
    }
    .stButton > button:hover { background-color: #1565c0; }
    .card { 
        background-color: rgba(255, 255, 255, 0.95); 
        border-radius: 16px; 
        padding: 24px; 
        box-shadow: 0 8px 20px rgba(0,0,0,0.15); 
        margin-bottom: 24px;
        border-left: 6px solid #1e88e5;
    }
    h1, h2, h3 { color: #0d47a1; }
    .stTextInput > div > div > input, .stTextInput > label { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-take-off.png", width=90)
    st.title("WanderWise")
    st.markdown("**Discover perfect stays tailored to you**")

# Main Header
st.title("✈️ WanderWise – Intelligent Travel Recommendations")
st.markdown("Tell us your interests and budget – get personalized stay suggestions worldwide!")

# Input Section
with st.container():
    st.subheader("Your Preferences")
    col1, col2 = st.columns([3, 2])

    with col1:
        interests = st.text_input(
            "What do you love doing on trips?",
            value="culture museum history",
            placeholder="e.g. culture museum history, beach relaxation, adventure hiking, luxury spa...",
            key="interests_input"
        )

    with col2:
        budget = st.slider("Daily Budget (USD)", 50, 3000, 800, step=50, key="budget_slider")
        nationality = st.text_input("Your Nationality (optional)", "Egyptian", key="nationality_input")
        top_n = st.slider("How many suggestions?", 3, 10, 5, key="top_n_slider")

    discover_button = st.button("Find My Perfect Stays 🌴", type="primary", use_container_width=True)

# Process Recommendations
if discover_button:
    if not interests.strip():
        st.warning("Please tell us your interests first 😊")
    else:
        with st.spinner("Searching for your ideal accommodations..."):
            recommendations = recommend_items(
                interests, budget, nationality, final_df, item_vectors, vectorizer, top_n=top_n
            )

            if recommendations.empty:
                st.error("No suitable accommodations found 😔 Try a higher budget or different interests.")
            else:
                st.success(f"Found {len(recommendations)} wonderful options for you!")

                for i, row in recommendations.iterrows():
                    st.markdown(f"""
                        <div class="card">
                            <h3>{i+1}. {row['city'].title()} – {row['room_type']}</h3>
                            <p><strong>Daily Price:</strong> ${row['price_usd']:.2f}</p>
                            <p><strong>Average Rating:</strong> {row['average_rating']:.2f} ⭐</p>
                            <p><strong>Description:</strong> {row['description']}</p>
                        </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Built with TF-IDF + Cosine Similarity | Graduation Project – Alaa")