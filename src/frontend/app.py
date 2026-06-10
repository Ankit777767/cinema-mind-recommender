import os
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# 1. Setup Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
QDRANT_DB_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
POSTERS_DIR = os.path.join(BASE_DIR, "data", "posters")

# Paths for our Cold Start Handler
TEXT_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "movie_embeddings.parquet")
VISUAL_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "visual_embeddings.parquet")

# 2. Page Configuration
st.set_page_config(page_title="Cinema Mind", page_icon="🍿", layout="wide")

# 3. Model & DB Initialization (Cached for blazing speed)
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_db_client():
    # Ensure the storage directory exists on the cloud server
    os.makedirs(QDRANT_DB_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_DB_PATH)
    
    # --- INTERVIEW FLEX: THE COLD START HANDLER ---
    try:
        # Check if the database collection actually loaded
        client.get_collection("movies")
    except Exception:
        # If it failed, the cloud server doesn't have the DB. Let's build it automatically!
        if not os.path.exists(TEXT_EMBEDDINGS_FILE):
            st.error("⚠️ Data files missing! Git did not upload your `data/` folder. Please ensure the parquet files are pushed to GitHub.")
            st.stop()
            
        print("Cold start triggered: Building Vector DB from Parquet files...")
        client.recreate_collection(
            collection_name="movies",
            vectors_config={
                "text": models.VectorParams(size=384, distance=models.Distance.COSINE),
                "visual": models.VectorParams(size=2048, distance=models.Distance.COSINE),
            }
        )
        
        # Load the raw math arrays we saved earlier
        text_df = pd.read_parquet(TEXT_EMBEDDINGS_FILE)
        visual_df = pd.read_parquet(VISUAL_EMBEDDINGS_FILE)
        df = pd.merge(text_df, visual_df, on="id", how="inner")
        
        # Package and upload to the local Qdrant instance
        points = []
        for index, row in df.iterrows():
            payload = {
                "title": row["title"],
                "overview": row["overview"],
                "poster_path": row["poster_path"],
                "release_date": row["release_date"]
            }
            points.append(models.PointStruct(
                id=int(row["id"]),
                vectors={"text": row["embedding"], "visual": row["visual_embedding"]},
                payload=payload
            ))
        client.upsert(collection_name="movies", points=points)
        print("Vector DB built successfully!")
        
    return client

model = load_ai_model()
client = load_db_client()

# 4. The UI Layout
st.title("🧠 Cinema Mind Recommender")
st.markdown("### Stop searching by genre. Search by *feeling*.")

# User Input
user_query = st.text_input("What kind of movie are you looking for today?", 
                           placeholder="e.g., A dark, visually stunning sci-fi with a hopeful ending...")

# 5. Search Logic
if user_query:
    with st.spinner("Analyzing the vibe..."):
        query_vector = model.encode(user_query).tolist()
        
        search_results = client.query_points(
            collection_name="movies",
            query=query_vector,
            using="text",
            limit=3
        )
        
        st.write("---")
        st.subheader("🍿 Your Matches")
        
        cols = st.columns(3)
        for index, hit in enumerate(search_results.points):
            with cols[index]:
                score = round(hit.score * 100, 1)
                title = hit.payload['title']
                overview = hit.payload['overview']
                poster_filename = f"{hit.id}.jpg"
                poster_path = os.path.join(POSTERS_DIR, poster_filename)
                
                if os.path.exists(poster_path):
                    st.image(poster_path, use_container_width=True)
                else:
                    st.markdown("*(No Poster Available)*")
                
                st.markdown(f"**{title}**")
                st.caption(f"✨ Match Score: {score}%")
                st.write(f"{overview[:150]}...")