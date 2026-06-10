import os
import streamlit as st 
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
QDRANT_DB_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
POSTERS_DIR = os.path.join(BASE_DIR, "data", "posters")

st.set_page_config(page_title="Cinema Mind", page_icon="🍿", layout="wide")

#MODEL and DB Initialization for high speed
@st.cache_resource
def load_ai_model():
  return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_db_client():
  return QdrantClient(path=QDRANT_DB_PATH)

model = load_ai_model()
client = load_db_client()

#THE UI LAYOUT

st.title("🧠 Cinema Mind Recommender")
st.markdown("### Stop searching by genre. Search by feeling.")

#USERINPUT
user_query = st.text_input(
  "What kind of movie you are looking for",
  placeholder="e.g, A dark chilling thriller in love theme"
)

#Search Logic
if user_query:
  with st.spinner("Analyzing the vibe:"):
    #convert strings to math
    query_vector = model.encode(user_query).tolist()
    # Search Database
    search_results = client.query_points(
        collection_name="movies",
        query = query_vector,
        using = "text",
        limit=3 # Get top 3
    )
    st.write("---")
    st.subheader("🍿 Your Matches")
    # Create 3 clean columns for our results
    cols = st.columns(3)
  
 
    for index, hit in enumerate(search_results.points):
        with cols[index]:
            # Calculate match percentage
            score = round(hit.score * 100, 1)
            
            # Fetch Data
            title = hit.payload['title']
            overview = hit.payload['overview']
            poster_filename = f"{hit.id}.jpg"
            poster_path = os.path.join(POSTERS_DIR, poster_filename)
            
            # Display Image
            if os.path.exists(poster_path):
                st.image(poster_path, use_container_width=True)
            else:
                st.markdown("*(No Poster Available)*")
            
            # Display Text
            st.markdown(f"**{title}**")
            st.caption(f"✨ Match Score: {score}%")
            # Show only the first 150 characters of the plot to keep it clean
            st.write(f"{overview[:150]}...")



