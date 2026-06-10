import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# 1. Setup Paths
BASE_DIR = os.getcwd()
QDRANT_DB_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage")
model = SentenceTransformer('all-MiniLM-L6-v2')

def search_movies(user_query, top_k = 3):
  print(f"\n🧠 Analyzing your prompt: '{user_query}'...")
  # 2. Load the exact same AI model we used to embed the movies
  # Interview point: The user query MUST be mapped into the exact same 
  # vector space as the database, otherwise the math won't align.

  query_vector = model.encode(user_query).tolist()
  print("⚡ Searching the Vector Database...")
  client = QdrantClient(path=QDRANT_DB_PATH)

  # Perform CosineSimilarity Search
  search_results = client.query_points(
    collection_name = "movies",
    query=query_vector,
    using = "text",
    limit = top_k

  )
  # 4. Display Results
  print("\n🍿 HERE ARE YOUR RECOMMENDATIONS 🍿\n" + "="*45)
  for hit in search_results.points:
    title = hit.payload['title']
    score = round(hit.score*100,1)
    overview = hit.payload['overview']
        
    print(f"🎬 {title} (Match Score: {score}%)")
    print(f"📝 {overview[:150]}...\n")
if __name__ == "__main__":
  # Change this string to whatever emotion or vibe you want to test!
  my_prompt = "One Wish Willow"
  search_movies(my_prompt)
    