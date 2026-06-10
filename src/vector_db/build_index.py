import os
import pandas as pd 
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Setup Paths
BASE_DIR = os.getcwd()
TEXT_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "movie_embeddings.parquet")
VISUAL_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "visual_embeddings.parquet")
QDRANT_DB_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage") # Saves DB 

def build_vector_db(batch_size=500):
  print("Loading Parquet files...")
  # print(TEXT_EMBEDDINGS_FILE)
  # print(os.path.exists(TEXT_EMBEDDINGS_FILE))
  # print(os.path.getsize(TEXT_EMBEDDINGS_FILE))
  # Load our two sets of math
  text_df = pd.read_parquet(TEXT_EMBEDDINGS_FILE)
  visual_df = pd.read_parquet(VISUAL_EMBEDDINGS_FILE)
  #DATA CLEANING
  # merge data based on index column for both image and visual
  # simply drops the movie, if it doesnt contain the image
  df = pd.merge(text_df,visual_df,on='id',how='inner')#inner will drop if missing
  
  total_movies = len(df)
  print(f"Merged successfully. Preparing {total_movies} movies for the Vector DB.")

  # 3. Initialize Qdrant
  # By providing a path, Qdrant runs entirely on your local hard drive.
  client = QdrantClient(path=QDRANT_DB_PATH)
  collection_name = "movies"

  print("Configuring Qdrant Collection with Named Vectors...")

  #[NOTE] recreate_collection destroys the old DB if we run this twice, keeping things fresh

  client.recreate_collection(
    collection_name=collection_name,
    vectors_config={
      "text":models.VectorParams(size=384, distance=models.Distance.COSINE),
      "visual": models.VectorParams(size=2048,distance=models.Distance.COSINE)
    }
  )


  # 3. Batch Processing Loop
  print(f"Beginning batched upsert routines (Chunk Size: {batch_size})...")
  
  # Loop through the data frame in steps of batch_size
  for start_idx in range(0, total_movies, batch_size):
      end_idx = min(start_idx + batch_size, total_movies)
      batch_df = df.iloc[start_idx:end_idx]
      
      points = []
      for index, row in batch_df.iterrows():
          payload = {
              "title": row["title"],
              "overview": row["overview"],
              "poster_path": row["poster_path"],
              "release_date": row["release_date"]
          }
          
          # Ensure type safety for Pydantic schema validation
          raw_text_vector = row["embedding"].tolist() if hasattr(row["embedding"], "tolist") else list(row["embedding"])
          raw_visual_vector = row["visual_embedding"].tolist() if hasattr(row["visual_embedding"], "tolist") else list(row["visual_embedding"])
          
          clean_text_vector = [float(x) for x in raw_text_vector]
          clean_visual_vector = [float(x) for x in raw_visual_vector]

          points.append(models.PointStruct(
              id=int(row["id"]),
              vector={
                  "text": clean_text_vector,
                  "visual": clean_visual_vector
              },
              payload=payload
          ))

      # Upload this chunk safely to the disk engine
      client.upsert(
          collection_name=collection_name,
          points=points
      )
      print(f"  Successfully processed points {start_idx} through {end_idx}...")

  # Verification check
  collection_info = client.get_collection(collection_name)
  print(f"\n✅ Build complete! Total records indexed in local database: {collection_info.points_count}")


if __name__ == "__main__":
  build_vector_db(batch_size=500)