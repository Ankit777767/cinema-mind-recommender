import os
import pandas as pd 
from qdrant_client import QdrantClient
from qdrant_client.http import models

# 1. Setup Paths
BASE_DIR = os.getcwd()
TEXT_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "movie_embeddings.parquet")
VISUAL_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "data", "visual_embeddings.parquet")
QDRANT_DB_PATH = os.path.join(BASE_DIR, "data", "qdrant_storage") # Saves DB 

def build_vector_db():
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
  print(f"Merged successfully. Preparing {len(df)} movies for the Vector DB.")

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

  points=[]
  for index,row in df.iterrows():
    #payload is the metadata user gets after search
    payload ={
      "title":row["title"],
      "overview": row["overview"],
      "poster_path": row["poster_path"],
      "release_date": row["release_date"]
    }

    # Create a Qdrant Point
    point = models.PointStruct(
        id=int(row["id"]),
        vector={
            "text": row["embedding"],
            "visual": row["visual_embedding"],
        },
        payload=payload
    )
    points.append(point)

  # 5. Upload to the Database
  print("Uploading to Qdrant...")
  client.upsert(
      collection_name=collection_name,
      points=points
  )
# Let's verify it worked
  collection_info = client.get_collection(collection_name)
  print(f"✅ Success! Vector DB built. Total records in DB: {collection_info.points_count}")

if __name__ == "__main__":
    build_vector_db()



