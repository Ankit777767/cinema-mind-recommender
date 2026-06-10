import os
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd 

# 1. Setup Paths
BASE_DIR = os.getcwd()
DATA_FILE = os.path.join(BASE_DIR, "data", "movies_raw.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "movie_embeddings.parquet")

def get_embeddings():
  print(f"Loading raw data from: {DATA_FILE}")

  df = pd.read_json(DATA_FILE, lines=True)
  #drop if any movie doesnt have the content, otherwise it will break the embedding process

  df = df.dropna(subset=['overview']).copy()

  #3. Feature engineering point to be noted, we not only give the overview for embeddings, 
  #instead we will give title + overview, beacuse title adds semantic weight
  df['text_to_embed'] = df['title'] + ". " + df['overview']

  print("Loading HuggingFace Transformer Model...")

  model = SentenceTransformer('all-MiniLM-L6-v2')
  print(f"Converting {len(df)} movies into math... This might take a minute.")

  embeddings = model.encode(df['text_to_embed'].tolist(),show_progress_bar=True)
  
  # Store the massive arrays of numbers back into our DataFrame
  df['embedding'] = embeddings.tolist()

  # 5. Save to Disk
  print(f"Saving to highly compressed Parquet format at: {OUTPUT_FILE}")
  # Interview point: Parquet preserves data types (like our arrays) and loads 
  # significantly faster than CSVs while taking up less disk space.
  df.to_parquet(OUTPUT_FILE, index=False)
  
  print("✅ Embeddings generated and saved successfully!")

if __name__ == "__main__":
    get_embeddings()
