import os
import time
import json
import requests
from dotenv import load_dotenv

# 1. Setup paths based on our new folder structure
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

# Get the absolute path to the root of our project
#BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "movies_raw.jsonl")

def fetch_movies(pages_to_fetch=2):
    print(f"Starting data pipeline. Saving to: {OUTPUT_FILE}")
    
    # Open the file in 'append' mode so we don't overwrite data if it crashes
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        
        for page in range(1, pages_to_fetch + 1):
            print(f"Fetching page {page}...")
            
            url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&sort_by=popularity.desc&page={page}&language=en-US"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Loop through each movie on this page
                for movie in data["results"]:
                    # We only save the fields we actually need for our ML models
                    movie_payload = {
                        "id": movie.get("id"),
                        "title": movie.get("title"),
                        "overview": movie.get("overview"), # For NLP (Emotions)
                        "genre_ids": movie.get("genre_ids"),
                        "poster_path": movie.get("poster_path"), # For Computer Vision
                        "release_date": movie.get("release_date")
                    }
                    
                    # Write it as a JSON line
                    f.write(json.dumps(movie_payload) + "\n")
                
                # Be polite to the API, wait a quarter of a second before the next page
                time.sleep(0.25)
                
            else:
                print(f"Error on page {page}: {response.status_code}")
                break

    print("Pipeline finished successfully! Check your data folder.")

if __name__ == "__main__":
    fetch_movies(pages_to_fetch=2)
