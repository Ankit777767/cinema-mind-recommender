import os
import json
import time
import requests

# 1. Setup our paths
BASE_DIR = os.getcwd()
DATA_FILE = os.path.join(BASE_DIR, "data", "movies_raw.jsonl")
POSTERS_DIR = os.path.join(BASE_DIR, "data", "posters")

# TMDB base URL for images. 'w500' means we want a 500-pixel wide image.
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def download_movie_posters():
    print(f"Reading data from: {DATA_FILE}")
    print(f"Saving images to: {POSTERS_DIR}")

    # Check if we actually have data to read
    if not os.path.exists(DATA_FILE):
        print("Error: movies_raw.jsonl not found. Run data_fetcher.py first.")
        return

    # 2. Read the JSONL file line by line
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            movie = json.loads(line)
            movie_id = movie.get("id")
            poster_path = movie.get("poster_path")

            # Some movies don't have posters, we just skip them
            if not poster_path:
                print(f"Skipping {movie.get('title')} - No poster available.")
                continue

            # 3. Build the full URL and the local file path
            image_url = f"{IMAGE_BASE_URL}{poster_path}"
            image_filepath = os.path.join(POSTERS_DIR, f"{movie_id}.jpg")

            # Skip if we already downloaded it (useful if the script stops halfway)
            if os.path.exists(image_filepath):
                print(f"Already have {movie_id}.jpg")
                continue

            # 4. Download and save the image
            print(f"Downloading poster for {movie.get('title')}...")
            response = requests.get(image_url)

            if response.status_code == 200:
                # Write the raw binary data (wb) to a file
                with open(image_filepath, "wb") as img_file:
                    img_file.write(response.content)
            else:
                print(f"Failed to download image for {movie.get('title')}")

            # Wait a fraction of a second to be polite to the TMDB image servers
            time.sleep(0.1)

    print("Finished downloading posters!")

if __name__ == "__main__":
    download_movie_posters()