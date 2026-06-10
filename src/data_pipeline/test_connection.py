#basic import
import os
import requests
from dotenv import load_dotenv

#load secret api_key
load_dotenv()
api_key = os.getenv("TMDB_API_KEY")

#test for movie=inception
movie_id = 27205
url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"

# get the response from the url
print("Connecting to TMDB...")
response = requests.get(url)

# 4. Check if it worked!
if response.status_code == 200:
    data = response.json()# important data = response=requests.url(f"https/{}/{}").json()
    print("\n✅ Success! Connection established.")
    print(f"Movie Title: {data['title']}")
    print(f"Tagline: {data['tagline']}")
    print(f"Poster Path: {data['poster_path']}")
else:
    print(f"\n❌ Failed to connect. Error code: {response.status_code}")
    print(response.text)