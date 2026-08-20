import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Base utilities
TMDB_TOKEN = os.getenv("TMDB_TOKEN")
BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}"
}

def search_movies(query):
    url = f"{BASE_URL}/search/movie"
    params = {
        "query": query
    }
    response = requests.get(url,headers=HEADERS,params=params)

    response.raise_for_status()

    return response.json()["results"]