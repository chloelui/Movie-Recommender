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


# Access API and movies
def search_movies(query):
    url = f"{BASE_URL}/search/movie"

    params = {
        "query": query
    }

    response = requests.get(url,headers=HEADERS,params=params)
    response.raise_for_status()

    return response.json()["results"]


# Add movie details
def get_movie(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"

    response = requests.get(url,headers=HEADERS)
    response.raise_for_status()

    return response.json()


# Director/cast credits
def get_movie_credits(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/credits"

    response = requests.get(url,headers=HEADERS)
    response.raise_for_status()

    return response.json()


# Retrieve movies from TMDB's discover endpoint
def discover_movies(page=1, **kwargs):
    url = f"{BASE_URL}/discover/movie"

    params = {"page": page, "sort_by": "popularity.desc"}
    params.update(kwargs)   

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()

    return response.json()["results"]


# Get TMDB's list of genres
def get_genres():
    url = f"{BASE_URL}/genre/movie/list"

    response = requests.get(url,headers=HEADERS)
    response.raise_for_status()

    return response.json()["genres"]