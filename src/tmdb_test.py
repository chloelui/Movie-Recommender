import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TMDB_TOKEN")
url = "https://api.themoviedb.org/3/search/movie"
headers = {
    "Authorization": f"Bearer {token}"
}

params = {
    "query": "The Dark Knight"
}

response = requests.get(url,headers=headers,params=params)
data = response.json()
movies = data["results"]

for movie in movies:
    print(
        movie["id"],
        movie["title"],
        movie["release_date"],
        movie["vote_average"]
    )