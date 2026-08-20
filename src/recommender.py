import csv

# Load movie dataset
def load_movies():
    with open("data/movies.csv", "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Get number of shared genres b/w two movies
def genre_similarity(movie_a, movie_b):            
    genres_a = set(movie_a["genres"].split("|"))
    genres_b = set(movie_b["genres"].split("|"))

    return len(genres_a & genres_b)   


movies = load_movies()

title = input("Enter a movie name: ")

matches = [movie for movie in movies if movie["title"].lower() == title.lower()]

if not matches:
    print("Movie not found in dataset.")
    exit()

target = matches[0]

recommendations = []

for movie in movies:
    if movie["id"] == target["id"]:
        continue

    # More similar genres gives larger score
    score = genre_similarity(target, movie)
    recommendations.append((score, movie))

# Sort recs by descending genre similarity
recommendations.sort(key=lambda x:x[0], reverse=True)

# Give top 5 recommendations
print(f"\nBecause you liked {target['title']}, we thought you might like:\n")
for score, movie in recommendations[:5]:
    print(f"{movie['title']} (genre similarity: {score})")