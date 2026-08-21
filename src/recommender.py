import csv

# Load movie dataset
def load_movies():
    with open("data/movies.csv", "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Get number of shared genres b/w two movies
def genre_similarity(movie_a, movie_b):            
    genres_a = set(movie_a["genres"].split("|"))
    genres_b = set(movie_b["genres"].split("|"))

    return len(genres_a & genres_b) if genres_a and genres_b else 0


# Cast similarity
def cast_similarity(movie_a, movie_b):
    cast_a = set(movie_a["cast"].split("|"))
    cast_b = set(movie_b["cast"].split("|"))

    return len(cast_a & cast_b) if cast_a and cast_b else 0


# Add rating bonus to score
def recommendation_score(target, movie):
    genre_score = genre_similarity(target, movie)
    cast_score = cast_similarity(target, movie)
    rating = float(movie["vote_average"])               # Higher-rated movies get higher score

    return round(genre_score * 2 + rating + cast_score * 0.5, 1)


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
    score = recommendation_score(target, movie)
    recommendations.append((score, movie))

# Sort recs by descending genre similarity
recommendations.sort(key=lambda x:x[0], reverse=True)

# Give top 5 recommendations
print(f"\nBecause you liked {target['title']}, we thought you might like:\n")
for score, movie in recommendations[:5]:
    print(f"{movie['title']} (genre similarity: {score})")