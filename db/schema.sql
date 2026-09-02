CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_movie_interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    movie_id TEXT NOT NULL,
    movie_title TEXT NOT NULL,
    watched BOOLEAN DEFAULT FALSE,
    rating REAL,
    liked BOOLEAN,
    updated_at TIMESTAMP DEFAULT now(),
    UNIQUE(user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS recommendation_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    source_movie_id TEXT,
    source_movie_title TEXT,
    recommended_movie_id TEXT,
    recommended_movie_title TEXT,
    score REAL,
    recommended_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS movie_detail_views (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    movie_id TEXT NOT NULL,
    movie_title TEXT NOT NULL,
    viewed_at TIMESTAMP DEFAULT now()
);