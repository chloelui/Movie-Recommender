import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        cursor_factory=RealDictCursor
    )


def username_exists(username):
    """Check to see if username already exists to enforce unique usernames."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
    exists = cur.fetchone() is not None
    cur.close()
    conn.close()
    return exists

def create_user(username):
    """Create new unique username based on user input."""
    username = username.strip().lower()
    if username_exists(username):
        return None
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (username) VALUES (%s) RETURNING id", (username,))
    user_id = cur.fetchone()["id"]
    conn.commit()
    cur.close()
    conn.close()
    return user_id

def get_user_id(username):
    """Retrieve user's unique id."""
    username = username.strip().lower()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row["id"] if row else None


def log_recommendation(user_id, source_movie, recommended_movie, score):
    """Record each recommendation already made to user into history."""
    conn = get_connection()
    cur = conn.cursor()
    source_id = source_movie["id"] if source_movie else None
    source_title = source_movie["title"] if source_movie else None
    cur.execute("""
        INSERT INTO recommendation_history (user_id, source_movie_id, source_movie_title, recommended_movie_id, recommended_movie_title, score)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (user_id, source_id, source_title, recommended_movie["id"], recommended_movie["title"], score))
    conn.commit()
    cur.close()
    conn.close()


def get_watched_movie_ids(user_id):
    """Get set of unique movies user already watched to prevent them from being recommended."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT movie_id FROM user_movie_interactions WHERE user_id = %s AND watched = TRUE", (user_id,))
    watched = {row["movie_id"] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return watched


def get_disliked_movie_ids(user_id):
    """Get set of unique movies that user dislikes to prevent them from being recommended."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT movie_id FROM user_movie_interactions WHERE user_id = %s AND liked = FALSE", (user_id,))
    disliked = {row["movie_id"] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return disliked


def record_feedback(user_id, movie_id, movie_title, watched=None, rating=None, liked=None):
    """Add or update user feedback for a mentioned movie."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO user_movie_interactions (user_id, movie_id, movie_title, watched, rating, liked)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id, movie_id)
        DO UPDATE SET
            watched = COALESCE(EXCLUDED.watched, user_movie_interactions.watched),
            rating = COALESCE(EXCLUDED.rating, user_movie_interactions.rating),
            liked = COALESCE(EXCLUDED.liked, user_movie_interactions.liked),
            updated_at = now()
    """, (user_id, movie_id, movie_title, watched, rating, liked))
    conn.commit()
    cur.close()
    conn.close()


def record_detail_view(user_id, movie_id, movie_title):
    """Record that user asked to see more details about movie."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""INSERT INTO movie_detail_views (user_id, movie_id, movie_title) VALUES (%s, %s, %s)""", (user_id, movie_id, movie_title))
    conn.commit()
    cur.close()
    conn.close()