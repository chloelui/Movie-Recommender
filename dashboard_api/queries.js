const pool = require('./db');

async function getKpis() {
  const users = await pool.query('SELECT COUNT(DISTINCT user_id) AS total_users FROM recommendation_history');
  const shown = await pool.query('SELECT COUNT(*) AS total_shown FROM recommendation_history');
  const rating = await pool.query('SELECT ROUND(AVG(rating)::numeric, 2) AS avg_rating FROM user_movie_interactions WHERE rating IS NOT NULL');
  return {
    total_users: Number(users.rows[0].total_users),
    total_shown: Number(shown.rows[0].total_shown),
    avg_rating: rating.rows[0].avg_rating,
  };
}

async function getFunnelCounts() {
  const result = await pool.query(`
    WITH shown AS (
      SELECT DISTINCT user_id, recommended_movie_id AS movie_id
      FROM recommendation_history
    ),
    clicked AS (
      SELECT DISTINCT dv.user_id, dv.movie_id
      FROM movie_detail_views dv
      JOIN shown s ON s.user_id = dv.user_id AND s.movie_id = dv.movie_id
    ),
    watched AS (
      SELECT DISTINCT umi.user_id, umi.movie_id
      FROM user_movie_interactions umi
      JOIN shown s ON s.user_id = umi.user_id AND s.movie_id = umi.movie_id
      WHERE umi.watched = TRUE
    ),
    liked AS (
      SELECT DISTINCT umi.user_id, umi.movie_id
      FROM user_movie_interactions umi
      JOIN shown s ON s.user_id = umi.user_id AND s.movie_id = umi.movie_id
      WHERE umi.liked = TRUE
    ),
    rated AS (
      SELECT DISTINCT umi.user_id, umi.movie_id
      FROM user_movie_interactions umi
      JOIN shown s ON s.user_id = umi.user_id AND s.movie_id = umi.movie_id
      WHERE umi.rating IS NOT NULL
    )
    SELECT
      (SELECT COUNT(*) FROM shown)   AS shown,
      (SELECT COUNT(*) FROM clicked) AS clicked,
      (SELECT COUNT(*) FROM watched) AS watched,
      (SELECT COUNT(*) FROM liked)   AS liked,
      (SELECT COUNT(*) FROM rated)   AS rated
  `);
  const row = result.rows[0];
  return [
    { name: 'Shown', value: Number(row.shown) },
    { name: 'Clicked', value: Number(row.clicked) },
    { name: 'Watched', value: Number(row.watched) },
    { name: 'Liked', value: Number(row.liked) },
    { name: 'Rated', value: Number(row.rated) },
  ];
}

async function getScoreQuintilePerformance() {
  const result = await pool.query(`
    WITH shown AS (
      SELECT user_id, recommended_movie_id AS movie_id, MAX(score) AS score
      FROM recommendation_history
      GROUP BY user_id, recommended_movie_id
    ),
    ranked AS (
      SELECT *, NTILE(5) OVER (ORDER BY score) AS score_quintile
      FROM shown
    ),
    outcomes AS (
      SELECT r.*, umi.watched, umi.liked
      FROM ranked r
      LEFT JOIN user_movie_interactions umi
        ON umi.user_id = r.user_id AND umi.movie_id = r.movie_id
    )
    SELECT
      score_quintile,
      COUNT(*) AS total_shown,
      COUNT(*) FILTER (WHERE watched = TRUE) AS watched_count,
      COUNT(*) FILTER (WHERE liked = TRUE) AS liked_count,
      ROUND(AVG(score)::numeric, 2) AS avg_score
    FROM outcomes
    GROUP BY score_quintile
    ORDER BY score_quintile
  `);
  return result.rows.map(r => ({
    score_quintile: r.score_quintile,
    total_shown: Number(r.total_shown),
    watch_rate: r.total_shown > 0 ? Math.round((r.watched_count / r.total_shown) * 1000) / 10 : 0,
    like_rate: r.total_shown > 0 ? Math.round((r.liked_count / r.total_shown) * 1000) / 10 : 0,
    avg_score: r.avg_score,
  }));
}

async function getTopMovies(limit = 5) {
  const result = await pool.query(
    `SELECT recommended_movie_title AS title,
            COUNT(*) AS times_recommended,
            ROUND(AVG(score)::numeric, 2) AS avg_score
     FROM recommendation_history
     GROUP BY recommended_movie_title
     ORDER BY times_recommended DESC
     LIMIT $1`,
    [limit]
  );
  return result.rows;
}

async function getRecentFeedback(limit = 10) {
  const result = await pool.query(
    `SELECT u.username, umi.movie_title, umi.watched, umi.liked, umi.rating, umi.updated_at
     FROM user_movie_interactions umi
     JOIN users u ON u.id = umi.user_id
     ORDER BY umi.updated_at DESC
     LIMIT $1`,
    [limit]
  );
  return result.rows;
}

module.exports = {getKpis, getFunnelCounts, getScoreQuintilePerformance, getTopMovies, getRecentFeedback};