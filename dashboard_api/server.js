require('dotenv').config();
const express = require('express');
const cors = require('cors');
const queries = require('./queries');

// Create web server to get requests from browser
const app = express();      
app.use(cors());

// Query SQL and send back requested data in JSON files
app.get('/api/kpis', async (req, res) => {
  try {
    res.json(await queries.getKpis());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load KPIs' });
  }
});

app.get('/api/funnel', async (req, res) => {
  try {
    res.json(await queries.getFunnelCounts());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load funnel' });
  }
});

app.get('/api/quintiles', async (req, res) => {
  try {
    res.json(await queries.getScoreQuintilePerformance());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load quintiles' });
  }
});

app.get('/api/top-movies', async (req, res) => {
  try {
    res.json(await queries.getTopMovies());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load top movies' });
  }
});

app.get('/api/recent-feedback', async (req, res) => {
  try {
    res.json(await queries.getRecentFeedback());
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to load recent feedback' });
  }
});

// Start server to get incoming requests on port
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`Dashboard API running on http://localhost:${PORT}`));     