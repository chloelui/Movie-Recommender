import { useEffect, useState, useCallback } from 'react';
import * as api from './api';
import Kpis from './components/Kpis';
import FunnelChartComponent from './components/FunnelChart';
import QuintileChart from './components/QuintileChart';
import TopMovies from './components/TopMovies';
import RecentFeedback from './components/RecentFeedback';
import './App.css';

export default function App() {
  // Give each React component memory
  const [kpis, setKpis] = useState(null);   // [current value, updating function]
  const [funnel, setFunnel] = useState(null);
  const [quintiles, setQuintiles] = useState(null);
  const [topMovies, setTopMovies] = useState(null);
  const [recent, setRecent] = useState(null);

  const loadAll = useCallback(async () => {
    const [k, f, q, t, r] = await Promise.all([     // 5 simultaneous API calls
      api.getKpis(),
      api.getFunnel(),
      api.getQuintiles(),
      api.getTopMovies(),
      api.getRecentFeedback(),
    ]);
    setKpis(k);     // store results and re-render React components
    setFunnel(f);
    setQuintiles(q);
    setTopMovies(t);
    setRecent(r);
  }, []);

  // Load data on page load
  useEffect(() => {
    loadAll();
    // Auto-reload data every 30 secs for live updates
    const interval = setInterval(loadAll, 30000);
    return () => clearInterval(interval);
  }, [loadAll]);

  // Compose everything onto page
  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Movie Chatbot Analytics Dashboard</h1>
        <button onClick={loadAll}>Refresh now</button>
      </div>

      <Kpis data={kpis} />

      <section>
        <h2>Recommendation funnel</h2>
        <p className="caption">"Clicked" = user asked for details on a movie they were recommended.</p>
        <FunnelChartComponent data={funnel} />
      </section>

      <section>
        <h2>Does higher score predict better outcome?</h2>
        <p className="caption">Quintile 5 = highest-scored recommendations. If the algorithm works, rates should climb toward quintile 5.</p>
        <QuintileChart data={quintiles} />
      </section>

      <section>
        <h2>Most-recommended movies</h2>
        <TopMovies data={topMovies} />
      </section>

      <section>
        <h2>Recent feedback</h2>
        <RecentFeedback data={recent} />
      </section>
    </div>
  );
}