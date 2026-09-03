const BASE_URL = 'http://localhost:4000/api';

// Call backend server and retrieve data cleanly
async function getJson(path) {
  const res = await fetch(`${BASE_URL}/${path}`);
  if (!res.ok) throw new Error(`Request failed: ${path}`);
  return res.json();
}

// 5 functions that mirror backend routes
export const getKpis = () => getJson('kpis');
export const getFunnel = () => getJson('funnel');
export const getQuintiles = () => getJson('quintiles');
export const getTopMovies = () => getJson('top-movies');
export const getRecentFeedback = () => getJson('recent-feedback');