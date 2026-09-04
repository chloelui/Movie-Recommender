import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Show how often each quintile got watched/liked
export default function QuintileChart({ data }) {
  if (!data || data.length === 0) {
    return <p>Oops, not enough data yet! Keep using the chatbot to generate recommendations and feedback.</p>;
  }
  // Render quintiles as grouped bar chart
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="score_quintile" label={{ value: 'Score quintile (1=lowest, 5=highest)', position: 'insideBottom', offset: -5 }} />
        <YAxis label={{ value: 'Rate (%)', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />
        <Bar dataKey="watch_rate" name="Watch rate %" fill="#4f8ef7" />
        <Bar dataKey="like_rate" name="Like rate %" fill="#f77f4f" />
      </BarChart>
    </ResponsiveContainer>
  );
}