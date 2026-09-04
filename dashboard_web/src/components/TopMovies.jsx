export default function TopMovies({ data }) {
  if (!data) return null;
  return (
    <table className="data-table">
      <thead>
        <tr><th>Title</th><th>Times recommended</th><th>Avg score</th></tr>
      </thead>
      <tbody>
        {data.map(row => (
          <tr key={row.title}>
            <td>{row.title}</td>
            <td>{row.times_recommended}</td>
            <td>{row.avg_score}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}