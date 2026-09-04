export default function RecentFeedback({ data }) {
  if (!data) return null;
  return (
    <table className="data-table">
      <thead>
        <tr><th>User</th><th>Movie</th><th>Watched</th><th>Liked</th><th>Rating</th><th>When</th></tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>      
            <td>{row.username}</td>
            <td>{row.movie_title}</td>
            <td>{row.watched === true ? 'Yes' : row.watched === false ? 'No' : '—'}</td>
            <td>{row.liked === true ? 'Liked' : row.liked === false ? 'Disliked' : '—'}</td>
            <td>{row.rating ?? '—'}</td>
            <td>{new Date(row.updated_at).toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}