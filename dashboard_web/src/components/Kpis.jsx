export default function Kpis({ data }) {
  if (!data) return null;
  const cards = [
    { label: 'Users with recommendations', value: data.total_users },
    { label: 'Recommendations shown', value: data.total_shown },
    { label: 'Average rating given', value: data.avg_rating ?? '—' },
  ];
  return (
    <div className="kpi-row">
      {cards.map(c => (
        <div className="kpi-card" key={c.label}>
          <div className="kpi-value">{c.value}</div>
          <div className="kpi-label">{c.label}</div>
        </div>
      ))}
    </div>
  );
}