import { FunnelChart, Funnel, LabelList, Tooltip, ResponsiveContainer } from 'recharts';

// Get backend funnel counts data
export default function FunnelChartComponent({ data }) {
  if (!data) return null;
  return (
    // Send funnel data to recharts to make displays
    <ResponsiveContainer width="100%" height={320}>
      <FunnelChart>
        <Tooltip />
        <Funnel dataKey="value" data={data} isAnimationActive>
          <LabelList position="right" dataKey="name" fill="#333" stroke="none" />
        </Funnel>
      </FunnelChart>
    </ResponsiveContainer>
  );
}