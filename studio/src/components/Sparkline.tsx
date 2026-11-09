export function Sparkline({ data }: { data: number[] }) {
  const values = data.slice(-120);
  const max = Math.max(...values, 0.001),
    min = Math.min(...values, 0);
  const points = values
    .map(
      (v, i) =>
        `${(i / Math.max(values.length - 1, 1)) * 220},${33 - ((v - min) / Math.max(max - min, 0.001)) * 28}`,
    )
    .join(" ");
  return (
    <svg
      viewBox="0 0 220 38"
      role="img"
      aria-label="Recent target error history"
      preserveAspectRatio="none"
    >
      <line x1="0" y1="34" x2="220" y2="34" stroke="#dddeda" />
      <polyline
        points={points}
        fill="none"
        stroke="#6d897a"
        strokeWidth="1.8"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}
