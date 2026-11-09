export const formatTime = (seconds: number) =>
  seconds.toFixed(2).padStart(6, "0");
export const nice = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";
