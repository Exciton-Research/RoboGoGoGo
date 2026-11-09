export async function request<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    `/api${path}`,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const result = await response.json();
      message =
        typeof result.detail === "string"
          ? result.detail
          : JSON.stringify(result.detail);
    } catch {
      /* preserve HTTP error */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export { useSimulation } from "./hooks/useSimulation";
