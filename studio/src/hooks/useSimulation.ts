import { useCallback, useEffect, useRef, useState } from "react";
import type { Connection, Snapshot } from "../types";

export function useSimulation() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [history, setHistory] = useState<Snapshot[]>([]);
  const [connection, setConnection] = useState<Connection>("connecting");
  const latest = useRef<Snapshot | null>(null);
  const accept = useCallback((next: Snapshot) => {
    if (latest.current && next.sequence <= latest.current.sequence) return;
    const previous = latest.current;
    latest.current = next;
    setSnapshot(next);
    setHistory((old) => {
      if (!previous || previous.epoch !== next.epoch) return [next];
      if (old.length && old[old.length - 1].state.time === next.state.time)
        return [...old.slice(0, -1), next];
      return [...old, next].slice(-600);
    });
  }, []);
  useEffect(() => {
    let disposed = false;
    let socket: WebSocket;
    let timer: ReturnType<typeof setTimeout>;
    const connect = () => {
      if (disposed) return;
      const socketHost =
        process.env.NODE_ENV === "development"
          ? `${location.hostname}:8765`
          : location.host;
      const socketUrl =
        process.env.NEXT_PUBLIC_ENGINE_WS ??
        `${location.protocol === "https:" ? "wss" : "ws"}://${socketHost}/api/ws`;
      socket = new WebSocket(socketUrl);
      socket.onopen = () => {
        if (!disposed) setConnection("connected");
      };
      let first = true;
      socket.onmessage = (event) => {
        if (disposed) return;
        try {
          const next = JSON.parse(event.data) as Snapshot;
          if (first) {
            latest.current = null;
            setHistory([]);
            first = false;
          }
          accept(next);
        } catch {
          setConnection("disconnected");
        }
      };
      socket.onclose = () => {
        if (disposed) return;
        setConnection("disconnected");
        timer = setTimeout(connect, 1500);
      };
      socket.onerror = () => socket.close();
    };
    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, [accept]);
  return { snapshot, history, connection, accept };
}
