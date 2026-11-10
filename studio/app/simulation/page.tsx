"use client";
import dynamic from "next/dynamic";
const Studio = dynamic(() => import("../../src/App"), {
  ssr: false,
  loading: () => (
    <main className="boot-screen">
      <strong>
        robogogogo<span>.</span>
      </strong>
      <p>Loading Robotics Studio…</p>
    </main>
  ),
});
export default function Page() {
  return <Studio />;
}
