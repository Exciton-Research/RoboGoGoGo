import { Check, CircleHelp, LoaderCircle } from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";

export function StatusBar() {
  const { snapshot, busy, error, notice } = useWorkspace();
  return (
    <>
      {" "}
      <footer
        className={`statusbar ${error || snapshot?.state.error ? "has-error" : ""}`}
      >
        <div>
          <span className={busy ? "spin" : ""}>
            {busy ? (
              <LoaderCircle size={13} />
            ) : error || snapshot?.state.error ? (
              <CircleHelp size={13} />
            ) : (
              <Check size={13} />
            )}
          </span>
          <p role="status">
            {error ||
              snapshot?.state.error ||
              notice ||
              snapshot?.state.message ||
              "Waiting for the local Python backend…"}
          </p>
        </div>
        <span className="status-version">
          robogogogo Studio<span>·</span>research preview
        </span>
      </footer>{" "}
    </>
  );
}
