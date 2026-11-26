import { ArrowRight, X } from "lucide-react";
import { useWorkspace } from "../state/WorkspaceProvider";

export function HelpDialog() {
  const { help, setHelp } = useWorkspace();
  return (
    <>
      {" "}
      {help && (
        <div className="modal-backdrop" onClick={() => setHelp(false)}>
          <section
            className="help-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="icon-button modal-close"
              aria-label="Close help"
              onClick={() => setHelp(false)}
            >
              <X size={18} />
            </button>
            <div className="section-kicker">A WORKSPACE FOR ROBOT CONTROL</div>
            <h2 id="help-title">Observe. Act. Learn.</h2>
            <p>
              Build language-conditioned episodes from aligned RGB observations,
              robot state, and explicit actions. Python/JAX advances the robot;
              the browser renders synchronized head and wrist camera views.
            </p>
            <ol>
              <li>
                <strong>Define the task</strong>
                <span>
                  Choose a reaching fixture and enter the language instruction.
                </span>
              </li>
              <li>
                <strong>Observe and act</strong>
                <span>
                  Capture RGB, execute joint-delta actions, or collect a
                  rule-based reach demo.
                </span>
              </li>
              <li>
                <strong>Keep an aligned episode</strong>
                <span>
                  Export PNG frames, instructions, state, and action transitions
                  as a versioned ZIP dataset.
                </span>
              </li>
            </ol>
            <div className="shortcut-row">
              <span>
                <kbd>Space</kbd>Capture RGB
              </span>
              <span>
                <kbd>F</kbd>Frame robot
              </span>
              <span>
                <kbd>⌘ / Ctrl S</kbd>Save
              </span>
            </div>
            <p className="help-foot">
              This local server is one shared robotics workspace. Saved setups
              store state checkpoints; episode exports contain aligned visual
              data. The reference collector uses rule-based IK, not a pretrained
              VLA. The timeline keeps the latest 600 browser frames. Physics
              recording stops at 18,000 control ticks. RG Mobile Duo is the
              supported robot. Its mobile dynamics use a flat-ground model with
              self-collision and support guards; external obstacle planning and
              physical grasping are not integrated.
            </p>
            <a href="/docs" target="_blank" rel="noreferrer" className="button">
              Open API documentation
              <ArrowRight size={14} />
            </a>
          </section>
        </div>
      )}{" "}
    </>
  );
}
