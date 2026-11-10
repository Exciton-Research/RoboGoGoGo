"""Internal numerical regression fixture; not a supported robot workflow."""



try:
    from .structural_handover import make_rig
except ImportError:
    from structural_handover import make_rig


def run(scenario="regrasp"):
    if scenario not in ("regrasp", "put_down", "warning"):
        raise ValueError("Unknown recovery scenario")
    session, grasp = make_rig(
        "soft_foam" if scenario == "warning" else "soft_elastomer",
        40.0 if scenario == "warning" else 16.0,
    )
    helper = enabled = False
    height = 0.0
    if scenario == "warning":
        grasp.enable_recovery()
        enabled = True
    minimum_helper_load = float("inf")
    for index in range(1800):
        if scenario == "regrasp":
            if grasp.owner and not helper:
                grasp._command("right", "closed", session)
                helper = True
            if index > 350 and grasp.stable["right"] > 0.3 and not enabled:
                grasp.enable_recovery()
                grasp.configure_gripper("left", mu_static=0.05, mu_dynamic=0.03)
                enabled = True
            if enabled and not grasp.recovery.active and grasp.recovery.attempts == 0:
                # Measured tangential sliding: only the low-friction tool moves.
                session.target_q = session.target_q.at[2].add(0.0004)
        elif scenario == "put_down":
            if grasp.owner and not enabled and not grasp.motion_hold:
                height = min(0.04, height + 0.0003)
                session.target_q = session.target_q.at[2].set(height)
            if height >= 0.04 and index > 450 and not enabled:
                grasp.enable_recovery()
                enabled = True
                nodes = grasp.contact.structure
                grasp.contact = grasp.contact._replace(
                    structure=nodes._replace(damage=nodes.damage.at[0].set(0.02))
                )
        session.advance()
        if grasp.recovery and grasp.recovery.helper and grasp.recovery.active:
            minimum_helper_load = min(
                minimum_helper_load, min(grasp.grippers[grasp.recovery.helper]["normal_force_n"])
            )
        if grasp.failure or (
            grasp.recovery and grasp.recovery.phase in ("regrasped", "placed", "blocked")
        ):
            break
    return dict(
        time_s=session.elapsed,
        seeded_bond_damage=0.02 if scenario == "put_down" else 0.0,
        minimum_helper_load_n=minimum_helper_load if minimum_helper_load != float("inf") else None,
        **grasp.summary(),
    )
