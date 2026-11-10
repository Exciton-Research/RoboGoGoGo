"""Coordinated task planning/execution over the existing aligned episode pipeline."""

from dataclasses import asdict, replace

import numpy as np

from robogogogo.runtime.supervisor import ControlSystem

from ..contracts import Observation
from ..planning import CollisionWorld, SphereObstacle, adjacent_exclusions, capsules_from_visuals
from ..planning.task import TaskExecutor, plan_shared_inspection
from .schemas import PolicyAction


class TaskService:
    def __init__(self, workspace):
        self.workspace = workspace
        self.world = None
        self.control = None
        self.manipulation = None
        self.handover_scene = None
        self.started_operations = set()

    def summary(self):
        task = self.workspace.task
        if task is None:
            return None
        return dict(
            **task.summary(),
            margin_m=self.world.margin,
            obstacles=[
                dict(
                    **asdict(o),
                    kind="box"
                    if hasattr(o, "size")
                    else "capsule"
                    if hasattr(o, "start")
                    else "sphere",
                )
                for o in self.world.obstacles
                if o.name != "transfer_bar"
            ],
            geometry="Conservative primitive capsules; attached links within two tree edges excluded",
            collision_checks=self.world.checks,
            support_contact_tolerance_m=0.003 if self.manipulation else None,
        )

    def _publish(self):
        w = self.workspace
        if w.task and w.episode:
            w.episode.task_progress = self.summary()
        w._refresh()
        return w.cached

    def plan(self, request):
        w = self.workspace
        with w.lock:
            if request.kind in ("handover", "material_grasp"):
                return self.plan_handover(request)
            if w.running:
                raise ValueError("Pause dynamics before planning a task")
            if len(w.session.arms) != 2:
                raise ValueError("Load the dual-arm rig before planning shared inspection")
            if np.max(np.abs(np.asarray(w.session.state.qd))) > 0.06:
                raise ValueError("Let the robot settle before planning")
            s = w.session
            station = np.asarray(request.station)
            obstacle = SphereObstacle("inspection_fixture", tuple(station - [0, 0, 0.30]), 0.10)
            world = CollisionWorld(
                s.model,
                capsules_from_visuals(s.model, w.model_info["visuals"]),
                excluded_pairs=adjacent_exclusions(s.model),
                obstacles=[obstacle],
                margin=0.015,
            )
            # Build the entire plan before replacing an existing episode.
            plan = plan_shared_inspection(
                s.model, s.arms, world, np.asarray(s.state.q), station=station, seed=request.seed
            )
            cameras = w.episode.cameras if w.episode else None
            episode = w._episode_for_state(
                "Both arms inspect the shared station, one at a time, then return.", cameras
            )
            executor = TaskExecutor(
                plan,
                max_delta=np.minimum(episode.action_spec["max_delta"], 0.08),
                started_at=s.elapsed,
            )
            control = ControlSystem(
                {"coordinated_task": executor},
                lower=s.model.lower,
                upper=s.model.upper,
                max_delta=executor.limit,
                collision_world=world,
            )
            self.world, self.control = world, control
            self.manipulation = self.handover_scene = None
            s.payload_model = None
            s.contact_system = None
            w.task, w.episode = executor, episode
            episode.task_plan = plan.to_dict()
            episode.task_plan["collision_model"] = dict(
                capsules=[asdict(c) for c in world.capsules],
                obstacles=[asdict(o) for o in world.obstacles],
                margin_m=world.margin,
                excluded_body_pairs=[list(pair) for pair in adjacent_exclusions(s.model)],
            )
            s.targets = {k: np.asarray(v) for k, v in executor.stage.targets.items()}
            episode.targets = {k: list(v) for k, v in executor.stage.targets.items()}
            episode.target = np.asarray(s.target, dtype=float).tolist()
            s.restart_recording()
            w.epoch += 1
            w.error = None
            w.message = "Coordinated inspection planned. Capture and run the task."
            executor.minimum_clearance = world.clearance(s.state.q).distance
            return self._publish()

    def step(self, observation_index):
        w = self.workspace
        with w.lock:
            task = w.task
            if task is None or task.status not in ("ready", "running"):
                raise ValueError("Plan a task, or resume a paused task, before stepping")
            episode = w.episode
            s = w.session
            if (
                w.running
                or not episode.observations
                or observation_index != len(episode.observations) - 1
            ):
                raise ValueError("Task step requires the latest paused camera observation")
            latest = episode.observations[-1]
            if latest["time"] != s.elapsed or not np.array_equal(
                latest["joint_position"], np.asarray(s.state.q)
            ):
                raise ValueError("Capture the current state before stepping the task")
            if (
                self.manipulation
                and self.manipulation.recovery
                and self.manipulation.recovery.planner.vision_required
            ):
                from .perception import update_recovery_observation

                update_recovery_observation(w, observation_index)
            event_start = len(self.manipulation.events) if self.manipulation else 0
            recovery_start = (
                len(self.manipulation.recovery.records)
                if self.manipulation and self.manipulation.recovery
                else 0
            )
            if self.manipulation and not (
                self.manipulation.recovery and self.manipulation.recovery.active
            ):
                self.start_handover_operation()
            observation = Observation(
                episode.instruction,
                b"",
                tuple(np.asarray(s.state.q)),
                tuple(np.asarray(s.state.qd)),
                s.elapsed,
            )
            stage_index = task.stage_index
            try:
                if self.manipulation and self.manipulation.motion_hold:
                    delta = np.asarray(self.manipulation.hold_target) - np.asarray(s.state.q)
                    control_ticks = task.control_ticks
                else:
                    decision = self.control.step(observation, controller="coordinated_task")
                    delta = list(decision.action.actions[0])
                    control_ticks = decision.action.control_ticks
            except ValueError as error:
                task.block(str(error))
                episode.done = True
                episode.reason = "collision_guard"
                if episode.actions:
                    episode.actions[-1]["truncated"] = True
                w.error = str(error)
                return self._publish()
            previous_actions = len(episode.actions)
            w.episode_service.policy_action(
                PolicyAction(
                    delta=list(delta),
                    observation=observation_index,
                    control_ticks=control_ticks,
                    source="task_planner",
                ),
                collision_world=self.world,
                task_step=True,
            )
            recovery = self.manipulation.recovery if self.manipulation else None
            current_world = (
                recovery.guarded_world(self.world, self.manipulation) if recovery else self.world
            )
            clearance = current_world.clearance(
                s.state.q, self.manipulation.object_segment() if self.manipulation else None
            )
            task.minimum_clearance = min(task.minimum_clearance, clearance.distance)
            if len(episode.actions) > previous_actions:
                episode.actions[-1]["coordination"] = dict(
                    stage_index=stage_index,
                    owner=task.stage.owner,
                    clearance_m=clearance.distance,
                    closest_pair=list(clearance.pair),
                )
            if w.error or episode.done:
                task.block(w.error or episode.reason)
                if w.error and "collision" in w.error.lower():
                    episode.reason = "collision_guard"
            elif recovery and (recovery.placed or recovery.secured):
                task.block(
                    "Recovery verified; inspect the object and replan from the new contact configuration"
                )
                episode.done, episode.success, episode.reason = (
                    True,
                    False,
                    ("recovered_put_down" if recovery.placed else "recovered_regrasp"),
                )
                episode.actions[-1].update(terminated=True, truncated=False, success=False)
            elif recovery and recovery.active:
                pass  # Recovery owns motion; original task waypoints cannot advance.
            else:
                task.update(
                    np.asarray(s.state.q),
                    np.asarray(s.state.qd),
                    s.elapsed,
                    w.cached["state"]["tool_positions"],
                    tool_rotations={a.name: w.cached["state"]["rotations"][a.body] for a in s.arms},
                    completion_ready=self.manipulation is None
                    or (self.manipulation.pending is None and not self.manipulation.motion_hold),
                )
                if self.manipulation and task.stage_index != stage_index:
                    try:
                        self.complete_handover_stage(stage_index)
                    except ValueError as error:
                        task.block(str(error))
                        w.error = str(error)
                        episode.done = True
                        episode.reason = "grasp_failure"
                        episode.actions[-1].update(terminated=True, success=False)
                if task.status == "completed":
                    episode.done = episode.success = True
                    episode.reason = "task_completed"
                    episode.actions[-1].update(
                        success=True, terminated=True, truncated=False, reward=1.0
                    )
                elif task.status == "blocked":
                    episode.done = True
                    episode.reason = "grasp_failure" if w.error else "task_timeout"
                    episode.actions[-1].update(truncated=not bool(w.error))
                elif task.stage_index != stage_index:
                    s.targets = {k: np.asarray(v) for k, v in task.stage.targets.items()}
                    episode.targets = {k: list(v) for k, v in task.stage.targets.items()}
                    episode.target = np.asarray(s.target, dtype=float).tolist()
            if self.manipulation and len(episode.actions) > previous_actions:
                episode.actions[-1]["grasp_events"] = self.manipulation.events[event_start:]
                if self.manipulation.recovery:
                    episode.actions[-1]["recovery_targets"] = self.manipulation.recovery.records[
                        recovery_start:
                    ]
            w.message = f"{task.plan.name}: {task.stage.name if task.status != 'completed' else 'completed'}"
            return self._publish()

    def pause(self):
        w = self.workspace
        with w.lock:
            if w.task:
                w.task.pause()
                w.message = "Coordinated task paused between actions"
            return self._publish()

    def resume(self):
        w = self.workspace
        with w.lock:
            if not w.task or w.task.status != "paused":
                raise ValueError("Only paused tasks can resume; blocked tasks must be replanned")
            w.task.resume()
            w.message = "Coordinated task ready to resume"
            return self._publish()

    def plan_handover(self, request):
        from ..manipulation import Manipulation
        from ..planning.handover import HandoverScene

        w = self.workspace
        s = w.session
        if w.running or np.max(np.abs(np.asarray(s.state.qd))) > 0.06:
            raise ValueError("Pause and settle the robot before planning handover")
        scene = HandoverScene(s.model, s.arms, w.model_info["visuals"])
        plan, worlds = scene.plan(s.state.q, seed=request.seed)
        if request.kind == "material_grasp":
            plan = replace(
                plan,
                name="Grasp and lift material specimen",
                kind="material_grasp",
                stages=plan.stages[:4] + (replace(plan.stages[4], operation="verify_grasp_lift"),),
            )
            worlds = worlds[:5]
        supports = [
            [t.center[0], t.center[1], t.size[0] / 2, t.size[1] / 2, t.center[2] + t.size[2] / 2]
            for t in scene.tables
        ]
        manipulation = Manipulation(
            s.model,
            s.arms,
            scene.pick,
            supports,
            material=request.object_material,
            structural=request.structural_dynamics,
        )
        for name in manipulation.names:
            manipulation.configure_gripper(
                name,
                force_limit_n=request.grip_force_n,
                mu_static=request.friction_static,
                mu_dynamic=request.friction_dynamic,
            )
        if request.adaptive_grasp:
            manipulation.enable_adaptive_grasp(
                initial_force_n=request.initial_grip_force_n,
                damage_force_n=request.object_damage_force_n,
            )
        if request.recovery_behaviors:
            manipulation.enable_recovery(
                camera_based=request.camera_recovery,
                active_cameras=(w.episode.cameras if w.episode else [])
                if request.active_perception
                else None,
            )
        cameras = w.episode.cameras if w.episode else None
        episode = w._episode_for_state(
            "Grasp and lift the specimen without damaging it."
            if request.kind == "material_grasp"
            else "Pick up the bar, pass it to the other arm, and place it on the destination support.",
            cameras,
        )
        delicate = manipulation.material is not None
        episode.horizon = 850 if delicate else 450
        episode.max_observations = episode.horizon + 1
        episode.default_control_ticks = 20
        executor = TaskExecutor(
            plan,
            max_delta=np.minimum(episode.action_spec["max_delta"], 0.035)
            if delicate
            else episode.action_spec["max_delta"],
            started_at=s.elapsed,
            timeout=170.0 if delicate else 90.0,
            joint_tolerance=0.004,
            velocity_tolerance=0.025,
            control_ticks=20,
        )
        world = worlds[0]
        self.handover_scene = scene
        self.manipulation = manipulation
        self.started_operations = set()
        self.world = world
        self.control = ControlSystem(
            {"coordinated_task": executor},
            lower=s.model.lower,
            upper=s.model.upper,
            max_delta=executor.limit,
            collision_world=world,
        )
        s.payload_model = None
        s.contact_system = manipulation
        w.task, w.episode = executor, episode
        episode.task_plan = plan.to_dict()
        episode.task_plan["grasp_model"] = manipulation.summary()["grasp_model"]
        episode.task_plan["structural_dynamics"] = request.structural_dynamics
        episode.task_plan["recovery_behaviors"] = dict(
            enabled=request.recovery_behaviors,
            camera_proposals=request.camera_recovery,
            active_perception=request.active_perception,
            regrasp="collision-checked helper approach and force-verified regrasp",
            put_down="checked nearby support transport and load-verified descent",
            support_weight_fraction=0.8,
            max_attempts=2,
        )
        episode.task_plan["object_material"] = manipulation.summary()["object"]["material"]
        episode.task_plan["handling_max_joint_delta"] = executor.limit.tolist()
        episode.task_plan["material_detection"] = dict(
            warning_risk=0.55,
            damage_onset_risk=1.0 if request.structural_dynamics else 0.7,
            fracture_risk=1.0,
            source="simulated material state",
            stop_on_damage=True,
        )
        episode.task_plan["adaptive_grasp"] = dict(
            enabled=request.adaptive_grasp,
            initial_force_n=request.initial_grip_force_n,
            damage_force_n=request.object_damage_force_n,
            command_reserve=0.75,
            fast_unload_fraction=0.9,
            sensor="simulated contact patch loads and relative speed",
            feedback_period_s=float(s.model.dt) * s.substeps,
        )
        episode.task_plan["contact_parameters"] = dict(
            contact_dt=manipulation.contact_dt,
            mu_static=request.friction_static,
            mu_dynamic=request.friction_dynamic,
            jaw_force_limit_n=request.grip_force_n,
            jaw_effective_mass_kg=0.05,
            normal_stiffness_per_sample=4000.0,
            normal_damping_per_sample=3.0,
            tangent_stiffness_per_sample=10000.0,
            tangent_damping_per_sample=2.0,
            samples_per_jaw=6,
            support_normal_stiffness_per_sample=12000.0,
            support_normal_damping_per_sample=60.0,
            support_rolling_resistance=0.01,
            support_mu_static=0.7,
            support_mu_dynamic=0.5,
            motor_target_speed_m_s=0.15,
            motor_stiffness=2000.0,
            motor_damping=8.0,
            controller_contact_load_feedforward=True,
        )
        if manipulation.structural:
            structural_model = manipulation.structural
            episode.task_plan["contact_parameters"] = dict(
                model="structural_surface_quadrature_v1",
                contact_dt=manipulation.contact_dt,
                normal_stiffness_per_area=4e7,
                normal_damping_per_area=1.5e4,
                surface_samples_per_cell=54,
                cells=structural_model.cells,
                pad_half_extent_m=structural_model.pad_half_extent,
                pad_half_height_m=structural_model.pad_half_height,
                jaw_force_limit_n=request.grip_force_n,
                jaw_effective_mass_kg=0.05,
                motor_stiffness=20000.0,
                motor_damping=40.0,
                guarded_closing_speed_m_s=0.02,
                opening_speed_m_s=0.15,
                mu_static=request.friction_static,
                mu_dynamic=request.friction_dynamic,
                supports=supports,
            )
            episode.task_plan["structural_model"] = asdict(structural_model.material)
            episode.task_plan["material_detection"].update(
                source="simulated cohesive separation forecast and contact pressure",
                forecast_s=0.005,
                warning_risk=0.55,
                damage_onset_risk=1.0,
            )
            episode.task_plan["load_transfer"] = dict(
                donor_ramp_s=1.0, receiver_dwell_s=0.25, donor_release_load_n=0.2
            )
        episode.task_plan["payload"] = dict(
            mass=manipulation.object.mass,
            radius=manipulation.object.radius,
            half_length=manipulation.object.half_length,
        )
        episode.task_plan["collision_model"] = dict(
            margin_m=0.015,
            support_contact_tolerance_m=0.003,
            excluded_body_pairs=[list(pair) for pair in adjacent_exclusions(s.model)],
            phases=[asdict(p) for p in scene.phases],
            capsules=[asdict(c) for c in scene.capsules],
            supports=[asdict(t) for t in scene.tables],
        )
        s.targets = {k: np.asarray(v) for k, v in executor.stage.targets.items()}
        episode.targets = {k: list(v) for k, v in executor.stage.targets.items()}
        episode.target = np.asarray(s.target, dtype=float).tolist()
        s.restart_recording()
        w.epoch += 1
        w.error = None
        w.message = "Pick-and-handover task planned with payload-aware collision checking."
        executor.minimum_clearance = world.clearance(s.state.q).distance
        return self._publish()

    def after_tick(self):
        if self.manipulation and self.manipulation.failure:
            raise FloatingPointError(self.manipulation.failure)
        return bool(
            self.manipulation
            and self.manipulation.recovery
            and (
                self.manipulation.recovery.placed
                or self.manipulation.recovery.secured
                or self.manipulation.recovery.phase == "view_capture"
            )
        )

    def start_handover_operation(self):
        w = self.workspace
        index = w.task.stage_index
        if index in self.started_operations:
            return
        operation = w.task.stage.operation
        m, scene, s = self.manipulation, self.handover_scene, w.session
        if operation == "grasp":
            m.grasp(scene.giver.name, s)
        elif operation == "transfer":
            m.transfer(scene.receiver.name, s)
        elif operation == "release":
            if np.linalg.norm(np.asarray(m.state.position) - scene.place) > 0.015:
                raise ValueError("Object is not aligned over the destination support")
            m.release(scene.receiver.name, s)
        self.started_operations.add(index)

    def complete_handover_stage(self, index):
        w = self.workspace
        s = w.session
        task = w.task
        scene = self.handover_scene
        operation = task.plan.stages[index].operation
        before = len(self.manipulation.events)
        if operation == "verify_grasp_lift":
            m = self.manipulation
            if (
                m.owner != scene.giver.name
                or not m.grippers[scene.giver.name]["holding"]
                or float(m.state.position[2]) < scene.pick[2] + 0.06
                or np.linalg.norm(np.asarray(m.state.velocity)) > 0.02
                or m.summary()["object"]["material"]["damaged"]
            ):
                raise ValueError("Material grasp/lift verification failed")
            m.events.append(dict(event="material_lift_verified", time=s.elapsed))
        if operation == "verify_placement":
            position = np.asarray(self.manipulation.state.position)
            if (
                self.manipulation.owner is not None
                or np.linalg.norm(position - scene.place) > 0.015
                or np.linalg.norm(np.asarray(self.manipulation.state.velocity)) > 0.02
            ):
                raise ValueError("Placement verification failed")
            self.manipulation.events.append(dict(event="placement_verified", time=s.elapsed))
        if len(self.manipulation.events) > before:
            w.episode.actions[-1]["grasp_events"] = self.manipulation.events[before:]
        if task.status != "completed":
            self.world = scene.world(scene.phases[task.stage_index], self.manipulation)
            self.control.collision_world = self.world
            clearance = self.world.clearance(
                s.state.q, self.manipulation.object_segment() if self.manipulation else None
            )
            if clearance.distance < self.world.margin:
                raise ValueError(f"Grasp phase transition violates clearance: {clearance.pair}")
