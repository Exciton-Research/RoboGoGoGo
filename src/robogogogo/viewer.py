"""Python-only interactive 3D workbench; Matplotlib is a rendering dependency."""

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import spatial as sp
from .session import Session as Session


class Workbench:
    def __init__(self, session, *, export_path=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.session, self.running = session, False
        self.export_path = export_path
        self._syncing = False
        self._wall_start, self._sim_start = time.perf_counter(), 0.0
        self.fig = plt.figure(figsize=(15, 9), facecolor="#101722")
        self.fig.canvas.manager.set_window_title("robogogogo • 3D Robotics Workbench")
        self.fig.text(0.035, 0.95, "robogogogo", color="#e8f0f6", size=23, weight="bold")
        self.fig.text(0.19, 0.952, "ROBOTICS WORKBENCH  /  3D", color="#8fabbc", size=10)
        self.fig.text(
            0.035,
            0.917,
            "Native Python + JAX dynamics  •  SI units  •  Fixed-base research preview",
            color="#8fabbc",
            size=10,
        )
        self.ax = self.fig.add_axes([0.025, 0.21, 0.67, 0.68], projection="3d", facecolor="#101722")
        self.ax.set(
            xlim=(-0.95, 0.95),
            ylim=(-0.95, 0.95),
            zlim=(-0.1, 1.4),
            xlabel="X / m",
            ylabel="Y / m",
            zlabel="Z / m",
        )
        self.ax.set_box_aspect((1.9, 1.9, 1.5))
        self.ax.view_init(elev=24, azim=-55)
        self.ax.tick_params(colors="#8fabbc", labelsize=8)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.label.set_color("#8fabbc")
            axis.set_pane_color((0.07, 0.10, 0.15, 1.0))
            axis._axinfo["grid"]["color"] = (0.2, 0.3, 0.4, 0.25)
        xx, yy = np.meshgrid(np.linspace(-0.9, 0.9, 13), np.linspace(-0.9, 0.9, 13))
        self.ax.plot_wireframe(xx, yy, xx * 0, color="#354957", alpha=0.4, linewidth=0.5)
        self.links = [
            self.ax.plot([], [], [], color="#70e4d2", linewidth=6, solid_capstyle="round")[0]
            for _ in session.model.parents
        ]
        (self.joints,) = self.ax.plot([], [], [], "o", color="#e6f3f7", markersize=5)
        (self.tip,) = self.ax.plot([], [], [], "o", color="#70e4d2", markersize=8)
        (self.goal,) = self.ax.plot(
            [], [], [], "+", color="#ffb568", markersize=14, markeredgewidth=2
        )
        (self.trail,) = self.ax.plot([], [], [], color="#70e4d2", alpha=0.35, linewidth=1)
        self.trail_points = []
        self.sphere_artists = []
        self.fig.text(0.725, 0.875, "CONTROL", color="#e8f0f6", size=12, weight="bold")
        self.fig.text(
            0.725,
            0.845,
            "Joint goals" if session.scene == "arm" else "Ground contact experiment",
            color="#8fabbc",
            size=10,
        )
        self.sliders, self.target_sliders = [], []
        if session.scene == "arm":
            moving_names = [
                name
                for name, index in zip(session.model.names, session.model.indices)
                if index >= 0
            ]
            for i, (name, home) in enumerate(zip(moving_names, session.model.home)):
                axis = self.fig.add_axes([0.79, 0.79 - i * 0.047, 0.16, 0.017], facecolor="#273343")
                low = max(float(session.model.lower[i]), float(home) - np.pi)
                high = min(float(session.model.upper[i]), float(home) + np.pi)
                slider = Slider(
                    axis, name, low, high, valinit=float(home), color="#70e4d2", valfmt="%1.2f"
                )
                slider.label.set(color="#b9cbd7", size=8)
                slider.valtext.set(color="#e8f0f6", size=8)
                slider.on_changed(self._joint_change)
                self.sliders.append(slider)
            y = 0.79 - len(self.sliders) * 0.047 - 0.04
            self.fig.text(0.725, y + 0.035, "Cartesian target / m", color="#8fabbc", size=10)
            for i, name in enumerate("XYZ"):
                axis = self.fig.add_axes([0.79, y - i * 0.043, 0.16, 0.017], facecolor="#273343")
                slider = Slider(
                    axis,
                    name,
                    -0.9 if i < 2 else 0.05,
                    0.9 if i < 2 else 1.3,
                    valinit=float(session.target[i]),
                    color="#ffb568",
                    valfmt="%1.2f",
                )
                slider.label.set(color="#b9cbd7", size=9)
                slider.valtext.set(color="#e8f0f6", size=8)
                self.target_sliders.append(slider)
            self.ik_button = self._button(
                [0.745, y - 0.16, 0.2, 0.037], "Solve position IK", self._solve_ik
            )
        else:
            self.fig.text(
                0.725,
                0.76,
                "1 kg guided sphere\nRadius: 0.10 m\nPlane: z = 0\n\nCompliant spring-damper\nNo external physics engine",
                color="#b9cbd7",
                size=11,
                linespacing=1.8,
            )
        self.status = self.fig.text(
            0.725, 0.19, "Ready · press Run", color="#ffb568", size=9, wrap=True
        )
        self.telemetry = self.fig.text(
            0.045, 0.145, "", color="#c3d5df", size=10, family="monospace"
        )
        self.run_button = self._button([0.04, 0.055, 0.105, 0.05], "Run", self._toggle)
        self.step_button = self._button([0.16, 0.055, 0.105, 0.05], "Step", self._single_step)
        self.reset_button = self._button([0.28, 0.055, 0.105, 0.05], "Reset", self._reset)
        self.export_button = self._button(
            [0.745, 0.055, 0.2, 0.05], "Export trajectory", self._export
        )
        self.fig.text(0.41, 0.072, "Drag to orbit · scroll to zoom", color="#8fabbc", size=9)
        self.update_plot()

    def _button(self, rect, label, callback):
        from matplotlib.widgets import Button

        button = Button(self.fig.add_axes(rect), label, color="#243442", hovercolor="#365364")
        button.label.set_color("#e8f0f6")
        button.on_clicked(callback)
        return button

    def _joint_change(self, _):
        if self._syncing:
            return
        self.session.set_joints([slider.val for slider in self.sliders])
        self.status.set_text("Joint goal updated")
        self.update_plot()

    def _solve_ik(self, _):
        target = [slider.val for slider in self.target_sliders]
        error = self.session.set_target(target)
        self._syncing = True
        for slider, value in zip(self.sliders, self.session.target_q):
            slider.set_val(float(value))
        self._syncing = False
        self.status.set_text(
            f"IK residual: {error * 1000:.2f} mm"
            + (" · target not reached" if error > 0.02 else "")
        )
        self.update_plot()

    def _toggle(self, _):
        self.running = not self.running
        self.run_button.label.set_text("Pause" if self.running else "Run")
        self.status.set_text("Running" if self.running else "Paused")
        self._wall_start, self._sim_start = time.perf_counter(), self.session.elapsed

    def _single_step(self, _):
        if self.running:
            self._toggle(None)
        self._advance()
        self.update_plot()
        self.fig.canvas.draw_idle()

    def _advance(self):
        try:
            self.session.advance()
        except FloatingPointError as error:
            self.running = False
            self.run_button.label.set_text("Run")
            self.status.set_text(str(error))

    def _reset(self, _):
        self.running = False
        self.run_button.label.set_text("Run")
        self.session.reset()
        self.trail_points = []
        self._syncing = True
        for slider, value in zip(self.sliders, self.session.target_q):
            slider.set_val(float(value))
        for slider, value in zip(self.target_sliders, self.session.target):
            slider.set_val(float(value))
        self._syncing = False
        self._wall_start, self._sim_start = time.perf_counter(), 0.0
        self.status.set_text("Reset · recording cleared")
        self.update_plot()

    def _export(self, _):
        destination = (
            self.export_path or Path("runs") / f"robogogogo-{datetime.now():%Y%m%d-%H%M%S-%f}.npz"
        )
        try:
            path = self.session.export(destination)
            self.status.set_text(f"Saved {path.name}\n{len(self.session.history)} samples")
            print(f"Trajectory saved: {path}")
        except OSError as error:
            self.status.set_text(f"Export failed: {error.strerror}")

    def update_plot(self):
        pose = self.session.poses()
        points = np.asarray(pose.position)
        for i, (line, parent) in enumerate(zip(self.links, self.session.model.parents)):
            start = np.zeros(3) if parent < 0 else points[parent]
            line.set_data_3d(*np.stack((start, points[i])).T)
        self.joints.set_data_3d(*points.T)
        self.tip.set_data_3d(*points[-1:, :].T)
        self.goal.set_data_3d(*self.session.target.reshape(1, 3).T)
        if not self.trail_points or np.linalg.norm(points[-1] - self.trail_points[-1]) > 1e-5:
            self.trail_points.append(points[-1].copy())
        self.trail_points = self.trail_points[-500:]
        self.trail.set_data_3d(*np.array(self.trail_points).T)
        for artist in self.sphere_artists:
            artist.remove()
        self.sphere_artists = []
        if self.session.scene == "drop":
            positions = np.asarray(sp.sphere_positions(self.session.model, self.session.state.q))
            u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 12))
            for center, radius in zip(positions, self.session.model.sphere_radius):
                x = center[0] + float(radius) * np.cos(u) * np.sin(v)
                y = center[1] + float(radius) * np.sin(u) * np.sin(v)
                z = center[2] + float(radius) * np.cos(v)
                self.sphere_artists.append(
                    self.ax.plot_surface(x, y, z, color="#70e4d2", alpha=0.9)
                )
        elapsed = self.session.elapsed
        rtf = (elapsed - self._sim_start) / max(time.perf_counter() - self._wall_start, 1e-6)
        error = np.linalg.norm(points[-1] - self.session.target)
        energy = self.session.mechanical_energy()
        metric = f"TARGET ERROR  {error * 1000:7.2f} mm"
        if self.session.scene == "drop":
            contacts, _ = sp.plane_contacts(self.session.model, self.session.state)
            metric = f"PENETRATION  {float(contacts.penetration[0]) * 1000:7.2f} mm"
        self.telemetry.set_text(
            f"TIME  {elapsed:7.3f} s    DT  {float(self.session.model.dt) * 1000:.1f} ms    DOF  {self.session.state.q.size}    RTF  {rtf:.2f}x\n"
            f"ENERGY  {energy:+8.3f} J    {metric}    SAMPLES  {len(self.session.history)}"
        )

    def show(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        def frame(_):
            if self.running:
                self._advance()
                self.update_plot()
            return ()

        self.animation = FuncAnimation(self.fig, frame, interval=20, cache_frame_data=False)
        plt.show()
