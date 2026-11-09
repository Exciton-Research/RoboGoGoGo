"""Native JAX XPBD elastic cells and irreversible cohesive interfaces.

Predefined hexahedral interfaces, surface-quadrature contacts, fixed topology arrays.
This is a calibrated-specimen model, not arbitrary mesh fracture or continuum FEM.
"""

from dataclasses import asdict, dataclass
from itertools import product
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class StructuralMaterial:
    name: str
    modulus: float
    mass: float
    tensile_strength: float
    compressive_strength: float
    fracture_energy: float
    cohesive_length: float = 0.001
    friction: float = 0.8
    color: str = "#77b8ad"

    def __post_init__(self):
        values = (
            self.modulus,
            self.mass,
            self.tensile_strength,
            self.compressive_strength,
            self.fracture_energy,
            self.cohesive_length,
            self.friction,
        )
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Structural parameters must be finite and positive")
        if (
            2 * self.fracture_energy / self.tensile_strength
            <= self.tensile_strength * self.cohesive_length / self.modulus
        ):
            raise ValueError("Fracture energy must exceed elastic initiation energy")


STRUCTURAL_MATERIALS = {
    "soft_foam": StructuralMaterial("soft_foam", 180_000, 0.04, 8_000, 110_000, 2),
    "soft_elastomer": StructuralMaterial(
        "soft_elastomer", 1_200_000, 0.16, 60_000, 1_100_000, 40, color="#9d98d1"
    ),
    "fragile_shell": StructuralMaterial(
        "fragile_shell", 15_000_000, 0.06, 90_000, 180_000, 5, color="#e6cf9d"
    ),
}


class StructureState(NamedTuple):
    position: jax.Array  # (cells * 8, 3), world coordinates
    velocity: jax.Array
    damage: jax.Array  # four independently failing bonds per interface
    maximum_opening: jax.Array
    crush_work: jax.Array  # irreversible overpressure work per cell, J
    peak_pressure: jax.Array
    contact_memory: jax.Array
    jaw_width: jax.Array
    jaw_velocity: jax.Array
    drive_width: jax.Array


class StructureReport(NamedTuple):
    normal: jax.Array
    friction: jax.Array
    wrench_force: jax.Array
    wrench_torque: jax.Array
    pressure: jax.Array
    cohesive_traction: jax.Array
    elastic_energy: jax.Array
    self_contacts: jax.Array
    slip_speed: jax.Array
    sliding: jax.Array
    penetration: jax.Array
    risk: jax.Array
    support_load: jax.Array


def _edge_colors(size=8):
    # Round-robin complete-graph coloring: each group has disjoint endpoints.
    order = list(range(size))
    groups = []
    for _ in range(size - 1):
        groups.append([(order[i], order[-i - 1]) for i in range(size // 2)])
        order = [order[0], order[-1], *order[1:-1]]
    return np.asarray(groups)


class CohesiveBar:
    """Eight-node cells, energy-calibrated struts, and four cohesive bonds per face."""

    def __init__(
        self,
        material="soft_foam",
        *,
        cells=8,
        length=0.30,
        width=0.04,
        grippers=1,
        iterations=8,
        dt=0.0001,
        pad_half_extent=0.035,
        pad_half_height=None,
    ):
        if isinstance(material, str):
            material = STRUCTURAL_MATERIALS[material]
        if not isinstance(material, StructuralMaterial):
            raise ValueError("Expected structural material")
        if not 3 <= cells <= 24 or not 4 <= iterations <= 40 or not 1 <= grippers <= 4:
            raise ValueError("Invalid discretization or gripper count")
        if (
            not np.isfinite([length, width, dt]).all()
            or not 0 < dt <= 0.0005
            or min(length, width) <= 0
        ):
            raise ValueError("Invalid dimensions or contact step")
        self.pad_half_extent = float(pad_half_extent)
        self.pad_half_height = (
            self.pad_half_extent if pad_half_height is None else float(pad_half_height)
        )
        if not 0.005 <= self.pad_half_height <= 0.05:
            raise ValueError("Invalid pad height")
        if not 0.005 <= self.pad_half_extent <= 0.05:
            raise ValueError("Invalid pad extent")
        self.material, self.cells, self.length, self.width = material, cells, length, width
        self.grippers, self.iterations, self.dt = grippers, iterations, dt
        self.cell_length = length / cells
        self.node_mass = material.mass / (8 * cells)
        corners = np.array(list(product((-1.0, 1.0), repeat=3)))
        local = corners * np.array([width, self.cell_length, width]) / 2
        centers = np.array(
            [[0, -length / 2 + (i + 0.5) * self.cell_length, 0] for i in range(cells)]
        )
        self.rest = jnp.asarray((local[None] + centers[:, None]).reshape(-1, 3))
        groups = _edge_colors()
        edges = np.stack([np.concatenate([g + 8 * c for c in range(cells)]) for g in groups])
        self.edges = jnp.asarray(edges)
        lengths = np.linalg.norm(
            np.asarray(self.rest)[edges[:, :, 1]] - np.asarray(self.rest)[edges[:, :, 0]], axis=2
        )
        self.rest_lengths = jnp.asarray(lengths)
        delta = local[groups[:, :, 1]] - local[groups[:, :, 0]]
        # Matches axial affine strain energy E*volume*epsilon^2/2 for one cell.
        self.strut_k = (
            material.modulus
            * width**2
            * self.cell_length
            / np.sum(delta[:, :, 1] ** 4 / np.sum(delta**2, axis=2))
        )
        bonds = []
        for c in range(cells - 1):
            for i, corner in enumerate(corners):
                if corner[1] == 1:
                    j = np.flatnonzero(np.all(corners == corner * [1, -1, 1], axis=1))[0]
                    bonds.append((8 * c + i, 8 * (c + 1) + j))
        self.bonds = jnp.asarray(bonds)
        self.bond_area = width**2 / 4
        self.bond_k = material.modulus * self.bond_area / material.cohesive_length
        # Trilinear samples, Simpson area weights, six exterior faces per cell.
        weights, areas, faces = [], [], []
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for side in (-1.0, 1.0):
                for u, v in product(range(3), repeat=2):
                    point = np.zeros(3)
                    point[axis] = side
                    point[others] = [u - 1, v - 1]
                    weights.append(np.prod((1 + corners * point) / 2, axis=1))
                    dims = [width, self.cell_length, width]
                    areas.append(
                        dims[others[0]] * dims[others[1]] * [1, 4, 1][u] * [1, 4, 1][v] / 36
                    )
                    faces.append((axis, side))
        self.weights = jnp.asarray(weights)
        self.areas = jnp.asarray(areas)
        self.faces = jnp.asarray(faces)
        # Surface triangles keep each cell closed when a neighboring interface breaks.
        triangles = []
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for side in (-1, 1):
                ids = []
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    point = np.zeros(3)
                    point[axis] = side
                    point[others] = [a, b]
                    ids.append(int(np.flatnonzero(np.all(corners == point, axis=1))[0]))
                tri = np.array([[ids[0], ids[1], ids[2]], [ids[0], ids[2], ids[3]]])
                if (
                    np.cross(
                        local[tri[0, 1]] - local[tri[0, 0]], local[tri[0, 2]] - local[tri[0, 0]]
                    )[axis]
                    * side
                    < 0
                ):
                    tri = tri[:, [0, 2, 1]]
                triangles.extend(tri.tolist())
        self.triangles = np.asarray(triangles, dtype=int)
        # Nonadjacent nodal self-contact, including separated fragment nodes. Skin is
        # small and coincident interface nodes remain excluded to avoid a fracture kick.
        pi, pj = np.triu_indices(8 * cells, 1)
        rest_distance = np.linalg.norm(
            np.asarray(self.rest)[pi] - np.asarray(self.rest)[pj], axis=1
        )
        mask = (pi // 8 != pj // 8) & (rest_distance > 0.004)
        self.self_i, self.self_j = jnp.asarray(pi[mask]), jnp.asarray(pj[mask])
        self.step = jax.jit(self._step)
        self.elastic_energy = jax.jit(self._elastic_energy)

    def initial(self, center=(0, 0, 0.022)):
        n = self.cells * 8
        return StructureState(
            self.rest + jnp.asarray(center),
            jnp.zeros((n, 3)),
            jnp.zeros(len(self.bonds)),
            jnp.zeros(len(self.bonds)),
            jnp.zeros(self.cells),
            jnp.zeros(self.cells),
            jnp.zeros((self.grippers, 2, self.cells, 54, 3)),
            jnp.full(self.grippers, 0.075),
            jnp.zeros(self.grippers),
            jnp.full(self.grippers, 0.075),
        )

    def _elastic_project(self, position, damage, inverse_mass):
        alpha = 1 / (self.strut_k * self.dt**2)
        bond_alpha = 1 / (self.bond_k * jnp.maximum(1 - damage, 1e-10) * self.dt**2)

        def sweep(_, carry):
            x, multipliers, bond_lambda = carry

            def group(g, values):
                x, multipliers = values
                i, j = self.edges[g, :, 0], self.edges[g, :, 1]
                d = x[j] - x[i]
                length = jnp.linalg.norm(d, axis=1)
                normal = d / jnp.maximum(length[:, None], 1e-12)
                change = (-(length - self.rest_lengths[g]) - alpha * multipliers[g]) / (
                    inverse_mass[i] + inverse_mass[j] + alpha
                )
                x = x.at[i].add(-inverse_mass[i, None] * change[:, None] * normal)
                x = x.at[j].add(inverse_mass[j, None] * change[:, None] * normal)
                return x, multipliers.at[g].add(change)

            x, multipliers = jax.lax.fori_loop(0, 7, group, (x, multipliers))
            i, j = self.bonds[:, 0], self.bonds[:, 1]
            delta = (-(x[j] - x[i]) - bond_alpha[:, None] * bond_lambda) / (
                inverse_mass[i] + inverse_mass[j] + bond_alpha
            )[:, None]
            delta = jnp.where((damage < 1)[:, None], delta, 0)
            x = x.at[i].add(-inverse_mass[i, None] * delta)
            x = x.at[j].add(inverse_mass[j, None] * delta)
            return x, multipliers, bond_lambda + delta

        return jax.lax.fori_loop(
            0,
            self.iterations,
            sweep,
            (position, jnp.zeros_like(self.rest_lengths), jnp.zeros((len(self.bonds), 3))),
        )

    def _step(
        self,
        state,
        centers,
        rotations,
        tool_velocity,
        tool_omega,
        target_width,
        force_limit,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        external_force=None,
        pinned=None,
        supports=None,
        protect=False,
        force_ceiling=None,
        mu_static=None,
        mu_dynamic=None,
    ):
        dt = self.dt
        mat = self.material
        if force_ceiling is None:
            force_ceiling = jnp.full(self.grippers, 1e12)
        if mu_static is None:
            mu_static = jnp.full(self.grippers, mat.friction)
        if mu_dynamic is None:
            mu_dynamic = mu_static
        if external_force is None:
            external_force = jnp.zeros_like(state.position)
        if pinned is None:
            pinned = jnp.zeros(len(state.position), dtype=bool)
        inverse_mass = jnp.where(pinned, 0.0, 1 / self.node_mass)
        cell = state.position.reshape(self.cells, 8, 3)
        cell_v = state.velocity.reshape(self.cells, 8, 3)
        samples = jnp.einsum("sn,cnk->csk", self.weights, cell)
        velocity = jnp.einsum("sn,cnk->csk", self.weights, cell_v)
        # Current cell frame from mean opposite faces; normals follow bending.
        corners = jnp.asarray(list(product((-1.0, 1.0), repeat=3)))
        axes = jnp.einsum("na,cnk->cak", corners, cell) / 4
        normals = jnp.stack(
            (
                jnp.cross(axes[:, 1], axes[:, 2]),
                jnp.cross(axes[:, 2], axes[:, 0]),
                jnp.cross(axes[:, 0], axes[:, 1]),
            ),
            axis=1,
        )
        normals = normals / jnp.maximum(jnp.linalg.norm(normals, axis=2, keepdims=True), 1e-12)
        face_normals = normals[:, self.faces[:, 0].astype(jnp.int32)] * self.faces[None, :, 1, None]
        broken = jnp.all(state.damage.reshape(-1, 4) >= 1, axis=1)
        exterior = jnp.ones((self.cells, 54), dtype=bool)
        exterior = exterior.at[1:, 18:27].set(broken[:, None])
        exterior = exterior.at[:-1, 27:36].set(broken[:, None])
        side = jnp.array([-1.0, 1.0])
        jaw_centers = (
            centers[:, None]
            + side[None, :, None] * state.jaw_width[:, None, None] / 2 * rotations[:, None, :, 1]
        )
        normal = -side[None, :, None] * rotations[:, None, :, 1]
        offset = samples[None, None] - jaw_centers[:, :, None, None]
        signed = jnp.sum(offset * normal[:, :, None, None], axis=-1)
        u = jnp.sum(offset * rotations[:, None, None, None, :, 0], axis=-1)
        v = jnp.sum(offset * rotations[:, None, None, None, :, 2], axis=-1)
        facing = jnp.sum(face_normals[None, None] * normal[:, :, None, None], axis=-1) < -0.3
        active = (
            (jnp.abs(u) < self.pad_half_extent)
            & (jnp.abs(v) < self.pad_half_height)
            & (signed < 0)
            & (signed > -0.025)
            & facing
            & exterior[None, None]
        )
        lever = samples[None, None] - centers[:, None, None, None]
        jaw_speed = (
            side[None, :, None] * state.jaw_velocity[:, None, None] / 2 * rotations[:, None, :, 1]
        )
        pad_velocity = (
            tool_velocity[:, None, None, None]
            + jnp.cross(tool_omega[:, None, None, None], lever)
            + jaw_speed[:, :, None, None]
        )
        relative = velocity[None, None] - pad_velocity
        vn = jnp.sum(relative * normal[:, :, None, None], axis=-1)
        area = self.areas[None, None, None, :]
        stiffness = 4e7 * area
        damping = 1.5e4 * area
        load = jnp.where(active, jnp.maximum(-stiffness * signed - damping * vn, 0), 0)
        tangent_velocity = relative - vn[..., None] * normal[:, :, None, None]
        memory = state.contact_memory + dt * tangent_velocity
        memory -= (
            jnp.sum(memory * normal[:, :, None, None], axis=-1)[..., None]
            * normal[:, :, None, None]
        )
        trial = -stiffness[..., None] * memory - damping[..., None] * tangent_velocity
        magnitude = jnp.linalg.norm(trial, axis=-1)
        sliding = magnitude > mu_static[:, None, None, None] * load
        friction = (
            trial
            * jnp.where(
                sliding, mu_dynamic[:, None, None, None] * load / jnp.maximum(magnitude, 1e-12), 1.0
            )[..., None]
        )
        friction = jnp.where((load > 0)[..., None], friction, 0.0)
        memory = jnp.where(
            (load > 0)[..., None],
            -(friction + damping[..., None] * tangent_velocity) / stiffness[..., None],
            0,
        )
        contact_force = load[..., None] * normal[:, :, None, None] + friction
        sample_force = jnp.sum(contact_force, axis=(0, 1))
        node_force = jnp.einsum("sn,csk->cnk", self.weights, sample_force).reshape(-1, 3)
        # Contact with a fixed ground plane, no object pose pinning.
        height = jnp.zeros(samples.shape[:2])
        if supports is not None:
            inside = (
                jnp.abs(samples[:, :, None, 0] - supports[None, None, :, 0])
                <= supports[None, None, :, 2]
            ) & (
                jnp.abs(samples[:, :, None, 1] - supports[None, None, :, 1])
                <= supports[None, None, :, 3]
            )
            height = jnp.maximum(
                height,
                jnp.max(jnp.where(inside, supports[None, None, :, 4], 0.0), axis=2, initial=0.0),
            )
        support_owner = jnp.zeros(samples.shape[:2], dtype=jnp.int32)
        support_count = 1
        if supports is not None:
            heights = jnp.concatenate(
                (
                    jnp.zeros((*samples.shape[:2], 1)),
                    jnp.where(inside, supports[None, None, :, 4], -jnp.inf),
                ),
                axis=2,
            )
            support_owner = jnp.argmax(heights, axis=2)
            support_count = supports.shape[0] + 1
        floor_gap = samples[:, :, 2] - height
        floor_active = (floor_gap < 0) & (face_normals[:, :, 2] < -0.2) & exterior
        floor_load = jnp.where(
            floor_active,
            jnp.maximum(-2e7 * self.areas * floor_gap - 2e4 * self.areas * velocity[:, :, 2], 0),
            0,
        )
        floor_tangent = -velocity[:, :, :2] * self.areas[None, :, None] * 2e4
        floor_tangent *= jnp.minimum(
            1.0, 0.6 * floor_load / jnp.maximum(jnp.linalg.norm(floor_tangent, axis=2), 1e-12)
        )[:, :, None]
        floor_force = jnp.concatenate((floor_tangent, floor_load[:, :, None]), axis=2)
        node_force += jnp.einsum("sn,csk->cnk", self.weights, floor_force).reshape(-1, 3)
        # Small nodal self-contact skin. Internal edges are never projected by this term.
        i, j = self.self_i, self.self_j
        d = state.position[j] - state.position[i]
        distance = jnp.linalg.norm(d, axis=1)
        direction = d / jnp.maximum(distance[:, None], 1e-12)
        closing = jnp.sum((state.velocity[j] - state.velocity[i]) * direction, axis=1)
        self_load = jnp.where(
            distance < 0.003, jnp.maximum(1500 * (0.003 - distance) - 0.03 * closing, 0), 0
        )
        sf = self_load[:, None] * direction
        node_force = node_force.at[i].add(-sf).at[j].add(sf)
        acceleration = gravity + (node_force + external_force) / self.node_mass
        predicted = state.position + dt * state.velocity + dt**2 * acceleration
        predicted = jnp.where(pinned[:, None], state.position, predicted)
        x, _, bond_lambda = self._elastic_project(predicted, state.damage, inverse_mass)
        # Constraint impulses act at material nodes. Breaking adds no velocity kick.
        speed = (x - state.position) / dt
        gap = jnp.linalg.norm(x[self.bonds[:, 1]] - x[self.bonds[:, 0]], axis=1)
        maximum = jnp.maximum(state.maximum_opening, gap)
        onset = mat.tensile_strength * mat.cohesive_length / mat.modulus
        failure = 2 * mat.fracture_energy / mat.tensile_strength
        tensile = jnp.where(
            maximum > onset,
            failure * (maximum - onset) / (jnp.maximum(maximum, 1e-12) * (failure - onset)),
            0,
        )
        pressure = jnp.max(load / area, axis=(0, 1, 3))
        # Only irreversible work supplied during overpressure contributes to crush failure.
        power = jnp.sum(
            jnp.where(load / area > mat.compressive_strength, load * jnp.maximum(-vn, 0), 0),
            axis=(0, 1, 3),
        )
        work = state.crush_work + dt * power
        shares = jnp.full(self.cells, 0.5).at[0].set(1.0).at[-1].set(1.0)
        interface_work = work[:-1] * shares[:-1] + work[1:] * shares[1:]
        crush = jnp.repeat(interface_work / (mat.fracture_energy * self.width**2), 4)
        damage = jnp.maximum(state.damage, jnp.clip(jnp.maximum(tensile, crush), 0, 1))
        close_rate = jnp.where(jnp.asarray(protect) & (state.jaw_width < 0.06), 0.02, 0.15)
        drive = state.drive_width + jnp.clip(
            target_width - state.drive_width, -close_rate * dt, 0.15 * dt
        )
        # Guard the physical jaw drive before the next contact step; loads are never clipped.
        separation = x[self.bonds[:, 1]] - x[self.bonds[:, 0]]
        relative_speed = speed[self.bonds[:, 1]] - speed[self.bonds[:, 0]]
        opening_speed = jnp.maximum(
            jnp.sum(separation * relative_speed, axis=1) / jnp.maximum(gap, 1e-12), 0.0
        )
        risk = jnp.maximum(
            jnp.max((gap + 0.005 * opening_speed) / onset),
            jnp.max(pressure / mat.compressive_strength),
        )
        normal_per_jaw = jnp.sum(load, axis=(2, 3))
        overload = jnp.max(normal_per_jaw, axis=1) > force_ceiling
        warning = jnp.asarray(protect) & (risk > 0.55)
        drive = jnp.where(warning | overload, jnp.minimum(state.jaw_width + 0.0002, 0.10), drive)
        jaw_speed_guarded = jnp.where(
            warning | overload, jnp.maximum(state.jaw_velocity, 0.0), state.jaw_velocity
        )
        motor = jnp.clip(
            20000 * (drive - state.jaw_width) - 40 * jaw_speed_guarded, -force_limit, force_limit
        )
        normal_per_jaw = jnp.sum(load, axis=(2, 3))
        opening = jnp.sum(normal_per_jaw, axis=1) / 2
        jaw_v = jaw_speed_guarded + dt * (motor + opening) / 0.05
        width = jnp.clip(state.jaw_width + dt * jaw_v, 0.004, 0.10)
        jaw_v = jnp.where((width > 0.004) & (width < 0.10), jaw_v, 0)
        following = StructureState(
            x,
            speed,
            damage,
            maximum,
            work,
            jnp.maximum(state.peak_pressure, pressure),
            memory,
            width,
            jaw_v,
            drive,
        )
        lengths = jnp.linalg.norm(x[self.edges[:, :, 1]] - x[self.edges[:, :, 0]], axis=2)
        energy = 0.5 * self.strut_k * jnp.sum(
            (lengths - self.rest_lengths) ** 2
        ) + 0.5 * self.bond_k * jnp.sum((1 - damage) * gap**2)
        report = StructureReport(
            normal_per_jaw,
            jnp.sum(jnp.linalg.norm(friction, axis=-1), axis=(2, 3)),
            -jnp.sum(contact_force, axis=(1, 2, 3)),
            -jnp.sum(jnp.cross(lever, contact_force), axis=(1, 2, 3)),
            pressure,
            jnp.linalg.norm(bond_lambda, axis=1) / (dt**2 * self.bond_area),
            energy,
            jnp.sum(self_load > 0),
            jnp.max(
                jnp.where(load > 0.01, jnp.linalg.norm(tangent_velocity, axis=-1), 0.0), axis=(2, 3)
            ),
            jnp.any(sliding & (load > 0.01), axis=(2, 3)),
            jnp.max(jnp.where(active, -signed, 0.0), axis=(2, 3)),
            risk,
            jnp.zeros(support_count).at[support_owner.reshape(-1)].add(floor_load.reshape(-1)),
        )
        return following, report

    def _elastic_energy(self, state):
        lengths = jnp.linalg.norm(
            state.position[self.edges[:, :, 1]] - state.position[self.edges[:, :, 0]], axis=2
        )
        gaps = state.position[self.bonds[:, 1]] - state.position[self.bonds[:, 0]]
        return 0.5 * self.strut_k * jnp.sum(
            (lengths - self.rest_lengths) ** 2
        ) + 0.5 * self.bond_k * jnp.sum((1 - state.damage) * jnp.sum(gaps**2, axis=1))

    def fragments(self, state):
        # Connectivity is based on any surviving physical bond, not a visual damage label.
        intact = np.any(np.asarray(state.damage).reshape(-1, 4) < 1, axis=1)
        groups = np.r_[0, np.cumsum(~intact)]
        return [
            dict(
                id=int(g),
                cells=np.flatnonzero(groups == g).tolist(),
                mass_kg=float(np.sum(groups == g) * self.material.mass / self.cells),
            )
            for g in np.unique(groups)
        ]

    def summary(self, state, report=None):
        damage = np.asarray(state.damage).reshape(-1, 4)
        centers = np.asarray(state.position).reshape(self.cells, 8, 3).mean(axis=1)
        axis = centers[-1] - centers[0]
        axis /= max(np.linalg.norm(axis), 1e-12)
        residual = centers - centers[0]
        residual -= np.outer(residual @ axis, axis)
        return dict(
            model="xpbd_cohesive_cells_v1",
            material=self.material.name,
            parameters=asdict(self.material),
            dimensions_m=[self.width, self.length, self.width],
            iterations=self.iterations,
            positions=np.asarray(state.position).reshape(self.cells, 8, 3).tolist(),
            velocities=np.asarray(state.velocity).reshape(self.cells, 8, 3).tolist(),
            damage=damage.tolist(),
            triangles=self.triangles.tolist(),
            fragments=self.fragments(state),
            broken_bonds=int(np.sum(damage >= 1)),
            maximum_damage=float(damage.max()),
            bend_m=float(np.linalg.norm(residual, axis=1).max()),
            peak_pressure_pa=float(np.max(state.peak_pressure)),
            crush_work_j=float(np.sum(state.crush_work)),
            jaw_width_m=np.asarray(state.jaw_width).tolist(),
            normal_force_n=np.asarray(report.normal).tolist()
            if report is not None
            else [[0.0, 0.0]] * self.grippers,
            reaction_force_n=np.asarray(report.wrench_force).tolist()
            if report is not None
            else [[0.0, 0.0, 0.0]] * self.grippers,
            elastic_energy_j=float(report.elastic_energy)
            if report is not None
            else float(self.elastic_energy(state)),
        )
