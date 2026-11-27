"""Equal-rollout comparisons with selection separated from held-out evaluation."""

import importlib
import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .design import REFERENCE, STRATEGIES, seed_for
from .robosuite_backend import RobosuiteBackend


def write_json(path, value):
    """Replace a report atomically; interrupted runs never look completed."""
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def load_callable(specification):
    module, name = specification.split(":")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise ValueError("module:function must name a callable")
    return function


def release_policy(policy):
    """Allow external GPU policies to release resources between fits/evaluations."""
    close = getattr(policy, "close", None)
    if callable(close):
        close()


class Runner:
    def __init__(self, study, output):
        self.study, self.output = study, Path(output)
        # Refuse to overwrite completed or interrupted studies.
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "plan.json", study.plan())
        self.trainer = load_callable(study.trainer)
        self.loader = load_callable(study.loader)
        self.backend = RobosuiteBackend(study)
        write_json(
            self.output / "backend.json",
            {
                **self.backend.metadata(),
                "python": platform.python_version(),
            },
        )
        self.events = self.output / "events.jsonl"
        self.paired_initial_states = {}

    def event(self, kind, **payload):
        with self.events.open("a") as stream:
            stream.write(json.dumps({"kind": kind, **payload}, allow_nan=False) + "\n")
        if kind in ("strategy_started", "selection", "strategy_finished", "completed"):
            print(json.dumps({"kind": kind, **payload}), flush=True)

    def rollout(self, profile, seed, **kwargs):
        result = self.backend.rollout(profile, seed, **kwargs)
        state = result["initial_state_sha256"]
        if seed in self.paired_initial_states and self.paired_initial_states[seed] != state:
            raise RuntimeError("Matched episode seed produced different initial states")
        self.paired_initial_states[seed] = state
        return result

    def train(self, episodes, directory, seed, *, phase, strategy):
        directory.mkdir()
        started = perf_counter()
        policy = self.trainer(episodes, output_dir=directory, seed=seed)
        if not callable(getattr(policy, "reset", None)) or not callable(
            getattr(policy, "act", None)
        ):
            raise TypeError("Trainer must return a policy with reset() and act()")
        seconds = perf_counter() - started
        self.event(
            "training",
            strategy=strategy,
            seed=seed,
            phase=phase,
            episodes=len(episodes),
            wall_seconds=seconds,
        )
        return policy, seconds

    def fit_strategy(self, strategy, seed):
        study = self.study
        directory = self.output / f"seed-{seed}" / strategy
        directory.mkdir(parents=True)
        data_dir = directory / "episodes"
        data_dir.mkdir()
        episodes, records, selection = [], [], []
        training_seconds = 0.0
        self.event("strategy_started", strategy=strategy, seed=seed)

        def collect(profile, task_index):
            path = data_dir / f"{len(episodes):05d}.npz"
            result = self.rollout(
                profile, seed_for(study.name, "train", seed, task_index), save=path
            )
            episodes.append(path)
            records.append(result)
            self.event("demonstration", strategy=strategy, training_seed=seed, **result)

        chosen = None
        if strategy == "guided":
            # Paired pilot trajectories across the biased and one-axis-refined
            # candidates. Each candidate is trained without other candidates'
            # data, then evaluated on the same selection initial conditions.
            candidates = [()] + [(axis,) for axis in study.axes]
            for index, refined in enumerate(candidates):
                first = len(episodes)
                for task_index in range(study.pilot_rollouts):
                    collect(study.profile(refined), task_index)
                policy, seconds = self.train(
                    episodes[first:],
                    directory / f"pilot-{index}",
                    seed_for(study.name, "pilot-policy", seed),
                    phase=f"pilot-{index}",
                    strategy=strategy,
                )
                training_seconds += seconds
                outcomes = []
                for task_index in range(study.selection_rollouts):
                    result = self.rollout(
                        dict(REFERENCE),
                        seed_for(study.name, "selection", seed, task_index),
                        policy=policy,
                    )
                    outcomes.append(result)
                    self.event(
                        "selection_rollout",
                        strategy=strategy,
                        training_seed=seed,
                        candidate=list(refined),
                        **result,
                    )
                selection.append(
                    {
                        "refined": list(refined),
                        "outcomes": outcomes,
                        "success_rate": sum(row["success"] for row in outcomes) / len(outcomes),
                    }
                )
                release_policy(policy)
                del policy
            # Deterministic ties prefer the biased candidate, then declared axis
            # order. No held-out score or future evaluation is consulted.
            winner = max(range(len(selection)), key=lambda i: selection[i]["success_rate"])
            chosen = candidates[winner]
            self.event(
                "selection",
                seed=seed,
                refined=list(chosen),
                scores=[row["success_rate"] for row in selection],
            )
            remaining = study.budget_rollouts - study.guided_overhead
            for index in range(remaining):
                collect(study.profile(chosen), study.pilot_rollouts + index)
        else:
            if strategy == "random":
                index = seed_for(study.name, "random-axis", seed) % len(study.axes)
                chosen = (study.axes[index],)
            for index in range(study.budget_rollouts):
                if strategy == "reference":
                    refined = study.axes
                elif strategy == "uniform":
                    refined = (study.axes[index % len(study.axes)],)
                elif strategy == "random":
                    refined = chosen
                else:
                    refined = ()
                collect(study.profile(refined), index)
        selection_count = sum(len(row["outcomes"]) for row in selection)
        used = len(episodes) + selection_count
        if used != study.budget_rollouts:
            raise RuntimeError(f"Budget mismatch: used {used}, expected {study.budget_rollouts}")
        policy, seconds = self.train(
            episodes,
            directory / "policy",
            seed_for(study.name, "final-policy", seed),
            phase="final",
            strategy=strategy,
        )
        training_seconds += seconds
        release_policy(policy)
        del policy
        report = {
            "strategy": strategy,
            "seed": seed,
            "selected_refinement": None if chosen is None else list(chosen),
            "selection": selection,
            "demonstrations": records,
            "budget": {
                "collection_rollouts": len(records),
                "selection_rollouts": selection_count,
                "total_rollouts": used,
                "control_steps": used * study.horizon,
                "training_wall_seconds": training_seconds,
                "simulation_wall_seconds": sum(row["wall_seconds"] for row in records)
                + sum(
                    row["wall_seconds"] for candidate in selection for row in candidate["outcomes"]
                ),
                "training_fits": len(selection) + 1,
            },
        }
        write_json(directory / "fit.json", report)
        return report

    def run(self):
        # Complete every selection/training decision before exposing any held-out
        # outcomes. Reload one checkpoint at a time so external VLA backends do
        # not need to keep every trained model resident in GPU memory.
        fitted = []
        for seed in self.study.seeds:
            for strategy in STRATEGIES:
                fitted.append(self.fit_strategy(strategy, seed))
        reports = []
        for report in fitted:
            directory = self.output / f"seed-{report['seed']}" / report["strategy"]
            policy = self.loader(directory / "policy")
            outcomes = []
            try:
                for task_index in range(self.study.evaluation_rollouts):
                    result = self.rollout(
                        dict(REFERENCE),
                        seed_for(self.study.name, "held-out", task_index),
                        policy=policy,
                    )
                    outcomes.append(result)
                    self.event(
                        "held_out",
                        strategy=report["strategy"],
                        training_seed=report["seed"],
                        **result,
                    )
            finally:
                release_policy(policy)
                del policy
            report["evaluation"] = outcomes
            write_json(
                self.output / f"seed-{report['seed']}" / report["strategy"] / "result.json", report
            )
            reports.append(report)
            self.event(
                "strategy_finished",
                strategy=report["strategy"],
                seed=report["seed"],
                successes=sum(row["success"] for row in outcomes),
                episodes=len(outcomes),
            )
        result = {"schema": 1, "status": "complete", "study": self.study.to_dict(), "runs": reports}
        write_json(self.output / "results.json", result)
        write_json(self.output / "summary.json", summarize(result))
        self.event("completed", output=str(self.output))
        return result


def summarize(result):
    """Paired differences; crossed bootstrap over training seeds and test cases.

    Shared test initial conditions are not treated as independent repetitions.
    With one training seed no confidence interval is reported.
    """
    if result.get("schema") != 1 or result.get("status") != "complete":
        raise ValueError("Only completed schema-1 studies can be summarized")
    seeds = result["study"]["seeds"]
    count = result["study"]["evaluation_rollouts"]
    matrices, summaries = {}, {}
    runs = result["runs"]
    expected_ids = {(strategy, seed) for strategy in STRATEGIES for seed in seeds}
    actual_ids = [(run["strategy"], run["seed"]) for run in runs]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise ValueError("Missing or duplicate strategy/seed results")
    expected_tasks = [seed_for(result["study"]["name"], "held-out", i) for i in range(count)]
    for strategy in STRATEGIES:
        selected = [
            next(row for row in runs if row["strategy"] == strategy and row["seed"] == seed)
            for seed in seeds
        ]
        if any([row["seed"] for row in run["evaluation"]] != expected_tasks for run in selected):
            raise ValueError("Held-out tasks must be paired and in identical order")
        matrix = np.asarray(
            [[float(row["success"]) for row in run["evaluation"]] for run in selected]
        )
        matrices[strategy] = matrix
        summaries[strategy] = {
            "success_rate": float(matrix.mean()),
            "success_rate_by_training_seed": matrix.mean(axis=1).tolist(),
            "failures": dict(
                Counter(
                    row["failure"]
                    for run in selected
                    for row in run["evaluation"]
                    if not row["success"]
                )
            ),
            "mean_training_wall_seconds": float(
                np.mean([run["budget"]["training_wall_seconds"] for run in selected])
            ),
            "mean_simulation_wall_seconds": float(
                np.mean([run["budget"]["simulation_wall_seconds"] for run in selected])
            ),
            "mean_demonstrator_success_rate": float(
                np.mean(
                    [np.mean([row["success"] for row in run["demonstrations"]]) for run in selected]
                )
            ),
        }
    comparisons = {}
    for baseline in (strategy for strategy in STRATEGIES if strategy != "guided"):
        difference = matrices["guided"] - matrices[baseline]
        interval = None
        if len(seeds) >= 2 and count >= 2:
            rng = np.random.default_rng(seed_for("paired-bootstrap", baseline))
            means = []
            for _ in range(2000):
                rows = rng.integers(len(seeds), size=len(seeds))
                columns = rng.integers(count, size=count)
                means.append(difference[np.ix_(rows, columns)].mean())
            interval = np.quantile(means, [0.025, 0.975]).tolist()
        comparisons[f"guided_minus_{baseline}"] = {
            "success_rate_difference": float(difference.mean()),
            "bootstrap_95_percent_interval": interval,
        }
    return {
        "scope": "Panda Lift sim-to-sim; nominal simulation is not physical ground truth",
        "strategies": summaries,
        "paired_comparisons": comparisons,
        "uncertainty": "Crossed bootstrap over training seeds and shared held-out initial conditions. "
        "Exploratory intervals; few seeds and one task cannot establish generality.",
        "budget": "Equal collection + selection control steps; training compute reported separately.",
    }
