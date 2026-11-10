"""Episode persistence independent of Studio, HTTP, and simulation ownership."""

import json
from pathlib import Path
from uuid import UUID


class EpisodeStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def archive(self, identifier):
        return self.directory / f"{UUID(str(identifier))}.zip"

    def save(self, episode, *, model, model_arrays):
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.archive(episode.id)
        temporary = path.with_suffix(".tmp")
        try:
            episode.export(temporary, model=model, model_arrays=model_arrays)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        summary = dict(
            id=episode.id,
            instruction=episode.instruction,
            observations=len(episode.observations),
            actions=len(episode.actions),
            success=episode.success,
            created=episode.created,
            environment="simulation",
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, allow_nan=False))
        temporary.replace(path.with_suffix(".json"))
        return summary

    def list(self):
        result = []
        for path in self.directory.glob("*.json"):
            try:
                if path.stat().st_size > 65536:
                    continue
                entry = json.loads(path.read_text())
                if not self.archive(entry["id"]).is_file() or str(entry["id"]) != path.stem:
                    continue
                if not isinstance(entry["created"], str):
                    continue
                result.append(entry)
            except (ValueError, KeyError, TypeError, OSError):
                continue
        return sorted(result, key=lambda entry: entry["created"], reverse=True)
