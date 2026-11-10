"""Training input validation; no trainer or learned policy is implied by a dataset."""

import hashlib
import json
import zipfile
from pathlib import Path

from robogogogo.data.datasets import iter_transitions


def inspect_dataset(path):
    """Validate every aligned transition and report its explicit learning contract.

    This reads the exported archive without extracting files or executing models.
    Simulation archives remain simulation data even if their actions came from a policy.
    """
    path = Path(path)
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        if archive.getinfo("episode.json").file_size > 20_000_000:
            raise ValueError("Episode manifest exceeds size limit")
        manifest = json.loads(archive.read("episode.json"))
    count, sources, shape = 0, set(), None
    for transition in iter_transitions(path):
        signature = (
            len(transition.action),
            tuple(sorted(transition.observation.images)),
            tuple(sorted(transition.next_observation.images)),
        )
        if signature[1] != signature[2] or (shape is not None and signature != shape):
            raise ValueError("Dataset changes action dimensions or camera streams")
        shape = signature
        count += 1
        sources.add(transition.source)
    if not count:
        raise ValueError("Training requires at least one aligned transition")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Episode changed during validation; retry with a stable archive")
    return dict(
        id=manifest["id"],
        sha256=digest,
        transitions=count,
        environment=manifest.get("environment", "unknown"),
        action_spec=manifest["action_spec"],
        cameras=list(shape[1]),
        sources=sorted(sources),
        validation="aligned_transitions",
        physical_deployment_ready=False,
    )
