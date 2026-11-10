"""Named arm groups derived from articulated-tree tool frames."""

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArmGroup:
    name: str
    tool: str
    body: int
    joint_indices: tuple[int, ...]
    joint_names: tuple[str, ...]
    root: str | None = None

    def to_dict(self):
        return asdict(self)


def resolve_arms(model, definitions=None) -> tuple[ArmGroup, ...]:
    """Resolve explicit tool links, or infer one group per moving terminal branch.

    Joint membership follows ancestry, including shared torso joints. For robots
    with fingers or non-arm branches, callers should supply explicit tool links.
    """
    inferred = definitions is None
    if definitions is None:
        parents = set(model.parents)
        leaves = [i for i in range(len(model.names)) if i not in parents]
        definitions = [dict(name=model.names[i], tool=model.names[i]) for i in leaves]
    groups = []
    names, tools = set(), set()
    for definition in definitions:
        name, tool = definition["name"], definition["tool"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,79}", name):
            raise ValueError("Arm names must be safe identifiers of at most 80 characters")
        if name in names or tool in tools:
            raise ValueError("Arm names and tool frames must be unique")
        if tool not in model.names:
            raise ValueError(f"Unknown tool frame: {tool}")
        root = definition.get("root")
        if root is not None and root not in model.names:
            raise ValueError(f"Unknown arm root: {root}")
        boundary = model.names.index(root) if root is not None else -1
        body = model.names.index(tool)
        indices, joint_names = [], []
        cursor = body
        while cursor >= 0 and cursor != boundary:
            if model.indices[cursor] >= 0:
                indices.append(model.indices[cursor])
                joint_names.append(model.names[cursor])
            cursor = model.parents[cursor]
        if root is not None and cursor != boundary:
            raise ValueError("Arm root must be an ancestor of its tool")
        if not indices and inferred:
            continue
        if not indices:
            raise ValueError(f"Tool frame {tool} has no movable ancestor")
        groups.append(
            ArmGroup(name, tool, body, tuple(reversed(indices)), tuple(reversed(joint_names)), root)
        )
        names.add(name)
        tools.add(tool)
    if not groups:
        raise ValueError("At least one arm tool frame is required")
    return tuple(groups)
