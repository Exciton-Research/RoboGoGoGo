"""Electrical netlist for the one RG Mobile Duo embodiment.

Logical terminal names deliberately do not invent vendor cavity numbering.
This validates electrical topology, not a production loom or connector crimp.
"""

from collections import defaultdict, deque

from .components import ARM_COMPONENTS

WIRING_REVISION = "dual-arm-power-can-1"
SOURCE = "https://img.cubemars.com/products/cubemars-product-parameter/AK-Series-Module-Product-Manual-v3.0.0-Download.pdf"


def robot_wiring(model):
    connectors = []
    wires = []
    terminations = []
    devices = []

    def connector(name, terminals, body, role):
        connectors.append(dict(name=name, terminals=terminals, body=body, role=role))

    def wire(a, ap, b, bp, circuit):
        wires.append(dict(source=[a, ap], target=[b, bp], circuit=circuit))

    for side in ("left", "right"):
        distribution = f"{side}_distribution"
        gateway = f"{side}_can_gateway"
        connector(distribution, ["48V+", "48V-", "24V+", "24V-"], "torso", "power_distribution")
        connector(gateway, ["CAN_H", "CAN_L"], "torso", "isolated_can_gateway")
        terminations.append(dict(connector=gateway, resistance_ohm=120))
        previous = gateway
        for address, (suffix, spec) in enumerate(ARM_COMPONENTS.items(), 1):
            name = f"{side}_{suffix}"
            body = model.names.index(name)
            panel = f"{name}_panel"
            drive = f"{name}_drive"
            host = model.names[model.parents[body]]
            connector(panel, ["V+", "V-", "CAN_H", "CAN_L"], host, "panel")
            connector(drive, ["V+", "V-", "CAN_H", "CAN_L"], host, "drive")
            # Each voltage rail has an individual branch to the panel; no wrist
            # gets the proximal 48 V rail. Protection ratings are a separate BOM.
            for polarity in ("+", "-"):
                wire(distribution, f"{spec.voltage}V{polarity}", panel, f"V{polarity}", "power")
                wire(panel, f"V{polarity}", drive, f"V{polarity}", "fixed_pigtail")
            for signal in ("CAN_H", "CAN_L"):
                wire(previous, signal, panel, signal, "can_trunk")
                wire(panel, signal, drive, signal, "fixed_pigtail")
            previous = panel
            devices.append(
                dict(
                    name=name,
                    connector=drive,
                    panel=panel,
                    voltage=spec.voltage,
                    can_bus=side,
                    can_id=address,
                    model=spec.model,
                    panel_body=host,
                    joint_index=model.indices[body],
                )
            )
        terminations.append(dict(connector=previous, resistance_ohm=120))
    result = dict(
        revision=WIRING_REVISION,
        connectors=connectors,
        wires=wires,
        devices=devices,
        terminations=terminations,
        source=SOURCE,
        qualification=dict(
            logical_topology="validated by connectivity checks",
            physical_pin_numbering="verify device revision and terminal legends",
            branch_protection="hardware BOM and bus-current measurements required",
            complete_3d_branch_routing=False,
        ),
    )
    errors = validate_wiring(result)
    if errors:
        raise ValueError("Invalid canonical robot wiring: " + "; ".join(errors))
    return result


def validate_wiring(wiring):
    """Detect opens, shorts, reversed CAN, missing terminations and duplicate IDs."""
    errors = []
    nodes = {}
    for connector in wiring["connectors"]:
        if connector["name"] in nodes:
            errors.append(f"Duplicate connector: {connector['name']}")
        nodes[connector["name"]] = set(connector["terminals"])
    graph = defaultdict(set)
    for wire in wiring["wires"]:
        a, b = tuple(wire["source"]), tuple(wire["target"])
        if any(name not in nodes or pin not in nodes[name] for name, pin in (a, b)):
            errors.append(f"Unknown wire endpoint: {a} -> {b}")
            continue
        graph[a].add(b)
        graph[b].add(a)

    def reachable(start):
        visited = {start}
        queue = deque([start])
        while queue:
            for node in graph[queue.popleft()]:
                if node not in visited:
                    visited.add(node)
                    queue.append(node)
        return visited

    expected_devices = {f"{side}_{suffix}" for side in ("left", "right") for suffix in ARM_COMPONENTS}
    names = [device["name"] for device in wiring["devices"]]
    if set(names) != expected_devices or len(names) != len(expected_devices):
        errors.append("Expected exactly the fourteen arm drives")
    addresses = set()
    for device in wiring["devices"]:
        key = (device["can_bus"], device["can_id"])
        if key in addresses:
            errors.append(f"Duplicate CAN address: {key}")
        addresses.add(key)
        side = device["can_bus"]
        drive = device["connector"]
        suffix = device["name"].removeprefix(f"{side}_")
        if suffix in ARM_COMPONENTS and device["voltage"] != ARM_COMPONENTS[suffix].voltage:
            errors.append(f"Incorrect catalog supply voltage: {device['name']}")
        if nodes.get(drive) != {"V+", "V-", "CAN_H", "CAN_L"}:
            errors.append(f"Missing or unexpected drive terminals: {drive}")
        expected = {
            "V+": (f"{side}_distribution", f"{device['voltage']}V+"),
            "V-": (f"{side}_distribution", f"{device['voltage']}V-"),
            "CAN_H": (f"{side}_can_gateway", "CAN_H"),
            "CAN_L": (f"{side}_can_gateway", "CAN_L"),
        }
        for pin, source in expected.items():
            connected = reachable((drive, pin))
            if source not in connected:
                errors.append(f"Open or miswired {drive}:{pin}")
            sources = {
                node for node in connected if node[0].endswith(("_distribution", "_can_gateway"))
            }
            if sources - {source}:
                errors.append(f"Short or crossed rail at {drive}:{pin}")
    for side in ("left", "right"):
        bus = reachable((f"{side}_can_gateway", "CAN_H"))
        ends = [t for t in wiring["terminations"] if (t["connector"], "CAN_H") in bus]
        if len(ends) != 2 or any(t["resistance_ohm"] != 120 for t in ends):
            errors.append(f"{side} CAN requires two 120 ohm terminations")
        expected = {f"{side}_can_gateway", f"{side}_tool_roll_panel"}
        if {t["connector"] for t in ends} != expected:
            errors.append(f"{side} CAN terminations must be at bus ends")
    return errors
