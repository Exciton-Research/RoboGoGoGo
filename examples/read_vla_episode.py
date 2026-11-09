"""Inspect exported VLA data without starting Studio or importing a trainer."""

import argparse

from robogogogo.datasets import iter_transitions

parser = argparse.ArgumentParser(description="Validate and inspect a robogogogo VLA episode")
parser.add_argument("episode", help="Path to an exported episode ZIP")
args = parser.parse_args()
count = 0
for transition in iter_transitions(args.episode):
    if count == 0:
        print("Instruction:", transition.observation.instruction)
        print("First RGB observation:", len(transition.observation.image_png), "PNG bytes")
        print("Action dimensions:", len(transition.action))
        print("Action source:", transition.source)
    count += 1
print("Aligned transitions:", count)
