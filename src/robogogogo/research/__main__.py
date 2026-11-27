"""Inspect a study plan or explicitly run local simulator experiments."""

import argparse
import json
from pathlib import Path

from .design import Study


def main():
    parser = argparse.ArgumentParser(description="Controlled simulation-error research")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name == "run":
            command.add_argument("--output", required=True, type=Path)
    analyze = commands.add_parser("summarize")
    analyze.add_argument("results", type=Path)
    args = parser.parse_args()
    if args.command == "summarize":
        from .runner import summarize

        print(json.dumps(summarize(json.loads(args.results.read_text())), indent=2))
        return
    study = Study.read(args.config)
    if args.command == "plan":
        print(json.dumps(study.plan(), indent=2))
        return
    from .runner import Runner

    Runner(study, args.output).run()


if __name__ == "__main__":
    main()
