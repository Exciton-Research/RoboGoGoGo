"""Check tracked public content without printing potentially sensitive matches."""

import io
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Add a repository only after reviewing whether its link belongs in public content.
PUBLIC_REPOSITORIES = {
    "david-oconnor/pyflow",
    "exciton-research/robogogogo",
    "github/gitignore",
    "google-deepmind/mujoco_menagerie",
    "googlefonts/dm-fonts",
    "joyent/node",
    "json5/json5",
    "lodash/lodash",
    "pdm-project/pdm",
    "prettier/prettier",
    "rich-harris/devalue",
    "ros2/common_interfaces",
    "ros2/message_filters",
}
REPOSITORY = re.compile(rb"(?<![\w.-])github[.]com[:/]([\w.-]+)/([\w.-]+)", re.I)
SENSITIVE = {
    "personal absolute path": re.compile(
        rb"/(?:home|Users)/[^/\s\"'<>]+/|[A-Za-z]:\\Users\\[^\\\s]+\\"
    ),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credential": re.compile(
        rb"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{36,}"
        rb"|github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-)?[A-Za-z0-9_-]{40,}"
    ),
}


def content_issues(data):
    issues = {label for label, pattern in SENSITIVE.items() if pattern.search(data)}
    for owner, repository in REPOSITORY.findall(data):
        owner, repository = owner.decode().lower(), repository.decode().lower()
        if owner == "sponsors":
            continue
        repository = repository.removesuffix(".git")
        if f"{owner}/{repository}" not in PUBLIC_REPOSITORIES:
            issues.add("unreviewed GitHub repository link")
    return sorted(issues)


def file_issues(name, data):
    issues = [f"{name}: {reason}" for reason in content_issues(data)]
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                for reason in content_issues(archive.read(member)):
                    issues.append(f"{name}: embedded archive content: {reason}")
    return issues


def main():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    names = [name.decode() for name in tracked.split(b"\0") if name]
    issues = []
    for name in names:
        issues.extend(file_issues(name, (ROOT / name).read_bytes()))
    if issues:
        print("Public content check failed (matched values are withheld):")
        print("\n".join(issues))
        return 1
    print(f"Checked {len(names)} tracked files and embedded ZIP contents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
