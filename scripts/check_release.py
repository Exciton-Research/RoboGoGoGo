"""Inspect built Studio wheels/sdists for legal files, assets, and private-file leaks.

Run after `npm run build --prefix studio` and `python -m build`.
No archive extraction, network access, or publication occurs here.
"""

import argparse
import tarfile
import tomllib
import zipfile
from email import message_from_bytes
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LEGAL = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")
FORBIDDEN = {
    ".git",
    ".aws",
    ".ssh",
    ".gnupg",
    ".venv",
    "node_modules",
    ".next",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "runs",
    ".DS_Store",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_names(names):
    for name in names:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, f"Unsafe archive path: {name}")
        require(not (set(path.parts) & FORBIDDEN), f"Local/private directory in release: {name}")
        require(
            not (path.name == ".env" or path.name.startswith(".env."))
            or path.name.endswith(".example"),
            f"Local environment file in release: {name}",
        )
        require(
            path.suffix
            not in (".pyc", ".p12", ".pfx", ".key", ".pem", ".node", ".so", ".dylib", ".dll"),
            f"Unexpected private, cache, or native binary file: {name}",
        )


def check_legal(names, read, prefix):
    for filename in LEGAL:
        name = prefix + filename
        require(name in names, f"Missing legal file: {name}")
        require(read(name) == (ROOT / filename).read_bytes(), f"Stale legal file: {name}")


def check_metadata(data):
    metadata = message_from_bytes(data)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    require(metadata["Name"] == project["name"], "Unexpected distribution name")
    require(metadata["Version"] == project["version"], "Stale distribution version")
    require(metadata["License-Expression"] == "Apache-2.0", "Missing Apache-2.0 metadata")
    require(
        set(LEGAL).issubset(metadata.get_all("License-File", [])), "Missing license-file metadata"
    )


def check_web(names, read, prefix):
    for page in (
        "index.html",
        "simulation/index.html",
        "hardware/index.html",
        "execution/index.html",
    ):
        require(prefix + page in names, f"Missing Studio page: {prefix}{page}")
    check_legal(names, read, prefix)
    for family in ("dm-sans", "ibm-plex-mono"):
        require(
            any(
                n.startswith(prefix + "_next/static/media/")
                and family in n
                and n.endswith(".woff2")
                for n in names
            ),
            f"Missing bundled font: {family}",
        )


def check_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        check_names(names)
        metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata_files) == 1, "Wheel requires exactly one METADATA file")
        metadata_name = metadata_files[0]
        check_metadata(archive.read(metadata_name))
        check_legal(names, archive.read, metadata_name.removesuffix("METADATA") + "licenses/")
        check_web(names, archive.read, "robogogogo/web/")


def check_sdist(path):
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            require(
                not member.issym() and not member.islnk(), f"Link in source release: {member.name}"
            )
            require(
                ".." not in PurePosixPath(member.name).parts, f"Unsafe source path: {member.name}"
            )
            if member.isfile():
                _, separator, name = member.name.partition("/")
                require(bool(separator), "Source files require a top-level directory")
                files[name] = member
        names = set(files)
        check_names(names)

        def read(name):
            stream = archive.extractfile(files[name])
            require(stream is not None, f"Unreadable source file: {name}")
            return stream.read()

        check_legal(names, read, "")
        require("PKG-INFO" in names, "Source release is missing PKG-INFO")
        check_metadata(read("PKG-INFO"))
        check_web(names, read, "src/robogogogo/web/")
        for name in (
            "README.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "scripts/setup.sh",
            "scripts/check_release.py",
            "studio/src/App.tsx",
            "studio/package-lock.json",
            "studio/scripts/generate-notices.mjs",
            "docs/installation.md",
            "docs/getting-started.md",
        ):
            require(name in names, f"Missing reproducible source/documentation: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    wheels, sdists = sorted(args.directory.glob("*.whl")), sorted(args.directory.glob("*.tar.gz"))
    try:
        require(
            bool(wheels) and bool(sdists),
            "Build a wheel and source archive before checking the release",
        )
        for path, check in [(p, check_wheel) for p in wheels] + [(p, check_sdist) for p in sdists]:
            check(path)
            print(
                f"Validated {path.name}: license metadata, legal notices, Studio assets, and file inventory"
            )
    except (ValueError, KeyError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        parser.exit(1, f"Release check failed: {error}\n")


if __name__ == "__main__":
    main()
